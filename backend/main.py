"""Точка входа FastAPI-приложения."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routes import cameras, events, health, stats, statuses, training
from backend.core.logger import LOGGING_CONFIG, logger
from backend.core.app import create_app
from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.core.migrations import run_startup_migrations
from backend.models import Camera
from backend.services.cleanup import cleanup_loop, perform_cleanup

app = create_app()
app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(statuses.router)
app.include_router(stats.router)
app.include_router(training.router)


async def _init_default_cameras(session_factory: SessionFactory) -> None:
    entries = settings.iter_rtsp_sources()
    if not entries:
        logger.info("В настройках не указаны камеры для авто-добавления")
        return

    with session_factory() as session:
        created: list[str] = []
        for name, url in entries:
            exists = session.query(Camera).filter(Camera.name == name).first()
            if exists:
                continue
            session.add(Camera(name=name, rtsp_url=url))
            created.append(name)
        session.commit()

    if created:
        logger.info(
            "Добавлены камеры из настроек: %s",
            ", ".join(created),
        )
    else:
        logger.info("Камеры из настроек уже присутствуют в базе")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения: инициализация и корректное завершение сервисов."""

    session_factory = app.state.session_factory
    ingest = app.state.ingest_manager

    run_startup_migrations(session_factory)
    await _init_default_cameras(session_factory)

    main_loop = asyncio.get_running_loop()
    ingest.set_main_loop(main_loop)
    logger.info("Запускаю ingest-воркеры")
    await ingest.start_all()
    logger.info("Все ingest-воркеры запущены")

    # Первый запуск очистки делаем вручную, чтобы не ждать таймер
    async with app.state.cleanup_lock:
        logger.info("Запускаю первичную очистку устаревших данных")
        await perform_cleanup(session_factory, app.state.retention_days, app.state.cleanup_state)
        logger.info("Первичная очистка завершена")

    task = asyncio.create_task(
        cleanup_loop(app, app.state.cleanup_interval_hours),
        name="retention-cleanup",
    )
    app.state.background_tasks.append(task)
    logger.info("Запущена фоновая задача очистки старых данных")

    try:
        yield
    finally:
        ingest = app.state.ingest_manager
        logger.info("Останавливаю ingest-воркеры")
        ingest.stop_all()
        logger.info("Остановка ingest-воркеров инициирована, ожидаю фоновые задачи")

        tasks = list(app.state.background_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                task_name = task.get_name() if hasattr(task, "get_name") else repr(task)
                logger.warning("Фоновая задача %s остановлена по сигналу отмены", task_name)
            except Exception:
                task_name = task.get_name() if hasattr(task, "get_name") else repr(task)
                logger.exception("Фоновая задача %s завершилась с ошибкой", task_name)

        app.state.background_tasks.clear()


app.router.lifespan_context = lifespan


def get_app() -> FastAPI:
    """Совместимость с ASGI-серверами, ожидающими фабрику приложения."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=LOGGING_CONFIG,
    )
