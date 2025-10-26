"""Точка входа FastAPI-приложения."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI

from backend.api.routes import cameras, events, health, stats
from backend.core.app import create_app
from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.models import Camera
from backend.services.cleanup import cleanup_loop, perform_cleanup

app = create_app()
app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(stats.router)


async def _init_default_cameras(session_factory: SessionFactory) -> None:
    entries = settings.iter_rtsp_sources()
    if not entries:
        return

    with session_factory() as session:
        for name, url in entries:
            exists = session.query(Camera).filter(Camera.name == name).first()
            if exists:
                continue
            session.add(Camera(name=name, rtsp_url=url))
        session.commit()


@app.on_event("startup")
async def startup_event() -> None:
    session_factory = app.state.session_factory
    ingest = app.state.ingest_manager

    await _init_default_cameras(session_factory)

    main_loop = asyncio.get_running_loop()
    ingest.set_main_loop(main_loop)
    await ingest.start_all()

    # Первый запуск очистки делаем вручную, чтобы не ждать таймер
    async with app.state.cleanup_lock:
        await perform_cleanup(session_factory, app.state.retention_days, app.state.cleanup_state)

    task = asyncio.create_task(
        cleanup_loop(app, app.state.cleanup_interval_hours),
        name="retention-cleanup",
    )
    app.state.background_tasks.append(task)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    ingest = app.state.ingest_manager
    ingest.stop_all()
    for task in app.state.background_tasks:
        task.cancel()
    for task in app.state.background_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass


def get_app() -> FastAPI:
    """Совместимость с ASGI-серверами, ожидающими фабрику приложения."""
    return app
