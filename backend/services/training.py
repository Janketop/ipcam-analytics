"""Сервис фонового запуска самообучения модели."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI

from backend.core.logger import logger


class SelfTrainingService:
    """Управляет жизненным циклом фоновой задачи дообучения модели."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    async def start_training(self, app: FastAPI) -> bool:
        """Запускает обучение, если оно ещё не выполняется.

        Возвращает ``True``, если задача была поставлена в очередь прямо сейчас.
        Если обучение уже идёт, возвращает ``False``.
        """

        async with self._lock:
            if self._task and not self._task.done():
                return False

            loop = asyncio.get_running_loop()
            task = loop.create_task(self._run_training(app), name="self-training")
            self._task = task
            app.state.background_tasks.append(task)
            return True

    async def _run_training(self, app: FastAPI) -> None:
        """Реализует фактический запуск обучения в отдельном потоке."""

        from backend.train_self import main as run_training

        logger.info("Старт фонового самообучения модели")
        try:
            await asyncio.to_thread(run_training)
            logger.info("Самообучение модели завершено")
        except Exception:
            logger.exception("Самообучение модели завершилось ошибкой")
        finally:
            async with self._lock:
                self._task = None
            current = asyncio.current_task()
            if current is not None and current in app.state.background_tasks:
                app.state.background_tasks.remove(current)  # type: ignore[arg-type]

    def is_running(self) -> bool:
        """Возвращает ``True``, если обучение ещё выполняется."""

        task = self._task
        return bool(task and not task.done())
