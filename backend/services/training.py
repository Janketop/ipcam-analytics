"""Сервисы фонового запуска задач обучения моделей."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from backend.core.config import settings
from backend.core.logger import logger


@dataclass(slots=True)
class FaceTrainingOptions:
    """Набор параметров для обучения детектора лиц."""

    dataset_root: Path
    skip_download: bool
    epochs: int
    batch: int
    imgsz: int
    device: str | None
    project: Path
    run_name: str
    base_weights: Path
    output_weights: Path


def build_face_training_options(
    *,
    dataset_root: str | Path | None = None,
    skip_download: bool | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    project: str | Path | None = None,
    run_name: str | None = None,
    base_weights: str | Path | None = None,
    output_weights: str | Path | None = None,
) -> FaceTrainingOptions:
    """Собирает итоговые параметры обучения из входных данных и настроек."""

    dataset_root_path = (
        settings.resolve_project_path(dataset_root)
        if dataset_root is not None
        else settings.face_training_dataset_root_path
    )
    project_path = (
        settings.resolve_project_path(project)
        if project is not None
        else settings.face_training_project_dir_path
    )
    output_weights_path = (
        settings.resolve_project_path(output_weights)
        if output_weights is not None
        else settings.face_training_output_weights_path
    )
    base_weights_path = (
        settings.resolve_project_path(base_weights)
        if base_weights is not None
        else settings.face_training_base_weights_path
    )

    options = FaceTrainingOptions(
        dataset_root=dataset_root_path,
        skip_download=skip_download if skip_download is not None else settings.face_training_skip_download,
        epochs=epochs if epochs is not None else settings.face_training_epochs,
        batch=batch if batch is not None else settings.face_training_batch,
        imgsz=imgsz if imgsz is not None else settings.face_training_imgsz,
        device=device if device is not None else settings.face_training_device,
        project=project_path,
        run_name=run_name if run_name is not None else settings.face_training_run_name,
        base_weights=base_weights_path,
        output_weights=output_weights_path,
    )
    return options


def _run_face_training(options: FaceTrainingOptions) -> Path:
    """Запускает обучение детектора лиц синхронно."""

    from backend.utils import train_face_detector as trainer

    options.dataset_root.mkdir(parents=True, exist_ok=True)
    options.project.mkdir(parents=True, exist_ok=True)
    options.output_weights.parent.mkdir(parents=True, exist_ok=True)

    dataset = trainer.prepare_widerface_dataset(
        options.dataset_root,
        download=not options.skip_download,
    )
    result = trainer.train(
        options.base_weights,
        dataset=dataset,
        epochs=options.epochs,
        batch=options.batch,
        imgsz=options.imgsz,
        device=options.device,
        project=options.project,
        name=options.run_name,
        output_weights=options.output_weights,
    )
    return result


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

            task = asyncio.create_task(self._run_training(app), name="self-training")
            self._task = task

            background_tasks = getattr(app.state, "background_tasks", None)
            if background_tasks is None:
                background_tasks = []
                app.state.background_tasks = background_tasks  # type: ignore[attr-defined]
            background_tasks.append(task)
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
            background_tasks = getattr(app.state, "background_tasks", None)
            current = asyncio.current_task()
            if (
                background_tasks is not None
                and current is not None
                and current in background_tasks
            ):
                background_tasks.remove(current)

    def is_running(self) -> bool:
        """Возвращает ``True``, если обучение ещё выполняется."""

        task = self._task
        return bool(task and not task.done())


class FaceDetectorTrainingService:
    """Управляет запуском обучения детектора лиц в фоне."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    async def start_training(self, app: FastAPI, options: FaceTrainingOptions) -> bool:
        """Ставит обучение в очередь, если нет активной задачи."""

        async with self._lock:
            if self._task and not self._task.done():
                return False

            task = asyncio.create_task(
                self._run_training(app, options),
                name="face-detector-training",
            )
            self._task = task

            background_tasks = getattr(app.state, "background_tasks", None)
            if background_tasks is None:
                background_tasks = []
                app.state.background_tasks = background_tasks  # type: ignore[attr-defined]
            background_tasks.append(task)
            return True

    async def _run_training(self, app: FastAPI, options: FaceTrainingOptions) -> None:
        """Запускает обучение в отдельном потоке."""

        logger.info(
            "Запуск обучения детектора лиц: dataset=%s, epochs=%d, batch=%d, imgsz=%d, device=%s",
            options.dataset_root,
            options.epochs,
            options.batch,
            options.imgsz,
            options.device or "auto",
        )
        try:
            await asyncio.to_thread(_run_face_training, options)
            logger.info("Обучение детектора лиц успешно завершено")
        except Exception:
            logger.exception("Обучение детектора лиц завершилось ошибкой")
        finally:
            async with self._lock:
                self._task = None
            background_tasks = getattr(app.state, "background_tasks", None)
            current = asyncio.current_task()
            if (
                background_tasks is not None
                and current is not None
                and current in background_tasks
            ):
                background_tasks.remove(current)

    def is_running(self) -> bool:
        """Показывает, запущено ли обучение детектора лиц."""

        task = self._task
        return bool(task and not task.done())
