"""Менеджер потоков захвата и обработки видеопотоков."""
from __future__ import annotations

from typing import Awaitable, Callable, List, Optional

from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.core.logger import logger
from backend.models import Camera
from backend.services.ingest_worker import IngestWorker


class IngestManager:
    def __init__(self, session_factory: SessionFactory, main_loop=None) -> None:
        self.session_factory = session_factory
        self.workers: List[IngestWorker] = []
        self.broadcaster: Optional[Callable[[dict], Awaitable[None]]] = None
        self.main_loop = main_loop

    def set_broadcaster(self, fn: Callable[[dict], Awaitable[None]]) -> None:
        self.broadcaster = fn
        for worker in self.workers:
            worker.set_broadcaster(fn)

    def set_main_loop(self, loop) -> None:
        self.main_loop = loop
        for worker in self.workers:
            worker.set_main_loop(loop)

    async def start_all(self) -> None:
        with self.session_factory() as session:
            cameras = (
                session.query(Camera)
                .filter(Camera.active.is_(True))
                .order_by(Camera.id)
                .all()
            )

        face_blur = settings.face_blur
        logger.info("Найдено %d активных камер для запуска", len(cameras))
        for camera in cameras:
            self.start_worker_for_camera(camera, face_blur=face_blur)

    def stop_all(self) -> None:
        if not self.workers:
            logger.info("Нет активных ingest-воркеров для остановки")
        for worker in self.workers:
            worker.stop_flag = True
            logger.info("Отправлен сигнал остановки ingest-воркеру '%s'", worker.name)
        logger.info("Все ingest-воркеры получили сигнал остановки")

    def get_worker(self, name: str):
        for worker in self.workers:
            if worker.name == name:
                return worker
        return None

    def start_worker_for_camera(self, camera: Camera, face_blur: bool | None = None) -> IngestWorker:
        """Запускает ingest-воркер для переданной камеры."""

        existing_worker = self.get_worker(camera.name)
        if existing_worker:
            logger.info(
                "Ingest-воркер для камеры '%s' уже запущен, повторный запуск пропущен",
                camera.name,
            )
            return existing_worker

        if face_blur is None:
            face_blur = settings.face_blur

        worker = IngestWorker(
            self.session_factory,
            camera.id,
            camera.name,
            camera.rtsp_url,
            face_blur=face_blur,
            broadcaster=self.broadcaster,
            main_loop=self.main_loop,
        )
        worker.start()
        self.workers.append(worker)
        logger.info(
            "Ingest-воркер для камеры '%s' (#%s) запущен",
            camera.name,
            camera.id,
        )
        return worker

    def runtime_status(self) -> dict:
        workers = [worker.runtime_status() for worker in self.workers]

        try:
            import torch
        except Exception:  # pragma: no cover
            torch = None

        torch_available = torch is not None
        cuda_available = bool(torch_available and torch.cuda.is_available()) if torch_available else False
        cuda_device_count = 0
        cuda_name = None
        if cuda_available:
            try:
                cuda_device_count = int(torch.cuda.device_count())
            except Exception:
                cuda_device_count = 0
            if cuda_device_count:
                try:
                    cuda_name = torch.cuda.get_device_name(0)
                except Exception:
                    cuda_name = None

        mps_available = False
        if torch_available and hasattr(torch.backends, "mps"):
            try:
                mps_available = bool(torch.backends.mps.is_available())
            except Exception:
                mps_available = False

        system = {
            "torch_available": torch_available,
            "torch_version": getattr(torch, "__version__", None) if torch_available else None,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_name": cuda_name,
            "mps_available": mps_available,
            "env_device": settings.yolo_device,
            "cuda_visible_devices": settings.cuda_visible_devices,
        }

        return {
            "system": system,
            "workers": workers,
        }
