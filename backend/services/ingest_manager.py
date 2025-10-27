"""Менеджер потоков захвата и обработки видеопотоков."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
        self.status_broadcaster: Optional[Callable[[dict], Awaitable[None]]] = None
        self.main_loop = main_loop

    def set_broadcaster(self, fn: Callable[[dict], Awaitable[None]]) -> None:
        self.broadcaster = fn
        for worker in self.workers:
            worker.set_broadcaster(fn)

    def set_status_broadcaster(self, fn: Callable[[dict], Awaitable[None]]) -> None:
        self.status_broadcaster = fn
        for worker in self.workers:
            worker.set_status_broadcaster(fn)

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

        logger.info("Найдено %d активных камер для запуска", len(cameras))
        for camera in cameras:
            self.start_worker_for_camera(camera)

    def stop_all(self) -> None:
        if not self.workers:
            logger.info("Нет активных ingest-воркеров для остановки")
            return

        for worker in list(self.workers):
            self.stop_worker_for_camera(worker.name)

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

        if face_blur not in (None, False):
            logger.info(
                "Настройка размытия лиц больше не используется и будет проигнорирована"
            )
        face_blur = False

        worker = IngestWorker(
            self.session_factory,
            camera.id,
            camera.name,
            camera.rtsp_url,
            face_blur=face_blur,
            detect_person=camera.detect_person,
            detect_car=camera.detect_car,
            capture_entry_time=camera.capture_entry_time,
            idle_alert_time=camera.idle_alert_time or settings.idle_alert_time,
            broadcaster=self.broadcaster,
            status_broadcaster=self.status_broadcaster,
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

    def stop_worker_for_camera(self, camera_name: str) -> bool:
        """Останавливает ingest-воркер, связанный с указанной камерой."""

        worker = self.get_worker(camera_name)
        if not worker:
            logger.info(
                "Запрос на остановку ingest-воркера для '%s', но активный поток не найден",
                camera_name,
            )
            return False

        worker.stop_flag = True
        logger.info("Отправлен сигнал остановки ingest-воркеру '%s'", worker.name)
        worker.join(timeout=5.0)
        if worker.is_alive():
            logger.warning(
                "Ingest-воркер '%s' не завершил работу в отведённое время",
                worker.name,
            )

        try:
            self.workers.remove(worker)
        except ValueError:
            pass

        self._notify_offline(worker)
        return True

    def runtime_status(self) -> dict:
        workers = [worker.runtime_status() for worker in self.workers]

        alive_workers = sum(1 for worker in self.workers if worker.is_alive())
        fps_values = [w.get("fps") for w in workers if isinstance(w.get("fps"), (int, float))]
        uptime_values = [w.get("uptime_seconds") for w in workers if isinstance(w.get("uptime_seconds"), (int, float))]
        last_frames: List[datetime] = []
        for w in workers:
            last_frame_at = w.get("last_frame_at")
            if isinstance(last_frame_at, str):
                try:
                    last_frames.append(datetime.fromisoformat(last_frame_at))
                except ValueError:
                    continue

        summary = {
            "total_workers": len(self.workers),
            "alive_workers": alive_workers,
            "avg_fps": sum(fps_values) / len(fps_values) if fps_values else None,
            "max_uptime_seconds": max(uptime_values) if uptime_values else None,
            "latest_frame_at": max(last_frames).isoformat() if last_frames else None,
        }

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
            "summary": summary,
        }

    def _notify_offline(self, worker: IngestWorker) -> None:
        if not self.status_broadcaster or not self.main_loop or self.main_loop.is_closed():
            return

        payload = {
            "cameraId": worker.cam_id,
            "camera": worker.name,
            "status": "offline",
            "fps": None,
            "lastFrameTs": None,
            "uptimeSec": None,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.status_broadcaster(payload),
                self.main_loop,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Не удалось отправить финальный статус камеры %s: %s",
                worker.name,
                exc,
            )
            return

        def _finalize(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except (asyncio.CancelledError, RuntimeError) as exc:
                logger.warning(
                    "Рассылка финального статуса для камеры %s была прервана: %s",
                    worker.name,
                    exc,
                )
            except Exception:
                logger.exception(
                    "Ошибка при завершении отправки финального статуса камеры %s",
                    worker.name,
                )

        future.add_done_callback(_finalize)
