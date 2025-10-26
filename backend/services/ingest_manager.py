"""Менеджер потоков захвата и обработки видеопотоков."""
from __future__ import annotations

import os
from typing import Awaitable, Callable, List, Optional

from sqlalchemy import text

from backend.services.ingest_worker import IngestWorker
from backend.utils.env import env_flag


class IngestManager:
    def __init__(self, engine, main_loop=None) -> None:
        self.engine = engine
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
        with self.engine.connect() as con:
            cams = con.execute(text("SELECT id,name,rtsp_url FROM cameras WHERE active=true")).mappings().all()
        face_blur = env_flag("FACE_BLUR", False)
        for camera in cams:
            worker = IngestWorker(
                self.engine,
                camera["id"],
                camera["name"],
                camera["rtsp_url"],
                face_blur=face_blur,
                broadcaster=self.broadcaster,
                main_loop=self.main_loop,
            )
            worker.start()
            self.workers.append(worker)

    def stop_all(self) -> None:
        for worker in self.workers:
            worker.stop_flag = True

    def get_worker(self, name: str):
        for worker in self.workers:
            if worker.name == name:
                return worker
        return None

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
            "env_device": os.getenv("YOLO_DEVICE", "auto"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        }

        return {
            "system": system,
            "workers": workers,
        }
