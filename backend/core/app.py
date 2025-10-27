"""Создание и базовая настройка FastAPI-приложения."""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings
from backend.core.database import get_session_factory
from backend.core.logger import logger
from backend.core.paths import STATIC_DIR
from backend.services.ingest_manager import IngestManager
from backend.services.notifications import EventBroadcaster
from backend.services.training import SelfTrainingService


def _setup_cors(app: FastAPI) -> None:
    allow_origins = settings.cors_allow_origin_list
    origin_regex = settings.frontend_origin_regex or r"https?://.*"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _log_gpu_status() -> None:
    """Вывести в лог информацию о доступности GPU и версии CUDA."""
    if importlib.util.find_spec("torch") is None:
        logger.info("GPU Detected: False (CUDA N/A - пакет torch не установлен)")
        return

    torch = importlib.import_module("torch")
    cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
    cuda_version = getattr(getattr(torch, "version", object()), "cuda", None)
    version_display = cuda_version if cuda_version else "N/A"

    logger.info("GPU Detected: %s (CUDA %s)", str(cuda_available), version_display)


def create_app() -> FastAPI:
    logger.info("Создание экземпляра FastAPI приложения")
    app = FastAPI(title=settings.app_title)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    _setup_cors(app)

    session_factory = get_session_factory()
    ingest = IngestManager(session_factory)
    broadcaster = EventBroadcaster()
    ingest.set_broadcaster(broadcaster.broadcast)

    app.state.session_factory = session_factory
    app.state.ingest_manager = ingest
    app.state.event_broadcaster = broadcaster
    app.state.cleanup_state = {
        "last_run": None,  # type: Optional[datetime]
        "deleted_events": 0,
        "deleted_snapshots": 0,
        "error": None,
        "cutoff": None,
    }
    app.state.cleanup_lock = asyncio.Lock()
    app.state.background_tasks: list[asyncio.Task] = []
    app.state.retention_days = settings.retention_days
    app.state.cleanup_interval_hours = settings.retention_cleanup_interval_hours
    app.state.self_training_service = SelfTrainingService()

    _log_gpu_status()
    logger.info(
        "Приложение подготовлено: face_blur=%s, retention=%d дней",
        settings.face_blur,
        settings.retention_days,
    )
    return app
