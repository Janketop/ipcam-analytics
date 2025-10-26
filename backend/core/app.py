"""Создание и базовая настройка FastAPI-приложения."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core import config
from backend.core.database import get_engine
from backend.core.paths import STATIC_DIR
from backend.services.ingest_manager import IngestManager
from backend.services.notifications import EventBroadcaster


def _setup_cors(app: FastAPI) -> None:
    frontend_origins = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_URL") or ""
    allow_origins = {origin.strip() for origin in frontend_origins.split(",") if origin.strip()}
    default_origins = {"http://localhost:3000", "http://127.0.0.1:3000"}
    allow_origins.update(default_origins)
    origin_regex = os.getenv("FRONTEND_ORIGIN_REGEX") or r"https?://.*"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(allow_origins),
        allow_origin_regex=origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_app() -> FastAPI:
    app = FastAPI(title=config.APP_TITLE)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    _setup_cors(app)

    engine = get_engine()
    ingest = IngestManager(engine)
    broadcaster = EventBroadcaster()
    ingest.set_broadcaster(broadcaster.broadcast)

    app.state.engine = engine
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
    app.state.retention_days = config.RETENTION_DAYS
    app.state.cleanup_interval_hours = config.CLEANUP_INTERVAL_HOURS

    return app
