"""Эндпоинты для мониторинга состояния сервиса."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from backend.core import config
from backend.core.dependencies import get_cleanup_state

router = APIRouter()


@router.get("/health")
def health(cleanup_state=Depends(get_cleanup_state)):
    last_run = cleanup_state.get("last_run")
    cutoff = cleanup_state.get("cutoff")
    return {
        "ok": True,
        "face_blur": config.FACE_BLUR,
        "retention_days": config.RETENTION_DAYS,
        "cleanup_interval_hours": config.CLEANUP_INTERVAL_HOURS,
        "cleanup": {
            "last_run": last_run.isoformat() if isinstance(last_run, datetime) else None,
            "cutoff": cutoff.isoformat() if isinstance(cutoff, datetime) else None,
            "deleted_events": cleanup_state.get("deleted_events", 0),
            "deleted_snapshots": cleanup_state.get("deleted_snapshots", 0),
            "error": cleanup_state.get("error"),
        },
    }
