"""Эндпоинты для мониторинга состояния сервиса."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from backend.core.config import settings
from backend.core.dependencies import get_cleanup_state

router = APIRouter()


@router.get("/health")
def health(cleanup_state=Depends(get_cleanup_state)):
    last_run = cleanup_state.get("last_run")
    cutoff = cleanup_state.get("cutoff")
    face_sample_cutoff = cleanup_state.get("face_sample_cutoff")
    return {
        "ok": True,
        "face_blur": settings.face_blur,
        "retention_days": settings.retention_days,
        "face_sample_unverified_retention_days": settings.face_sample_unverified_retention_days,
        "cleanup_interval_hours": settings.retention_cleanup_interval_hours,
        "cleanup": {
            "last_run": last_run.isoformat() if isinstance(last_run, datetime) else None,
            "cutoff": cutoff.isoformat() if isinstance(cutoff, datetime) else None,
            "deleted_events": cleanup_state.get("deleted_events", 0),
            "deleted_snapshots": cleanup_state.get("deleted_snapshots", 0),
            "deleted_dataset_copies": cleanup_state.get("deleted_dataset_copies", 0),
            "deleted_face_samples": cleanup_state.get("deleted_face_samples", 0),
            "face_sample_cutoff": face_sample_cutoff.isoformat()
            if isinstance(face_sample_cutoff, datetime)
            else None,
            "error": cleanup_state.get("error"),
        },
    }
