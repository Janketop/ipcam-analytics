"""Эндпоинты для мониторинга состояния сервиса и управления очисткой."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.core.config import settings
from backend.core.dependencies import get_cleanup_state
from backend.services.cleanup import perform_cleanup

router = APIRouter()


def _serialize_cleanup_state(cleanup_state: Dict[str, Any], *, in_progress: bool) -> Dict[str, Any]:
    """Подготавливает словарь с состоянием очистки к выдаче через API."""

    last_run = cleanup_state.get("last_run")
    cutoff = cleanup_state.get("cutoff")
    face_sample_cutoff = cleanup_state.get("face_sample_cutoff")

    return {
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
        "in_progress": in_progress,
    }


@router.get("/health")
def health(request: Request, cleanup_state=Depends(get_cleanup_state)):
    cleanup_lock = getattr(request.app.state, "cleanup_lock", None)
    in_progress = bool(cleanup_lock.locked()) if cleanup_lock is not None else False
    return {
        "ok": True,
        "face_blur": settings.face_blur,
        "retention_days": settings.retention_days,
        "face_sample_unverified_retention_days": settings.face_sample_unverified_retention_days,
        "cleanup_interval_hours": settings.retention_cleanup_interval_hours,
        "cleanup": _serialize_cleanup_state(cleanup_state, in_progress=in_progress),
    }


@router.post("/cleanup/run", status_code=status.HTTP_200_OK)
async def run_cleanup(request: Request):
    cleanup_lock = request.app.state.cleanup_lock
    if cleanup_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Очистка уже выполняется, дождитесь завершения.",
        )

    async with cleanup_lock:
        await perform_cleanup(
            request.app.state.session_factory,
            request.app.state.retention_days,
            request.app.state.cleanup_state,
            request.app.state.face_sample_unverified_retention_days,
        )

    return {
        "ok": True,
        "cleanup": _serialize_cleanup_state(
            request.app.state.cleanup_state,
            in_progress=False,
        ),
    }
