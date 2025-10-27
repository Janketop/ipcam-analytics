"""Маршруты для работы с событиями."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    get_ingest_manager,
    get_session,
    get_ws_broadcaster,
)
from backend.models import Camera, Event

router = APIRouter()


def _parse_timestamp(value: str | None, field_name: str) -> datetime | None:
    """Преобразовать ISO-строку во временную метку."""

    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - защитный блок
        raise HTTPException(
            status_code=422,
            detail=f"Параметр {field_name} должен быть в формате ISO 8601",
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    return parsed


@router.get("/events")
def list_events(
    limit: int = Query(200, ge=1, le=1000),
    type: str | None = Query(default=None),
    camera: str | None = Query(default=None),
    from_ts: str | None = Query(default=None),
    to_ts: str | None = Query(default=None),
    session: Session = Depends(get_session),
):
    event_type = type.strip() if type else None
    camera_name = camera.strip() if camera else None
    from_dt = _parse_timestamp(from_ts, "from_ts")
    to_dt = _parse_timestamp(to_ts, "to_ts")

    if from_dt and to_dt and from_dt > to_dt:
        raise HTTPException(
            status_code=422,
            detail="Параметр from_ts не может быть больше to_ts",
        )

    query = session.query(Event, Camera).join(Camera, Event.camera_id == Camera.id)
    if event_type:
        query = query.filter(Event.type == event_type)
    if camera_name:
        query = query.filter(Camera.name == camera_name)
    if from_dt:
        query = query.filter(Event.start_ts >= from_dt)
    if to_dt:
        query = query.filter(Event.start_ts <= to_dt)

    rows = query.order_by(Event.start_ts.desc()).limit(limit).all()
    return {
        "events": [
            {
                "id": event.id,
                "type": event.type,
                "start_ts": event.start_ts,
                "end_ts": event.end_ts,
                "confidence": event.confidence,
                "snapshot_url": event.snapshot_url,
                "meta": event.meta,
                "camera": camera.name,
            }
            for event, camera in rows
        ]
    }


@router.get("/runtime")
def runtime(ingest=Depends(get_ingest_manager)):
    return ingest.runtime_status()


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket, broadcaster=Depends(get_ws_broadcaster)):
    await broadcaster.handle_connection(ws)
