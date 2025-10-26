"""Маршруты для работы с событиями."""
from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    get_ingest_manager,
    get_session,
    get_ws_broadcaster,
)
from backend.models import Camera, Event

router = APIRouter()


@router.get("/events")
def list_events(
    limit: int = 200,
    type: str | None = None,
    session: Session = Depends(get_session),
):
    limit = max(1, limit)
    event_type = type
    query = session.query(Event, Camera).join(Camera, Event.camera_id == Camera.id)
    if event_type:
        query = query.filter(Event.type == event_type)

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
