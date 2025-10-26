"""Маршруты для работы с событиями."""
from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy import text

from backend.core.dependencies import get_broadcaster, get_engine, get_ingest_manager

router = APIRouter()


@router.get("/events")
def list_events(limit: int = 200, type: str | None = None, engine=Depends(get_engine)):
    query = (
        "SELECT e.id, e.type, e.start_ts, e.end_ts, e.confidence, e.snapshot_url, e.meta, "
        "c.name AS camera FROM events e JOIN cameras c ON e.camera_id = c.id "
    )
    params = {"lim": limit}
    if type:
        query += " WHERE e.type=:t "
        params["t"] = type
    query += " ORDER BY e.start_ts DESC LIMIT :lim"
    with engine.connect() as con:
        rows = con.execute(text(query), params).mappings().all()
        return {"events": list(rows)}


@router.get("/runtime")
def runtime(ingest=Depends(get_ingest_manager)):
    return ingest.runtime_status()


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket, broadcaster=Depends(get_broadcaster)):
    await broadcaster.handle_connection(ws)
