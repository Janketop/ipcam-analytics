"""WebSocket-маршрут для трансляции статусов камер."""
from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket

from backend.core.dependencies import get_ws_status_broadcaster


router = APIRouter()


@router.websocket("/ws/statuses")
async def ws_statuses(ws: WebSocket, broadcaster=Depends(get_ws_status_broadcaster)):
    await broadcaster.handle_connection(ws)
