"""Управление WebSocket-клиентами и рассылкой уведомлений."""
from __future__ import annotations

from typing import List

from fastapi import WebSocket, WebSocketDisconnect


class EventBroadcaster:
    """Простой менеджер подключений WebSocket."""

    def __init__(self) -> None:
        self._clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._clients:
            self._clients.remove(websocket)

    async def handle_connection(self, websocket: WebSocket) -> None:
        await self.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            self.disconnect(websocket)

    async def broadcast(self, payload: dict) -> None:
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(ws)
