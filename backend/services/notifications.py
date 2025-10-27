"""Управление WebSocket-клиентами и рассылкой уведомлений."""
from __future__ import annotations

from typing import List

from fastapi import WebSocket, WebSocketDisconnect

from backend.core.logger import logger


class EventBroadcaster:
    """Простой менеджер подключений WebSocket."""

    def __init__(self) -> None:
        self._clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.append(websocket)
        logger.info("WebSocket %s подключен", websocket.client)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._clients:
            self._clients.remove(websocket)
            logger.info("WebSocket %s отключен", websocket.client)

    async def handle_connection(self, websocket: WebSocket) -> None:
        await self.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.warning("WebSocket %s разорвал соединение", websocket.client)
        finally:
            self.disconnect(websocket)

    async def broadcast(self, payload: dict) -> None:
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                logger.exception("Ошибка при отправке события по WebSocket %s", ws.client)
                self.disconnect(ws)
