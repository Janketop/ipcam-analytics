"""Управление WebSocket-клиентами и рассылкой уведомлений."""
from __future__ import annotations

from typing import List

from fastapi import WebSocket, WebSocketDisconnect

from backend.core.logger import logger


class _BaseBroadcaster:
    """Общий функционал для менеджеров WebSocket-подключений."""

    def __init__(self, channel_name: str) -> None:
        self._channel_name = channel_name
        self._clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.append(websocket)
        logger.info("[%s] WebSocket %s подключен", self._channel_name, websocket.client)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._clients:
            self._clients.remove(websocket)
            logger.info("[%s] WebSocket %s отключен", self._channel_name, websocket.client)

    async def handle_connection(self, websocket: WebSocket) -> None:
        await self.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.warning(
                "[%s] WebSocket %s разорвал соединение",
                self._channel_name,
                websocket.client,
            )
        finally:
            self.disconnect(websocket)

    async def broadcast(self, payload: dict) -> None:
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                logger.exception(
                    "[%s] Ошибка при отправке данных по WebSocket %s",
                    self._channel_name,
                    ws.client,
                )
                self.disconnect(ws)


class EventBroadcaster(_BaseBroadcaster):
    """Менеджер рассылки событий."""

    def __init__(self) -> None:
        super().__init__("events")


class StatusBroadcaster(_BaseBroadcaster):
    """Менеджер рассылки статусов камер."""

    def __init__(self) -> None:
        super().__init__("statuses")
