"""Управление WebSocket-клиентами и рассылкой уведомлений."""
from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

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
        self._last_payloads: Dict[Tuple[str, int | str], dict] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await super().connect(websocket)

        async with self._lock:
            snapshots = list(self._last_payloads.values())

        if not snapshots:
            return

        def _sort_key(item: dict) -> Tuple[int, int | str]:
            camera_id = item.get("cameraId")
            if isinstance(camera_id, int):
                return (0, camera_id)
            camera_name = item.get("camera")
            if isinstance(camera_name, str):
                return (1, camera_name)
            return (2, "")

        for payload in sorted(snapshots, key=_sort_key):
            try:
                await websocket.send_json(payload)
            except Exception:
                logger.exception(
                    "[%s] Ошибка при отправке накопленных статусов WebSocket %s",
                    self._channel_name,
                    websocket.client,
                )
                self.disconnect(websocket)
                break

    async def broadcast(self, payload: dict) -> None:
        key: Tuple[str, int | str] | None = None
        camera_id = payload.get("cameraId")
        camera_name = payload.get("camera")
        if isinstance(camera_id, int):
            key = ("id", camera_id)
        elif isinstance(camera_name, str) and camera_name:
            key = ("name", camera_name)

        if key is not None:
            async with self._lock:
                self._last_payloads[key] = dict(payload)

        await super().broadcast(payload)
