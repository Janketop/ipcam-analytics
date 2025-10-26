"""Общие зависимости FastAPI."""
from __future__ import annotations

from typing import Generator

from fastapi import Request, WebSocket
from sqlalchemy.orm import Session

from backend.services.notifications import EventBroadcaster


def _get_event_broadcaster(app) -> EventBroadcaster:
    """Возвращает менеджер рассылки событий из состояния приложения."""

    return app.state.event_broadcaster


def get_session(request: Request) -> Generator[Session, None, None]:
    """Создаёт и возвращает ORM-сессию для запроса."""

    session_factory = request.app.state.session_factory
    session: Session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_ingest_manager(request: Request):
    return request.app.state.ingest_manager


def get_cleanup_state(request: Request):
    return request.app.state.cleanup_state


def get_broadcaster(request: Request) -> EventBroadcaster:
    """Получает менеджер рассылки для HTTP-запроса."""

    return _get_event_broadcaster(request.app)


def get_ws_broadcaster(websocket: WebSocket) -> EventBroadcaster:
    """Получает менеджер рассылки для WebSocket-подключения."""

    return _get_event_broadcaster(websocket.app)
