"""Модуль с настройкой подключения к базе данных и фабрикой сессий."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import settings

SessionFactory = Callable[[], Session]


def _make_engine() -> Engine:
    """Создаёт подключение к PostgreSQL с учётом переменных окружения."""

    return create_engine(settings.postgres_dsn, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Возвращает singleton-экземпляр движка базы данных."""

    return _make_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Создаёт и кэширует фабрику синхронных сессий SQLAlchemy."""

    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


__all__ = ["get_engine", "get_session_factory", "SessionFactory"]
