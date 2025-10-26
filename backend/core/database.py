"""Модуль с настройкой подключения к базе данных и фабрикой сессий."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

SessionFactory = Callable[[], Session]


def _make_engine() -> Engine:
    """Создаёт подключение к PostgreSQL с учётом переменных окружения."""

    user = os.getenv("POSTGRES_USER", "ipcam")
    pwd = os.getenv("POSTGRES_PASSWORD", "ipcam")
    db = os.getenv("POSTGRES_DB", "ipcam")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)


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
