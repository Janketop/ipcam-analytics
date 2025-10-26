"""Модуль для работы с подключением к базе данных."""
from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Создаёт и кэширует подключение к базе данных PostgreSQL."""
    user = os.getenv("POSTGRES_USER", "ipcam")
    pwd = os.getenv("POSTGRES_PASSWORD", "ipcam")
    db = os.getenv("POSTGRES_DB", "ipcam")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    engine = create_engine(url, pool_pre_ping=True, future=True)
    return engine
