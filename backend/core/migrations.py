"""Небольшие idempotent-миграции, выполняемые при старте приложения."""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import inspect, text
from sqlalchemy.exc import NoSuchTableError

from backend.core.logger import logger
from backend.core.database import SessionFactory
from backend.core.config import settings


def _add_missing_camera_flags(columns: set[str]) -> Iterable[str]:
    """Генерирует SQL-запросы для добавления недостающих флагов камеры."""

    alter_template = (
        "ALTER TABLE cameras ADD COLUMN {column} BOOLEAN NOT NULL DEFAULT TRUE"
    )

    for column in ("detect_person", "detect_car", "capture_entry_time"):
        if column not in columns:
            yield alter_template.format(column=column)

    if "idle_alert_time" not in columns:
        default_value = int(settings.idle_alert_time)
        yield (
            "ALTER TABLE cameras ADD COLUMN idle_alert_time INTEGER NOT NULL "
            f"DEFAULT {default_value}"
        )


def run_startup_migrations(session_factory: SessionFactory) -> None:
    """Запускает простые миграции БД, безопасные при повторном выполнении."""

    with session_factory() as session:
        inspector = inspect(session.bind)

        try:
            existing_columns = {col["name"] for col in inspector.get_columns("cameras")}
        except NoSuchTableError:
            logger.warning(
                "Таблица cameras не найдена, миграции флагов пропущены."
            )
            return

        statements = list(_add_missing_camera_flags(existing_columns))
        if not statements:
            logger.info("Миграции флагов камер не требуются — схема актуальна.")
            return

        for statement in statements:
            session.execute(text(statement))

        session.commit()

        logger.info(
            "Добавлены отсутствующие флаги камер: %s",
            ", ".join(stmt.split()[5] for stmt in statements),
        )


__all__ = ["run_startup_migrations"]

