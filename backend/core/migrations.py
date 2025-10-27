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
        if statements:
            for statement in statements:
                session.execute(text(statement))

            session.commit()

            logger.info(
                "Добавлены отсутствующие флаги камер: %s",
                ", ".join(stmt.split()[4] for stmt in statements),
            )
        else:
            logger.info("Миграции флагов камер не требуются — схема актуальна.")

        face_samples_statements = (
            """
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS face_samples (
                id BIGSERIAL PRIMARY KEY,
                event_id BIGINT UNIQUE REFERENCES events(id) ON DELETE CASCADE,
                employee_id INT REFERENCES employees(id) ON DELETE SET NULL,
                camera_id INT REFERENCES cameras(id) ON DELETE SET NULL,
                snapshot_url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'unverified',
                candidate_key TEXT,
                captured_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            )
            """,
            "CREATE INDEX IF NOT EXISTS face_samples_status_idx ON face_samples(status)",
            "CREATE INDEX IF NOT EXISTS face_samples_employee_idx ON face_samples(employee_id)",
            "CREATE INDEX IF NOT EXISTS face_samples_captured_idx ON face_samples(captured_at)",
        )

        for statement in face_samples_statements:
            session.execute(text(statement))

        session.commit()

        logger.info(
            "Убедились в наличии таблиц employees/face_samples и индексов.",
        )


__all__ = ["run_startup_migrations"]

