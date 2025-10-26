"""Служебные функции для очистки устаревших событий и снапшотов."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Set, Tuple

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.core.paths import SNAPSHOT_DIR


async def perform_cleanup(engine: Engine, retention_days: int, state: dict) -> None:
    """Запускает очистку в отдельном потоке и обновляет state."""
    started_at = datetime.now(timezone.utc)
    try:
        deleted_events, deleted_snapshots, cutoff_dt = await asyncio.to_thread(
            cleanup_expired_events_and_snapshots,
            engine,
            retention_days,
        )
        state.update(
            {
                "last_run": started_at,
                "deleted_events": deleted_events,
                "deleted_snapshots": deleted_snapshots,
                "error": None,
                "cutoff": cutoff_dt,
            }
        )
    except Exception as exc:
        state.update(
            {
                "last_run": started_at,
                "error": str(exc),
                "cutoff": None,
            }
        )


async def cleanup_loop(app: FastAPI, interval_hours: float) -> None:
    engine = app.state.engine
    retention_days = app.state.retention_days
    cleanup_state = app.state.cleanup_state
    lock = app.state.cleanup_lock
    interval_hours = max(interval_hours, 1.0)
    interval_seconds = interval_hours * 3600
    while True:
        async with lock:
            await perform_cleanup(engine, retention_days, cleanup_state)
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break


def cleanup_expired_events_and_snapshots(engine: Engine, retention_days: int) -> Tuple[int, int, datetime]:
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)

    delete_stmt = text("DELETE FROM events WHERE start_ts < now() - interval ':days day'")
    with engine.begin() as con:
        result = con.execute(delete_stmt, {"days": retention_days})
        deleted_events = result.rowcount or 0

        rows = con.execute(
            text("SELECT snapshot_url FROM events WHERE snapshot_url IS NOT NULL")
        )
        snapshot_names: Set[str] = {
            Path(url).name
            for url in rows.scalars()
            if isinstance(url, str) and url.strip()
        }

    deleted_snapshots = 0
    for file_path in SNAPSHOT_DIR.iterdir():
        if not file_path.is_file():
            continue

        file_should_be_removed = False
        if file_path.name not in snapshot_names:
            file_should_be_removed = True
        else:
            try:
                created_at = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
            except OSError:
                created_at = None
            if created_at is None or created_at < cutoff_dt:
                file_should_be_removed = True

        if file_should_be_removed:
            try:
                file_path.unlink()
                deleted_snapshots += 1
            except FileNotFoundError:
                continue
            except OSError:
                continue

    return deleted_events, deleted_snapshots, cutoff_dt
