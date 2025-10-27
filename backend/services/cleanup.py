"""Служебные функции для очистки устаревших событий и снапшотов."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from fastapi import FastAPI
from sqlalchemy import select

from backend.core.database import SessionFactory
from backend.core.logger import logger
from backend.models import Event
from backend.core.paths import SNAPSHOT_DIR


async def perform_cleanup(
    session_factory: SessionFactory, retention_days: int, state: dict
) -> None:
    """Запускает очистку в отдельном потоке и обновляет state."""
    started_at = datetime.now(timezone.utc)
    logger.info("Очистка: старт (retention=%d дней)", retention_days)
    try:
        deleted_events, deleted_snapshots, cutoff_dt = await asyncio.to_thread(
            cleanup_expired_events_and_snapshots,
            session_factory,
            retention_days,
            started_at,
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
        logger.info(
            "Очистка завершена: удалено %d событий и %d снимков (граница %s)",
            deleted_events,
            deleted_snapshots,
            cutoff_dt.isoformat(),
        )
    except Exception as exc:
        state.update(
            {
                "last_run": started_at,
                "error": str(exc),
                "cutoff": None,
            }
        )
        logger.exception("Очистка завершилась с ошибкой")


async def cleanup_loop(app: FastAPI, interval_hours: float) -> None:
    session_factory = app.state.session_factory
    retention_days = app.state.retention_days
    cleanup_state = app.state.cleanup_state
    lock = app.state.cleanup_lock
    interval_hours = max(interval_hours, 1.0)
    interval_seconds = interval_hours * 3600
    while True:
        async with lock:
            logger.info("Фоновая очистка: запуск цикла")
            await perform_cleanup(session_factory, retention_days, cleanup_state)
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.warning("Фоновая очистка остановлена по CancelledError")
            break


def cleanup_expired_events_and_snapshots(
    session_factory: SessionFactory,
    retention_days: int,
    started_at: Optional[datetime] = None,
) -> Tuple[int, int, datetime]:
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
    if started_at is None:
        started_at = datetime.now(timezone.utc)

    with session_factory() as session:
        deleted_events = (
            session.query(Event)
            .filter(Event.start_ts < cutoff_dt)
            .delete(synchronize_session=False)
        )
        session.commit()

        snapshot_rows = session.execute(
            select(Event.id, Event.snapshot_url).where(Event.snapshot_url.is_not(None))
        ).all()

    snapshot_events: Dict[str, List[int]] = defaultdict(list)
    for event_id, url in snapshot_rows:
        if not isinstance(url, str):
            continue
        url = url.strip()
        if not url:
            continue
        snapshot_events[Path(url).name].append(event_id)

    snapshot_names: Set[str] = set(snapshot_events.keys())
    existing_files: Set[str] = set()

    deleted_snapshots = 0
    for file_path in SNAPSHOT_DIR.iterdir():
        if not file_path.is_file():
            continue

        existing_files.add(file_path.name)

        file_should_be_removed = False
        try:
            created_at = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
        except OSError as exc:
            logger.warning(
                "Не удалось получить время создания файла %s: %s",
                file_path,
                exc,
            )
            created_at = None

        if file_path.name not in snapshot_names:
            if created_at is not None and created_at >= started_at:
                continue
            file_should_be_removed = True
        else:
            if created_at is None or created_at < cutoff_dt:
                file_should_be_removed = True

        if file_should_be_removed:
            try:
                file_path.unlink()
                deleted_snapshots += 1
            except FileNotFoundError as exc:
                logger.warning(
                    "Не удалось удалить файл %s: файл уже удалён (%s)",
                    file_path,
                    exc,
                )
                continue
            except OSError as exc:
                logger.warning(
                    "Ошибка при удалении файла %s: %s",
                    file_path,
                    exc,
                )
                continue

    missing_snapshot_event_ids = [
        event_id
        for name, ids in snapshot_events.items()
        if name not in existing_files
        for event_id in ids
    ]

    if missing_snapshot_event_ids:
        with session_factory() as session:
            (
                session.query(Event)
                .filter(Event.id.in_(missing_snapshot_event_ids))
                .update({Event.snapshot_url: None}, synchronize_session=False)
            )
            session.commit()
        logger.warning(
            "Обнаружено %d событий без файлов снимков, ссылки обнулены",
            len(missing_snapshot_event_ids),
        )

    return deleted_events, deleted_snapshots, cutoff_dt
