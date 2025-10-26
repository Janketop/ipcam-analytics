"""Статистические эндпоинты."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from backend.core.dependencies import get_engine

router = APIRouter()


@router.get("/stats")
def stats(engine=Depends(get_engine)):
    query = text(
        """
        SELECT type, COUNT(*) AS cnt
        FROM events
        WHERE start_ts > now() - interval '1 day'
        GROUP BY type
        ORDER BY cnt DESC
        """
    )
    with engine.connect() as con:
        rows = con.execute(query).mappings().all()
    return {"stats": list(rows)}
