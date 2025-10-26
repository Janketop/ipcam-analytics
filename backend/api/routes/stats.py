"""Статистические эндпоинты."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.dependencies import get_session
from backend.models import Event

router = APIRouter()


@router.get("/stats")
def stats(session: Session = Depends(get_session)):
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    count_expr = func.count(Event.id)
    rows = (
        session.query(Event.type, count_expr.label("cnt"))
        .filter(Event.start_ts > cutoff)
        .group_by(Event.type)
        .order_by(count_expr.desc())
        .all()
    )
    return {
        "stats": [
            {
                "type": row.type,
                "cnt": row.cnt,
            }
            for row in rows
        ]
    }
