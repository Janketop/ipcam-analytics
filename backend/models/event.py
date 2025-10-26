"""Модель события, зафиксированного системой."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base

if TYPE_CHECKING:  # pragma: no cover
    from backend.models.camera import Camera


class Event(Base):
    """Информация о событии, связанном с камерой."""

    __tablename__ = "events"
    __table_args__ = (
        Index("events_time_idx", "start_ts"),
        Index("events_type_idx", "type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    camera_id: Mapped[Optional[int]] = mapped_column(ForeignKey("cameras.id"))
    type: Mapped[str] = mapped_column(Text, nullable=False)
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    snapshot_url: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    camera: Mapped[Optional["Camera"]] = relationship(back_populates="events")

    def __repr__(self) -> str:  # pragma: no cover - для отладки
        return f"Event(id={self.id!r}, type={self.type!r}, camera_id={self.camera_id!r})"


__all__ = ["Event"]
