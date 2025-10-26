"""Модель камеры наблюдения."""
from __future__ import annotations

from datetime import datetime
from typing import List, TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base

if TYPE_CHECKING:  # pragma: no cover
    from backend.models.event import Event


class Camera(Base):
    """Описание камеры видеонаблюдения."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    rtsp_url: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    events: Mapped[List["Event"]] = relationship(back_populates="camera")

    def __repr__(self) -> str:  # pragma: no cover - представление для отладки
        return f"Camera(id={self.id!r}, name={self.name!r})"


__all__ = ["Camera"]
