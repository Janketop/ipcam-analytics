"""Модель сотрудника, используемая для привязки лиц."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base


class Employee(Base):
    """Карточка сотрудника, для которого подбираются снимки."""

    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    account_id: Mapped[str | None] = mapped_column(
        Text, nullable=True, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    face_samples = relationship("FaceSample", back_populates="employee")

    def __repr__(self) -> str:  # pragma: no cover - отладка
        return (
            f"Employee(id={self.id!r}, name={self.name!r}, account_id={self.account_id!r})"
        )


__all__ = ["Employee"]

