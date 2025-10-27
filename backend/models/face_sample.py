"""Модель снимка лица, собранного из событий."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base


class FaceSample(Base):
    """Снимок лица, который нужно классифицировать как сотрудника или клиента."""

    __tablename__ = "face_samples"
    __table_args__ = (
        Index("face_samples_status_idx", "status"),
        Index("face_samples_employee_idx", "employee_id"),
        Index("face_samples_captured_idx", "captured_at"),
    )

    STATUS_UNVERIFIED = "unverified"
    STATUS_EMPLOYEE = "employee"
    STATUS_CLIENT = "client"
    STATUS_DISCARDED = "discarded"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("events.id", ondelete="CASCADE"), unique=True
    )
    employee_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("employees.id", ondelete="SET NULL"), nullable=True
    )
    camera_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    snapshot_url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default=STATUS_UNVERIFIED,
        server_default=text("'unverified'"),
    )
    candidate_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    employee = relationship("Employee", back_populates="face_samples")
    event = relationship("Event", backref="face_sample", uselist=False)

    def mark_employee(self, employee_id: int) -> None:
        self.employee_id = employee_id
        self.status = self.STATUS_EMPLOYEE

    def mark_client(self) -> None:
        self.employee_id = None
        self.status = self.STATUS_CLIENT

    def set_embedding(self, data: bytes, *, dim: int, model: str) -> None:
        self.embedding = data
        self.embedding_dim = int(dim)
        self.embedding_model = model

    def clear_embedding(self) -> None:
        self.embedding = None
        self.embedding_dim = None
        self.embedding_model = None

    def __repr__(self) -> str:  # pragma: no cover - отладка
        return f"FaceSample(id={self.id!r}, status={self.status!r}, snapshot={self.snapshot_url!r})"


__all__ = ["FaceSample"]

