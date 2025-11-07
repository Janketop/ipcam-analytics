"""Тесты для сервиса очистки файлов и ссылок на них."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, Text, create_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, sessionmaker

from backend.services import cleanup as cleanup_service


TestBase = declarative_base()


class TestEvent(TestBase):
    """Упрощённая модель события для тестов очистки."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TestFaceSample(TestBase):
    """Упрощённая модель карточки лица для тестирования очистки."""

    __tablename__ = "face_samples"

    STATUS_UNVERIFIED = "unverified"
    STATUS_EMPLOYEE = "employee"
    STATUS_CLIENT = "client"
    STATUS_DISCARDED = "discarded"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default=STATUS_UNVERIFIED)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def test_cleanup_removes_face_sample_without_snapshot(tmp_path, monkeypatch):
    """Если файл карточки лица отсутствует, запись должна удаляться."""

    db_path = tmp_path / "test.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    TestBase.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    monkeypatch.setattr(cleanup_service, "Event", TestEvent)
    monkeypatch.setattr(cleanup_service, "FaceSample", TestFaceSample)
    monkeypatch.setattr(
        cleanup_service,
        "FACE_SAMPLE_UNUSED_STATUSES",
        {TestFaceSample.STATUS_CLIENT, TestFaceSample.STATUS_DISCARDED},
    )

    snapshots_dir = tmp_path / "snaps"
    snapshots_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    monkeypatch.setattr(cleanup_service, "SNAPSHOT_DIR", snapshots_dir)
    monkeypatch.setattr(cleanup_service, "DATASET_PHONE_USAGE_DIR", dataset_dir)

    with session_factory() as session:
        sample = TestFaceSample(
            snapshot_url="/static/snaps/missing_face.jpg",
            status=TestFaceSample.STATUS_UNVERIFIED,
            captured_at=datetime.now(timezone.utc),
        )
        session.add(sample)
        session.commit()
        sample_id = sample.id

    cleanup_service.cleanup_expired_events_and_snapshots(
        session_factory=session_factory,
        retention_days=30,
        face_sample_retention_days=30,
    )

    with session_factory() as session:
        updated = session.get(TestFaceSample, sample_id)

    assert updated is None

    engine.dispose()
