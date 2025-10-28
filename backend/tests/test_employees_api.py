from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.routes.employees import FaceSampleAssignRequest, assign_face_sample
from backend.models import Camera, Employee, FaceSample
from backend.models.base import Base
from backend.services.face_embeddings import FaceEmbeddingResult, get_embedding_metadata


@pytest.fixture()
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=[Employee.__table__, FaceSample.__table__])
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS cameras (id INTEGER PRIMARY KEY, name TEXT)")
        )
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def _factory():
        return factory()

    return _factory


def test_assign_face_sample_stores_embedding(monkeypatch, session_factory):
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        employee = Employee(name="Alice")
        session.add(employee)
        session.flush()
        sample = FaceSample(
            snapshot_url="/static/snaps/example.jpg",
            status=FaceSample.STATUS_UNVERIFIED,
            captured_at=now,
            updated_at=now,
        )
        sample.id = 1
        session.add(sample)
        session.commit()
        sample_id = sample.id
        employee_id = employee.id

    metadata = get_embedding_metadata()
    fake_embedding = FaceEmbeddingResult(
        vector=np.ones(metadata["embedding_dim"], dtype=np.float32),
        model=metadata["model"],
    )
    monkeypatch.setattr(
        "backend.api.routes.employees.compute_face_embedding_from_snapshot",
        lambda *args, **kwargs: fake_embedding,
    )

    notified: list[bool] = []
    monkeypatch.setattr(
        "backend.api.routes.employees.EmployeeRecognizer.notify_embeddings_updated",
        lambda: notified.append(True),
    )

    with session_factory() as session:
        payload = FaceSampleAssignRequest(employee_id=employee_id)
        response = assign_face_sample(sample_id=sample_id, payload=payload, session=session)

    face_sample = response["faceSample"]
    assert face_sample["status"] == FaceSample.STATUS_EMPLOYEE
    assert face_sample["employee"]["id"] == employee_id

    with session_factory() as session:
        updated = session.get(FaceSample, sample_id)
        assert updated is not None
        assert updated.employee_id == employee_id
        assert updated.embedding is not None
        assert updated.embedding_dim == metadata["embedding_dim"]
        assert updated.embedding_model == metadata["model"]

    assert notified
