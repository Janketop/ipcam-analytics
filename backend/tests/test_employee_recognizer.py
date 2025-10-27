from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.models import Employee, FaceSample
from backend.models.base import Base
from backend.services.employee_recognizer import EmployeeRecognizer


@pytest.fixture()
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=[Employee.__table__, FaceSample.__table__])
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def _factory():
        return factory()

    return _factory


def _make_embedding(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def test_recognizer_identifies_employee(session_factory):
    now = datetime.now(timezone.utc)
    vec_a = np.ones(128, dtype=np.float32)

    with session_factory() as session:
        alice = Employee(name="Alice")
        session.add(alice)
        session.flush()
        sample = FaceSample(
            snapshot_url="/static/snaps/alice.jpg",
            status=FaceSample.STATUS_EMPLOYEE,
            employee_id=alice.id,
            captured_at=now,
            updated_at=now,
        )
        sample.id = 1
        sample.set_embedding(_make_embedding(vec_a), dim=vec_a.size, model="small")
        session.add(sample)
        session.commit()

    recognizer = EmployeeRecognizer(session_factory, threshold=0.8, encoding_model="small")

    match = recognizer.identify(vec_a)
    assert match is not None
    assert match.employee_name == "Alice"
    assert pytest.approx(match.distance, rel=1e-5) == 0.0

    vec_b = np.zeros(128, dtype=np.float32)
    with session_factory() as session:
        bob = Employee(name="Bob")
        session.add(bob)
        session.flush()
        sample = FaceSample(
            snapshot_url="/static/snaps/bob.jpg",
            status=FaceSample.STATUS_EMPLOYEE,
            employee_id=bob.id,
            captured_at=now,
            updated_at=now,
        )
        sample.id = 2
        sample.set_embedding(_make_embedding(vec_b), dim=vec_b.size, model="small")
        session.add(sample)
        session.commit()

    EmployeeRecognizer.notify_embeddings_updated()
    match_b = recognizer.identify(vec_b)
    assert match_b is not None
    assert match_b.employee_name == "Bob"
