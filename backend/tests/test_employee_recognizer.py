from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.models import Employee, FaceSample
from backend.models.base import Base
import backend.services.employee_recognizer as employee_recognizer_module
from backend.services.employee_recognizer import EmployeeRecognizer
from backend.services.face_embeddings import (
    FaceEmbeddingResult,
    get_embedding_metadata,
    normalize_encoding_model_name,
)


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
    metadata = get_embedding_metadata()
    canonical = normalize_encoding_model_name(metadata["model"])
    dim = metadata["embedding_dim"]
    vec_a = np.ones(dim, dtype=np.float32)

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
        sample.set_embedding(_make_embedding(vec_a), dim=vec_a.size, model=canonical)
        session.add(sample)
        session.commit()

    recognizer = EmployeeRecognizer(
        session_factory,
        threshold=0.8,
        encoding_model=canonical,
    )

    match = recognizer.identify(vec_a)
    assert match is not None
    assert match.employee_name == "Alice"
    assert pytest.approx(match.distance, rel=1e-5) == 0.0
    assert match.backend == "numpy"
    assert match.metric == "euclidean"

    vec_b = np.zeros(dim, dtype=np.float32)
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
        sample.set_embedding(_make_embedding(vec_b), dim=vec_b.size, model=canonical)
        session.add(sample)
        session.commit()

    EmployeeRecognizer.notify_embeddings_updated()
    match_b = recognizer.identify(vec_b)
    assert match_b is not None
    assert match_b.employee_name == "Bob"
    assert match_b.backend == "numpy"
    assert match_b.metric == "euclidean"


def test_recognizer_rebuilds_embeddings_when_backend_changes(monkeypatch, session_factory):
    now = datetime.now(timezone.utc)

    arcface_name = normalize_encoding_model_name("arcface_insightface")
    arcface_meta = get_embedding_metadata("arcface_insightface")
    arcface_dim = arcface_meta["embedding_dim"]
    arcface_vec = np.ones(arcface_dim, dtype=np.float32)

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
        sample.set_embedding(
            _make_embedding(arcface_vec),
            dim=arcface_vec.size,
            model=arcface_name,
        )
        session.add(sample)
        session.commit()

    dlib_meta = get_embedding_metadata("dlib_resnet")
    dlib_dim = dlib_meta["embedding_dim"]
    dlib_vec = np.ones(dlib_dim, dtype=np.float32)
    dlib_vec /= np.linalg.norm(dlib_vec)

    def fake_compute(snapshot_url: str, *, encoding_model=None, detection_model="hog", num_jitters=1):
        assert encoding_model == "dlib_resnet"
        return FaceEmbeddingResult(
            vector=np.array(dlib_vec, dtype=np.float32, copy=True),
            model="dlib_resnet",
        )

    monkeypatch.setattr(
        employee_recognizer_module,
        "compute_face_embedding_from_snapshot",
        fake_compute,
    )

    recognizer = EmployeeRecognizer(
        session_factory,
        threshold=0.8,
        encoding_model="dlib_resnet",
    )

    match = recognizer.identify(dlib_vec)
    assert match is not None
    assert match.employee_name == "Alice"

    with session_factory() as session:
        sample = session.get(FaceSample, 1)
        assert sample is not None
        assert sample.embedding_model == "dlib_resnet"
        assert sample.embedding_dim == dlib_dim
        restored = FaceEmbeddingResult.from_bytes(
            sample.embedding,
            dim=sample.embedding_dim,
            model=sample.embedding_model,
        )
        assert restored is not None
        assert np.allclose(restored.vector, dlib_vec, atol=1e-6)
