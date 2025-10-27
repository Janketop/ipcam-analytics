from types import SimpleNamespace

import numpy as np

from backend.services import face_embeddings
from backend.services.face_embeddings import FaceEmbeddingResult


def test_compute_face_embedding_from_array_success(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    fake_module = SimpleNamespace(
        face_locations=lambda img, model=None: [(0, img.shape[1], img.shape[0], 0)],
        face_encodings=lambda *args, **kwargs: [np.ones(128, dtype=np.float64)],
    )
    monkeypatch.setattr(face_embeddings, "face_recognition", fake_module)
    monkeypatch.setattr(face_embeddings, "_FACE_RECOGNITION_ERROR", None)

    result = face_embeddings.compute_face_embedding_from_array(
        image,
        encoding_model="small",
        detection_model="hog",
        assume_bgr=True,
    )

    assert isinstance(result, FaceEmbeddingResult)
    assert result.dimension == 128
    assert result.model == "small"
    assert result.vector.dtype == np.float32


def test_compute_face_embedding_from_array_no_face(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    fake_module = SimpleNamespace(
        face_locations=lambda img, model=None: [],
        face_encodings=lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(face_embeddings, "face_recognition", fake_module)
    monkeypatch.setattr(face_embeddings, "_FACE_RECOGNITION_ERROR", None)

    result = face_embeddings.compute_face_embedding_from_array(
        image,
        encoding_model="small",
        detection_model="hog",
        assume_bgr=True,
    )

    assert result is None
