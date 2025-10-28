from types import SimpleNamespace

import numpy as np
import torch

from backend.services import face_embeddings
from backend.services.face_embeddings import FaceEmbeddingResult


def test_compute_face_embedding_from_array_success(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    canonical = face_embeddings.normalize_encoding_model_name("small")
    metadata = face_embeddings.get_embedding_metadata(canonical)

    fake_module = SimpleNamespace(
        face_locations=lambda img, model=None: [(0, img.shape[1], img.shape[0], 0)],
    )
    monkeypatch.setattr(face_embeddings, "face_recognition", fake_module)
    monkeypatch.setattr(face_embeddings, "_FACE_RECOGNITION_ERROR", None)
    monkeypatch.setattr(face_embeddings, "_embedding_backend_ready", lambda: True)
    monkeypatch.setattr(face_embeddings, "_get_device", lambda: torch.device("cpu"))

    input_size = metadata["input_size"]
    fake_tensor = torch.ones((3, input_size[0], input_size[1]), dtype=torch.float32)
    monkeypatch.setattr(
        face_embeddings,
        "_prepare_face_tensor",
        lambda img, spec, assume_bgr: fake_tensor,
    )
    monkeypatch.setattr(
        face_embeddings,
        "_run_embedding",
        lambda tensor, spec: torch.ones(metadata["embedding_dim"], dtype=torch.float32),
    )

    result = face_embeddings.compute_face_embedding_from_array(
        image,
        encoding_model="small",
        detection_model="hog",
        assume_bgr=True,
    )

    assert isinstance(result, FaceEmbeddingResult)
    assert result.dimension == metadata["embedding_dim"]
    assert result.model == canonical
    assert np.allclose(result.vector, 1.0)
    assert result.vector.dtype == np.float32


def test_compute_face_embedding_from_array_no_face(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    fake_module = SimpleNamespace(
        face_locations=lambda img, model=None: [],
    )
    monkeypatch.setattr(face_embeddings, "face_recognition", fake_module)
    monkeypatch.setattr(face_embeddings, "_FACE_RECOGNITION_ERROR", None)
    monkeypatch.setattr(face_embeddings, "_embedding_backend_ready", lambda: True)

    result = face_embeddings.compute_face_embedding_from_array(
        image,
        encoding_model="small",
        detection_model="hog",
        assume_bgr=True,
    )

    assert result is None
