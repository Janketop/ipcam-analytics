import numpy as np
from types import SimpleNamespace

from backend.services import face_embeddings
from backend.services import face_embeddings_onnx
from backend.services.face_embeddings import FaceEmbeddingResult


def _setup_backend(monkeypatch, vector: np.ndarray) -> None:
    monkeypatch.setattr(face_embeddings_onnx, "backend_ready", lambda: True)
    monkeypatch.setattr(face_embeddings, "face_embeddings_onnx", face_embeddings_onnx)
    monkeypatch.setattr(
        face_embeddings_onnx,
        "compute_embedding",
        lambda image_rgb, input_size: np.array(vector, dtype=np.float32),
    )


def test_compute_face_embedding_from_array_success(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(face_embeddings_onnx, "backend_ready", lambda: True)
    canonical = face_embeddings.normalize_encoding_model_name("arcface")
    metadata = face_embeddings.get_embedding_metadata(canonical)
    embedding_dim = metadata["embedding_dim"]
    fake_vector = np.ones(embedding_dim, dtype=np.float32)
    fake_vector /= np.linalg.norm(fake_vector)

    fake_module = SimpleNamespace(
        face_locations=lambda img, model=None: [(0, img.shape[1], img.shape[0], 0)],
    )
    monkeypatch.setattr(face_embeddings, "face_recognition", fake_module)
    monkeypatch.setattr(face_embeddings, "_FACE_RECOGNITION_ERROR", None)
    _setup_backend(monkeypatch, fake_vector)

    result = face_embeddings.compute_face_embedding_from_array(
        image,
        encoding_model="arcface",
        detection_model="hog",
        assume_bgr=True,
    )

    assert isinstance(result, FaceEmbeddingResult)
    assert result.dimension == embedding_dim
    assert result.model == canonical
    assert np.allclose(result.vector, fake_vector)
    assert np.isclose(np.linalg.norm(result.vector), 1.0)


def test_compute_face_embedding_for_bbox(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    bbox = (10, 10, 40, 40)

    monkeypatch.setattr(face_embeddings_onnx, "backend_ready", lambda: True)
    metadata = face_embeddings.get_embedding_metadata()
    embedding_dim = metadata["embedding_dim"]
    fake_vector = np.full(embedding_dim, 0.5, dtype=np.float32)
    fake_vector /= np.linalg.norm(fake_vector)

    _setup_backend(monkeypatch, fake_vector)

    result = face_embeddings.compute_face_embedding_for_bbox(
        frame,
        bbox,
        encoding_model=metadata["model"],
    )

    assert isinstance(result, FaceEmbeddingResult)
    assert result.location is None
    assert np.allclose(result.vector, fake_vector)


def test_compute_face_embedding_from_array_no_face(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    fake_module = SimpleNamespace(
        face_locations=lambda img, model=None: [],
    )
    monkeypatch.setattr(face_embeddings, "face_recognition", fake_module)
    monkeypatch.setattr(face_embeddings, "_FACE_RECOGNITION_ERROR", None)
    monkeypatch.setattr(face_embeddings_onnx, "backend_ready", lambda: True)

    result = face_embeddings.compute_face_embedding_from_array(
        image,
        encoding_model="arcface",
        detection_model="hog",
        assume_bgr=True,
    )

    assert result is None


def test_compute_face_embedding_fallback_to_dlib(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    bbox = (5, 5, 40, 40)

    monkeypatch.setattr(face_embeddings_onnx, "backend_ready", lambda: False)

    vector = np.ones(128, dtype=np.float32)
    vector /= np.linalg.norm(vector)

    fake_module = SimpleNamespace(
        face_encodings=lambda img, known_face_locations=None, num_jitters=1: [
            np.array(vector, dtype=np.float32)
        ],
    )
    monkeypatch.setattr(face_embeddings, "face_recognition", fake_module)
    monkeypatch.setattr(face_embeddings, "_FACE_RECOGNITION_ERROR", None)

    result = face_embeddings.compute_face_embedding_for_bbox(
        frame,
        bbox,
        encoding_model="arcface",
    )

    assert isinstance(result, FaceEmbeddingResult)
    assert result.model == "dlib_resnet"
    assert result.dimension == vector.size
    assert np.isclose(np.linalg.norm(result.vector), 1.0)
