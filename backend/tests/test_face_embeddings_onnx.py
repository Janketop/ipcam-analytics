from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.services import face_embeddings_onnx


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def get_inputs(self):
        return [SimpleNamespace(name="input_tensor")]

    def get_outputs(self):
        return [SimpleNamespace(name="embedding")]

    def run(self, output_names, feeds):
        assert output_names == ["embedding"]
        assert "input_tensor" in feeds
        return [self._output.copy()]


def test_compute_embedding_normalizes_output(monkeypatch):
    raw_output = np.arange(1, 5, dtype=np.float32)
    fake_session = _FakeSession(raw_output)
    monkeypatch.setattr(face_embeddings_onnx, "_get_session", lambda: fake_session)

    image = np.ones((112, 112, 3), dtype=np.uint8)
    result = face_embeddings_onnx.compute_embedding(image, input_size=(112, 112))

    assert result.shape == (4,)
    assert np.isclose(np.linalg.norm(result), 1.0)
    expected = raw_output / np.linalg.norm(raw_output)
    assert np.allclose(result, expected)


def test_ensure_model_file_ready_missing(monkeypatch, tmp_path):
    missing = tmp_path / "missing.onnx"

    monkeypatch.setattr(
        face_embeddings_onnx,
        "ensure_arcface_weights",
        lambda path, **_: False,
    )

    with pytest.raises(RuntimeError) as excinfo:
        face_embeddings_onnx._ensure_model_file_ready(missing)  # type: ignore[attr-defined]

    assert "не удалось подготовить" in str(excinfo.value)


def test_ensure_model_file_ready_too_small(monkeypatch, tmp_path):
    dummy = tmp_path / "dummy.onnx"
    dummy.write_bytes(b"ONNX")

    monkeypatch.setattr(
        face_embeddings_onnx,
        "ensure_arcface_weights",
        lambda path, **_: False,
    )

    with pytest.raises(RuntimeError) as excinfo:
        face_embeddings_onnx._ensure_model_file_ready(dummy)  # type: ignore[attr-defined]

    assert "не удалось подготовить" in str(excinfo.value)


def test_ensure_model_file_ready_invalid_header(monkeypatch, tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"FAKE" + b"\x00" * 2048)

    monkeypatch.setattr(
        face_embeddings_onnx,
        "ensure_arcface_weights",
        lambda path, **_: False,
    )

    with pytest.raises(RuntimeError) as excinfo:
        face_embeddings_onnx._ensure_model_file_ready(bad)  # type: ignore[attr-defined]

    assert "не удалось подготовить" in str(excinfo.value)


def test_ensure_model_file_ready_valid(tmp_path):
    valid = tmp_path / "valid.onnx"
    valid.write_bytes(b"ONNX" + b"\x00" * 2048)

    face_embeddings_onnx._ensure_model_file_ready(valid)  # type: ignore[attr-defined]


def test_ensure_model_file_ready_downloads_when_possible(monkeypatch, tmp_path):
    destination = tmp_path / "auto.onnx"

    def _fake_downloader(target: Path, *, sources):
        target.write_bytes(b"ONNX" + b"\x01" * 4096)
        return True

    monkeypatch.setattr(face_embeddings_onnx, "ensure_arcface_weights", _fake_downloader)

    face_embeddings_onnx._ensure_model_file_ready(destination)  # type: ignore[attr-defined]
    assert destination.exists()
    assert destination.read_bytes().startswith(b"ONNX")


def test_ensure_model_file_ready_respects_settings(monkeypatch, tmp_path):
    destination = tmp_path / "custom.onnx"

    captured: dict[str, tuple[str, ...]] = {}

    def _fake_downloader(target: Path, *, sources):
        captured["sources"] = tuple(sources)
        target.write_bytes(b"ONNX" + b"\x02" * 4096)
        return True

    monkeypatch.setattr(face_embeddings_onnx, "ensure_arcface_weights", _fake_downloader)
    monkeypatch.setattr(
        face_embeddings_onnx.settings,
        "arcface_weights_sources",
        ("https://example.com/custom.onnx",),
    )

    face_embeddings_onnx._ensure_model_file_ready(destination)  # type: ignore[attr-defined]

    assert destination.exists()
    assert captured["sources"] == ("https://example.com/custom.onnx",)
