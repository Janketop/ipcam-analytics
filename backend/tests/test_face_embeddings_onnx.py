from types import SimpleNamespace

import numpy as np

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
