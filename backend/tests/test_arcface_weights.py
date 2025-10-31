from pathlib import Path
from zipfile import ZipFile

from backend.services.arcface_weights import ensure_arcface_weights, validate_arcface_model


def test_validate_arcface_model_detects_small(tmp_path):
    dummy = tmp_path / "dummy.onnx"
    dummy.write_bytes(b"ONNX")

    ok, reason = validate_arcface_model(dummy)
    assert not ok
    assert "слишком маленький" in reason


def test_ensure_arcface_weights_uses_local_file(tmp_path):
    destination = tmp_path / "arcface.onnx"
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX" + b"\x00" * 4096)

    success = ensure_arcface_weights(destination, sources=[source])

    assert success
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()


def test_ensure_arcface_weights_extracts_from_zip(tmp_path):
    destination = tmp_path / "arcface.onnx"
    archive = tmp_path / "weights.zip"

    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("models/w600k_r50.onnx", b"ONNX" + b"\x01" * 8192)

    success = ensure_arcface_weights(destination, sources=[archive])

    assert success
    assert destination.exists()
    assert destination.read_bytes().startswith(b"ONNX")
