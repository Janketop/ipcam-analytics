import io
import tarfile
from pathlib import Path
from zipfile import ZipFile

from onnx import TensorProto, helper

from backend.services.arcface_weights import ensure_arcface_weights, validate_arcface_model


def _dummy_onnx_bytes(vector_size: int = 2048) -> bytes:
    dims = [1, vector_size]
    weights = helper.make_tensor(
        "W", TensorProto.FLOAT, dims, [0.0] * vector_size
    )
    node = helper.make_node("Add", inputs=["X", "W"], outputs=["Y"])
    graph = helper.make_graph(
        [node],
        "test_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, dims)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, dims)],
        [weights],
    )
    model = helper.make_model(graph, producer_name="tests")
    return model.SerializeToString()


def test_validate_arcface_model_detects_small(tmp_path):
    dummy = tmp_path / "dummy.onnx"
    dummy.write_bytes(b"ONNX")

    ok, reason = validate_arcface_model(dummy)
    assert not ok
    assert "слишком маленький" in reason


def test_ensure_arcface_weights_uses_local_file(tmp_path):
    destination = tmp_path / "arcface.onnx"
    source = tmp_path / "source.onnx"
    source.write_bytes(_dummy_onnx_bytes())

    success = ensure_arcface_weights(destination, sources=[source])

    assert success
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()


def test_ensure_arcface_weights_extracts_from_zip(tmp_path):
    destination = tmp_path / "arcface.onnx"
    archive = tmp_path / "weights.zip"

    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("models/w600k_r50.onnx", _dummy_onnx_bytes())

    success = ensure_arcface_weights(destination, sources=[archive])

    assert success
    assert destination.exists()
    assert validate_arcface_model(destination)[0]


def test_ensure_arcface_weights_extracts_from_tar(tmp_path):
    destination = tmp_path / "arcface.onnx"
    archive = tmp_path / "weights.tar.gz"

    data = _dummy_onnx_bytes()
    info = tarfile.TarInfo("models/glint360k_r100.onnx")
    info.size = len(data)

    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))

    success = ensure_arcface_weights(destination, sources=[archive])

    assert success
    assert destination.exists()
    assert validate_arcface_model(destination)[0]
