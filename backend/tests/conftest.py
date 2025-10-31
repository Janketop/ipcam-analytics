from pathlib import Path

import pytest

from backend.services import face_embeddings_onnx
from backend.services.arcface_weights import (
    ensure_arcface_weights as _real_ensure,
    validate_arcface_model as _real_validate,
)


@pytest.fixture(autouse=True)
def _prevent_real_arcface_download(monkeypatch):
    """Не даёт тестам перезаписывать файл-заглушку ArcFace в репозитории."""

    def _safe_ensure(path: Path) -> bool:
        if Path(path).name == "face_recognition_arcface_dummy.onnx":
            return True
        return _real_ensure(path)

    def _safe_validate(path: Path):
        if Path(path).name == "face_recognition_arcface_dummy.onnx" and Path(path).exists():
            return True, ""
        return _real_validate(path)

    monkeypatch.setattr(face_embeddings_onnx, "ensure_arcface_weights", _safe_ensure)
    monkeypatch.setattr(face_embeddings_onnx, "validate_arcface_model", _safe_validate)
