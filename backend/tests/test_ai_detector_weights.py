import pathlib
import sys


class _Cv2Stub:
    @staticmethod
    def setNumThreads(*_args, **_kwargs) -> None:
        return None

    class cuda:
        @staticmethod
        def getCudaEnabledDeviceCount() -> int:
            return 0

    __version__ = "0.0"

    @staticmethod
    def getBuildInformation() -> str:  # pragma: no cover - используется при импорте Ultralytics
        return "cv2 stub for tests"

    def __getattr__(self, _name: str):  # pragma: no cover - для непредвиденных обращений
        def _stub(*_args, **_kwargs):
            return None

        return _stub


sys.modules.setdefault("cv2", _Cv2Stub())

import pytest

from backend.core.config import settings
from backend.services import ai_detector


@pytest.fixture()
def restore_settings(monkeypatch):
    original_model = settings.yolo_face_model
    original_url = settings.yolo_face_model_url
    try:
        yield
    finally:
        monkeypatch.setattr(settings, "yolo_face_model", original_model)
        monkeypatch.setattr(settings, "yolo_face_model_url", original_url)


def test_candidate_urls_order(monkeypatch):
    monkeypatch.setattr(
        ai_detector,
        "_fetch_github_face_weight_urls",
        lambda: [
            "https://example.org/yolo11n.pt",
            "https://example.org/yolo11n.pt",
            "https://example.org/yolov8n.pt",
        ],
    )
    urls = ai_detector._candidate_face_weight_urls("https://manual/url.pt")
    expected_prefix = [
        "https://manual/url.pt",
        "https://example.org/yolo11n.pt",
        "https://example.org/yolov8n.pt",
    ]
    assert urls[: len(expected_prefix)] == expected_prefix
    assert len(urls) == len(set(urls))


def test_ensure_face_weights_returns_existing(tmp_path, monkeypatch, restore_settings):
    weight_path = tmp_path / "face.pt"
    weight_path.write_bytes(b"ok")
    monkeypatch.setattr(settings, "yolo_face_model", str(weight_path))
    monkeypatch.setattr(settings, "yolo_face_model_url", "")

    resolved = ai_detector._ensure_face_weights()
    assert resolved == str(weight_path)


def test_ensure_face_weights_downloads(monkeypatch, tmp_path, restore_settings):
    destination = tmp_path / "face.pt"
    monkeypatch.setattr(settings, "yolo_face_model", str(destination))
    monkeypatch.setattr(settings, "yolo_face_model_url", "")

    downloaded: list[str] = []

    def fake_download(url: str, dest: pathlib.Path) -> None:
        downloaded.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")

    monkeypatch.setattr(ai_detector, "_download_file", fake_download)
    monkeypatch.setattr(
        ai_detector,
        "_candidate_face_weight_urls",
        lambda manual: ["https://primary.example/face.pt", "https://secondary.example/face.pt"],
    )

    resolved = ai_detector._ensure_face_weights()
    assert resolved == str(destination)
    assert downloaded == ["https://primary.example/face.pt"]
