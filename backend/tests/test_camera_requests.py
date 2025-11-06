"""Проверки валидаторов запросов добавления камер."""

from __future__ import annotations

import importlib
import sys
import types

import pytest
from pydantic import ValidationError


def _import_camera_module():
    """Импортирует модуль с маршрутами камер, подменяя зависимости onnx."""

    checker_stub = types.SimpleNamespace(check_model=lambda *args, **kwargs: None)
    onnx_stub = types.SimpleNamespace(load=lambda *args, **kwargs: None, checker=checker_stub)
    sys.modules["onnx"] = onnx_stub
    sys.modules["onnx.checker"] = checker_stub

    for name in (
        "backend.api.routes.cameras",
        "backend.core.config",
        "backend.services.arcface_weights",
    ):
        sys.modules.pop(name, None)

    return importlib.import_module("backend.api.routes.cameras")


def test_create_request_allows_http_scheme():
    module = _import_camera_module()
    CameraCreateRequest = module.CameraCreateRequest

    payload = {
        "name": "Вход",
        "rtsp_url": "http://example.com/mjpeg",
    }

    data = CameraCreateRequest(**payload)

    assert str(data.rtsp_url) == payload["rtsp_url"]


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/feed",
        "ws://example.com/stream",
        "file:///tmp/video.mp4",
    ],
)
def test_create_request_rejects_unsupported_scheme(url: str):
    module = _import_camera_module()
    CameraCreateRequest = module.CameraCreateRequest

    with pytest.raises(ValidationError):
        CameraCreateRequest(name="Склад", rtsp_url=url)
