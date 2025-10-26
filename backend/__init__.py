"""Пакет с backend-приложением FastAPI."""

from .main import app, get_app  # noqa: F401

__all__ = ["app", "get_app"]
