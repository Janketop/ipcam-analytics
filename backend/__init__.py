"""Пакет с backend-приложением FastAPI."""

__all__ = ["app", "get_app"]


def __getattr__(name: str):  # pragma: no cover - ленивый импорт для тестового окружения
    if name in {"app", "get_app"}:
        from .main import app, get_app  # pylint: disable=import-outside-toplevel

        return {"app": app, "get_app": get_app}[name]
    raise AttributeError(name)
