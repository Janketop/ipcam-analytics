"""Утилиты для безопасного чтения переменных окружения."""
from __future__ import annotations

from typing import Optional

from backend.core.config import settings


def _get_setting_value(name: str, default):
    attr = name.lower()
    return getattr(settings, attr, default)


def env_flag(name: str, default: bool) -> bool:
    raw = _get_setting_value(name, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, min_value: Optional[int] = None) -> int:
    raw = _get_setting_value(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(value, min_value)
    return value


def env_float(name: str, default: float, min_value: Optional[float] = None) -> float:
    raw = _get_setting_value(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(value, min_value)
    return value
