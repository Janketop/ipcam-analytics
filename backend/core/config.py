"""Конфигурационные параметры приложения."""
from __future__ import annotations

import os

from backend.utils.env import env_flag

FACE_BLUR = env_flag("FACE_BLUR", False)
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "7"))
CLEANUP_INTERVAL_HOURS = float(os.getenv("RETENTION_CLEANUP_INTERVAL_HOURS", "6"))
APP_TITLE = "IPCam Analytics (RU)"
