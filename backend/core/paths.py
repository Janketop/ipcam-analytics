"""Определение основных путей приложения."""
from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BACKEND_DIR / "static"
SNAPSHOT_DIR = STATIC_DIR / "snaps"
LOG_DIR = BACKEND_DIR / "logs"
DATASET_DIR = BACKEND_DIR.parent / "dataset"
DATASET_PHONE_USAGE_DIR = DATASET_DIR / "phone_usage"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PHONE_USAGE_DIR.mkdir(parents=True, exist_ok=True)
