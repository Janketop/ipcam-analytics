"""Определение основных путей приложения."""
from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BACKEND_DIR / "static"
SNAPSHOT_DIR = STATIC_DIR / "snaps"
LOG_DIR = BACKEND_DIR / "logs"
WEIGHTS_DIR = BACKEND_DIR / "weights"
DATASET_PHONE_USAGE_DIR = BACKEND_DIR.parent / "dataset" / "phone_usage"
DATASET_FACE_DETECTION_DIR = BACKEND_DIR.parent / "dataset" / "widerface"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PHONE_USAGE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_FACE_DETECTION_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "BACKEND_DIR",
    "STATIC_DIR",
    "SNAPSHOT_DIR",
    "LOG_DIR",
    "WEIGHTS_DIR",
    "DATASET_PHONE_USAGE_DIR",
    "DATASET_FACE_DETECTION_DIR",
]
