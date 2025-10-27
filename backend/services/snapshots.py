"""Работа со снимками событий."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4
import os

import cv2

from backend.core.logger import logger
from backend.core.paths import SNAPSHOT_DIR


def load_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """Загружает каскад Хаара для последующего размытия лиц."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if cascade_path and os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
    except Exception as exc:
        logger.exception("Не удалось загрузить каскад Хаара: %s", exc)
        return None
    return None


def prepare_snapshot(img_bgr, face_blur: bool, face_cascade: Optional[cv2.CascadeClassifier]):
    """Копирует кадр и размывает лица, если это разрешено настройками."""
    if img_bgr is None:
        return None
    snap = img_bgr.copy()
    if face_blur and face_cascade is not None:
        try:
            gray = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = snap[y : y + h, x : x + w]
                roi = cv2.GaussianBlur(roi, (99, 99), 30)
                snap[y : y + h, x : x + w] = roi
        except Exception:
            logger.exception("Ошибка при размытии лиц на снимке")
    return snap


def save_snapshot(img_bgr, ts: datetime, camera_name: str, event_type: str = "event") -> str:
    """Сохраняет изображение на диск и возвращает относительный URL."""
    if img_bgr is None:
        return ""
    filename = f"{camera_name}_{event_type}_{int(ts.timestamp())}_{uuid4().hex[:6]}.jpg"
    path = SNAPSHOT_DIR / filename
    success = cv2.imwrite(str(path), img_bgr)
    if success:
        logger.info("Снимок сохранён: %s", path)
    else:
        logger.warning("Не удалось сохранить снимок %s", path)
    return f"/static/snaps/{filename}"
