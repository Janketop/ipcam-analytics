"""Распознавание сотрудников на основе эталонных снимков лиц."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.logger import logger
from backend.core.paths import BACKEND_DIR, SNAPSHOT_DIR
from backend.models import Employee, FaceSample
from backend.services.snapshots import load_face_cascade


def _resolve_snapshot_path(snapshot_url: str) -> Optional[Path]:
    """Преобразует URL снимка в путь на файловой системе."""

    if not snapshot_url:
        return None

    cleaned = snapshot_url.lstrip("/")
    if not cleaned:
        return None

    candidate = BACKEND_DIR.parent / cleaned
    if candidate.exists():
        return candidate

    filename = Path(cleaned).name
    fallback = SNAPSHOT_DIR / filename
    if fallback.exists():
        return fallback

    return None


@dataclass
class _ReferenceSample:
    """Внутреннее представление эталонного снимка."""

    name: str
    embedding: np.ndarray


class EmployeeRecognizer:
    """Сравнивает текущие лица с эталонными снимками сотрудников."""

    def __init__(
        self,
        *,
        threshold: float = 0.75,
        face_detector: Optional[cv2.CascadeClassifier] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.face_detector = face_detector or load_face_cascade()
        self._samples: list[_ReferenceSample] = []
        self._known_names: set[str] = set()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def employee_count(self) -> int:
        return len(self._known_names)

    @property
    def has_samples(self) -> bool:
        return bool(self._samples)

    def add_sample(self, name: str, image_bgr: np.ndarray) -> bool:
        """Добавляет эталонный снимок сотрудника."""

        if image_bgr is None or image_bgr.size == 0:
            return False

        embedding = self._compute_embedding(image_bgr)
        if embedding is None:
            logger.debug("Не удалось вычислить embedding для сотрудника %s", name)
            return False

        self._samples.append(_ReferenceSample(name=name, embedding=embedding))
        self._known_names.add(name)
        return True

    def identify(self, frame_bgr: np.ndarray, bbox: Sequence[float] | np.ndarray | None) -> Optional[str]:
        """Ищет сотрудника, наиболее похожего на лицо в кадре."""

        if not self._samples or frame_bgr is None or frame_bgr.size == 0:
            return None

        roi = self._extract_roi(frame_bgr, bbox)
        if roi is None:
            return None

        embedding = self._compute_embedding(roi)
        if embedding is None:
            return None

        best_name = None
        best_score = -1.0
        for sample in self._samples:
            score = float(np.dot(sample.embedding, embedding))
            if score > best_score:
                best_score = score
                best_name = sample.name

        if best_name is None or best_score < self.threshold:
            return None

        return best_name

    @classmethod
    def from_session(
        cls,
        session: Session,
        *,
        threshold: float = 0.75,
    ) -> Optional["EmployeeRecognizer"]:
        """Создаёт распознаватель на основе данных БД."""

        stmt = (
            select(FaceSample.snapshot_url, Employee.name)
            .join(Employee, FaceSample.employee)
            .where(
                FaceSample.status == FaceSample.STATUS_EMPLOYEE,
                FaceSample.snapshot_url.is_not(None),
            )
        )
        rows = session.execute(stmt).all()
        if not rows:
            return None

        recognizer = cls(threshold=threshold)
        loaded = 0
        for snapshot_url, employee_name in rows:
            path = _resolve_snapshot_path(snapshot_url)
            if path is None:
                logger.warning(
                    "Не удалось найти файл снимка сотрудника %s по пути %s",
                    employee_name,
                    snapshot_url,
                )
                continue
            image = cv2.imread(str(path))
            if image is None:
                logger.warning(
                    "Не удалось загрузить изображение %s для сотрудника %s",
                    path,
                    employee_name,
                )
                continue
            if recognizer.add_sample(employee_name, image):
                loaded += 1

        if not recognizer.has_samples:
            return None

        logger.info(
            "Загружено %d эталонных изображений для %d сотрудников",
            loaded,
            recognizer.employee_count,
        )
        return recognizer

    def _extract_roi(
        self,
        frame_bgr: np.ndarray,
        bbox: Sequence[float] | np.ndarray | None,
    ) -> Optional[np.ndarray]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        if bbox is not None:
            h, w = frame_bgr.shape[:2]
            try:
                x1, y1, x2, y2 = map(float, bbox)
            except Exception:
                x1 = y1 = 0.0
                x2, y2 = float(w), float(h)
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            pad_x = width * 0.1
            pad_y = height * 0.1
            x1 = int(max(0, min(x1 - pad_x, w - 1)))
            y1 = int(max(0, min(y1 - pad_y, h - 1)))
            x2 = int(max(0, min(x2 + pad_x, w)))
            y2 = int(max(0, min(y2 + pad_y, h)))
            if x2 <= x1 or y2 <= y1:
                roi = frame_bgr
            else:
                roi = frame_bgr[y1:y2, x1:x2]
        else:
            roi = frame_bgr

        if roi is None or roi.size == 0:
            return None

        return roi

    def _compute_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            logger.debug("Не удалось конвертировать изображение в grayscale", exc_info=True)
            return None

        face_region = gray
        detector = self.face_detector
        if detector is not None:
            try:
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            except Exception:
                faces = ()
            if faces is not None and len(faces) > 0:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                face_region = gray[y : y + h, x : x + w]

        if face_region is None or face_region.size == 0:
            return None

        resized = cv2.resize(face_region, (96, 96), interpolation=cv2.INTER_AREA)
        vector = resized.astype(np.float32).flatten()
        vector -= float(np.mean(vector))
        std = float(np.std(vector))
        if std < 1e-6:
            return None
        vector /= std
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return vector / norm


__all__ = ["EmployeeRecognizer"]
