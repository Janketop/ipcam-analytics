"""Сервисные функции для расчёта эмбеддингов лиц."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from backend.core.config import settings
from backend.core.logger import logger
from backend.core.paths import BACKEND_DIR, SNAPSHOT_DIR

try:  # pragma: no cover - импорт может падать только в рантайме контейнера
    import face_recognition  # type: ignore
except Exception as exc:  # pragma: no cover - обработка отсутствия зависимости
    face_recognition = None  # type: ignore
    _FACE_RECOGNITION_ERROR = exc
else:  # pragma: no cover - выполняется только при успешном импорте
    _FACE_RECOGNITION_ERROR = None


@dataclass(slots=True)
class FaceEmbeddingResult:
    """Результат вычисления эмбеддинга лица."""

    vector: np.ndarray
    model: str
    location: Optional[tuple[int, int, int, int]] = None

    @property
    def dimension(self) -> int:
        return int(self.vector.shape[0])

    def as_bytes(self) -> bytes:
        return np.asarray(self.vector, dtype=np.float32).tobytes()

    @classmethod
    def from_bytes(
        cls, data: Optional[bytes], *, dim: Optional[int], model: Optional[str]
    ) -> Optional["FaceEmbeddingResult"]:
        """Восстанавливает эмбеддинг из байтового представления."""

        if data is None or dim is None or dim <= 0 or not model:
            return None

        array = np.frombuffer(data, dtype=np.float32)
        if array.size < dim:
            logger.warning(
                "Содержимое эмбеддинга меньше ожидаемого размера: %s < %s",
                array.size,
                dim,
            )
            return None

        vector = np.array(array[:dim], dtype=np.float32, copy=True)
        return cls(vector=vector, model=model)


def _resolve_snapshot_path(snapshot_url: str) -> Optional[Path]:
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


def load_image(snapshot_url: str) -> Optional[np.ndarray]:
    """Загружает снимок по URL и возвращает его в формате RGB."""

    if face_recognition is None:  # pragma: no cover - зависит от окружения
        logger.error(
            "Модуль face_recognition недоступен: %s",
            _FACE_RECOGNITION_ERROR,
        )
        return None

    path = _resolve_snapshot_path(snapshot_url)
    if path is None:
        logger.warning("Файл снимка %s не найден на диске", snapshot_url)
        return None

    try:
        return face_recognition.load_image_file(str(path))
    except FileNotFoundError:
        logger.warning("Файл %s исчез до загрузки", path)
    except Exception:  # pragma: no cover - ошибки чтения файлов
        logger.exception("Не удалось загрузить изображение лица %s", path)
    return None


def _ensure_rgb(image: np.ndarray, *, assume_bgr: bool) -> np.ndarray:
    """Преобразует входной массив в RGB и гарантирует непрерывность памяти."""

    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] >= 3:
        if assume_bgr:
            rgb = image[..., :3][:, :, ::-1]
        else:
            rgb = image[..., :3]
    else:
        raise ValueError("Некорректный формат изображения для вычисления эмбеддинга")

    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _largest_bbox(locations: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    def area(item: tuple[int, int, int, int]) -> int:
        top, right, bottom, left = item
        return max(0, bottom - top) * max(0, right - left)

    return max(locations, key=area)


def compute_face_embedding_from_array(
    image: np.ndarray,
    *,
    encoding_model: Optional[str] = None,
    detection_model: str = "hog",
    num_jitters: int = 1,
    assume_bgr: bool = True,
) -> Optional[FaceEmbeddingResult]:
    """Вычисляет эмбеддинг лица для массива пикселей."""

    if face_recognition is None:  # pragma: no cover - зависит от окружения
        logger.error(
            "Нельзя вычислить эмбеддинг: библиотека face_recognition недоступна",
        )
        return None

    if image is None or image.size == 0:
        return None

    try:
        rgb = _ensure_rgb(image, assume_bgr=assume_bgr)
    except ValueError:
        logger.debug("Не удалось подготовить изображение для face_recognition")
        return None

    try:
        locations = face_recognition.face_locations(rgb, model=detection_model)
    except Exception:  # pragma: no cover - ошибки детектора лиц
        logger.exception("Ошибка face_recognition.face_locations")
        return None

    if not locations:
        logger.info("Лицо не найдено на переданном изображении")
        return None

    location = _largest_bbox(locations)
    model_name = (encoding_model or settings.face_recognition_model or "small").strip()
    if not model_name:
        model_name = "small"

    try:
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=[location],
            num_jitters=int(max(1, num_jitters)),
            model=model_name,
        )
    except Exception:  # pragma: no cover - ошибки face_recognition
        logger.exception("Не удалось вычислить эмбеддинг лица")
        return None

    if not encodings:
        logger.info("face_recognition не вернул эмбеддинг для обнаруженного лица")
        return None

    vector = np.asarray(encodings[0], dtype=np.float32)
    return FaceEmbeddingResult(vector=vector, model=model_name, location=location)


def _clip_bbox(value: float, minimum: float, maximum: float) -> int:
    return int(max(minimum, min(maximum, value)))


def _crop_with_padding(
    frame: np.ndarray,
    bbox: Sequence[float],
    *,
    padding: float = 0.15,
) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None

    if bbox is None or len(bbox) < 4:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox[:4])
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * padding
    pad_y = height * padding

    left = _clip_bbox(x1 - pad_x, 0, w - 1)
    top = _clip_bbox(y1 - pad_y, 0, h - 1)
    right = _clip_bbox(x2 + pad_x, 0, w)
    bottom = _clip_bbox(y2 + pad_y, 0, h)

    if right <= left or bottom <= top:
        return frame

    return frame[top:bottom, left:right]


def compute_face_embedding_for_bbox(
    frame_bgr: np.ndarray,
    bbox: Sequence[float] | None,
    *,
    encoding_model: Optional[str] = None,
    detection_model: str = "hog",
    num_jitters: int = 1,
    padding: float = 0.15,
) -> Optional[FaceEmbeddingResult]:
    """Вычисляет эмбеддинг лица в ROI, заданном прямоугольником."""

    roi = _crop_with_padding(frame_bgr, bbox or [])
    if roi is None or roi.size == 0:
        return None

    return compute_face_embedding_from_array(
        roi,
        encoding_model=encoding_model,
        detection_model=detection_model,
        num_jitters=num_jitters,
        assume_bgr=True,
    )


def compute_face_embedding_from_snapshot(
    snapshot_url: str,
    *,
    encoding_model: Optional[str] = None,
    detection_model: str = "hog",
    num_jitters: int = 1,
) -> Optional[FaceEmbeddingResult]:
    """Загружает снимок по URL и возвращает эмбеддинг лица."""

    image = load_image(snapshot_url)
    if image is None:
        return None

    return compute_face_embedding_from_array(
        image,
        encoding_model=encoding_model,
        detection_model=detection_model,
        num_jitters=num_jitters,
        assume_bgr=False,
    )


__all__ = [
    "FaceEmbeddingResult",
    "compute_face_embedding_for_bbox",
    "compute_face_embedding_from_array",
    "compute_face_embedding_from_snapshot",
    "load_image",
]
