"""Сервисные функции для расчёта эмбеддингов лиц."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from backend.core.config import settings
from backend.core.logger import logger
from backend.core.paths import BACKEND_DIR, SNAPSHOT_DIR
from backend.services import face_embeddings_onnx

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


@dataclass(frozen=True, slots=True)
class _EmbeddingSpec:
    """Описание поддерживаемой модели для расчёта эмбеддингов."""

    canonical_name: str
    input_size: tuple[int, int]
    embedding_dim: int


_DEFAULT_MODEL_KEY = "arcface_insightface"

_MODEL_SPECS: dict[str, _EmbeddingSpec] = {
    _DEFAULT_MODEL_KEY: _EmbeddingSpec(
        canonical_name=_DEFAULT_MODEL_KEY,
        input_size=(112, 112),
        embedding_dim=512,
    ),
}

_MODEL_ALIASES: dict[str, str] = {
    "": _DEFAULT_MODEL_KEY,
    _DEFAULT_MODEL_KEY: _DEFAULT_MODEL_KEY,
    "arcface": _DEFAULT_MODEL_KEY,
    "insightface": _DEFAULT_MODEL_KEY,
    "r100": _DEFAULT_MODEL_KEY,
}


def _resolve_model_key(name: Optional[str]) -> str:
    raw = (name or "").strip().lower()
    if raw in _MODEL_SPECS:
        return raw
    if raw in _MODEL_ALIASES:
        return _MODEL_ALIASES[raw]

    if raw:  # pragma: no cover - логирование выполняется только при неизвестном имени
        logger.warning(
            "Неизвестная модель эмбеддингов %s. Используется %s",
            name,
            _DEFAULT_MODEL_KEY,
        )

    return _DEFAULT_MODEL_KEY


def normalize_encoding_model_name(name: Optional[str]) -> str:
    """Возвращает каноничное имя модели эмбеддингов."""

    key = _resolve_model_key(name)
    return _MODEL_SPECS[key].canonical_name


def _get_model_spec(name: Optional[str]) -> _EmbeddingSpec:
    key = _resolve_model_key(name)
    return _MODEL_SPECS[key]


def _embedding_backend_ready() -> bool:
    return face_embeddings_onnx.backend_ready()


def _run_embedding(
    image: np.ndarray,
    spec: _EmbeddingSpec,
    *,
    assume_bgr: bool,
) -> Optional[np.ndarray]:
    if not _embedding_backend_ready():
        return None

    if image is None or image.size == 0:
        return None

    try:
        rgb = _ensure_rgb(image, assume_bgr=assume_bgr)
    except ValueError:
        logger.debug("Не удалось привести изображение лица к RGB формату")
        return None

    try:
        vector = face_embeddings_onnx.compute_embedding(
            rgb, input_size=spec.input_size
        )
    except Exception:
        logger.exception("Ошибка при вычислении эмбеддинга лица ONNX-моделью")
        return None

    return np.asarray(vector, dtype=np.float32)


def _build_embedding_result(
    vector: Optional[np.ndarray],
    spec: _EmbeddingSpec,
    *,
    location: Optional[tuple[int, int, int, int]],
) -> Optional[FaceEmbeddingResult]:
    if vector is None or vector.size == 0:
        return None

    return FaceEmbeddingResult(
        vector=np.array(vector, dtype=np.float32, copy=True),
        model=spec.canonical_name,
        location=location,
    )


def get_embedding_metadata(name: Optional[str] = None) -> dict[str, object]:
    """Возвращает метаданные о поддерживаемой модели эмбеддингов."""

    spec = _get_model_spec(name)
    return {
        "model": spec.canonical_name,
        "input_size": spec.input_size,
        "embedding_dim": spec.embedding_dim,
        "providers": tuple(settings.onnx_providers),
        "onnx_model": str(settings.face_recognition_onnx_model_path),
    }


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

    if not _embedding_backend_ready():
        return None

    spec = _get_model_spec(encoding_model or settings.face_recognition_model)

    if image is None or image.size == 0:
        return None

    try:
        rgb = _ensure_rgb(image, assume_bgr=assume_bgr)
    except ValueError:
        logger.debug("Не удалось подготовить изображение для поиска лица")
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
    top, right, bottom, left = location
    top = max(0, int(top))
    left = max(0, int(left))
    bottom = max(top + 1, int(bottom))
    right = max(left + 1, int(right))

    face_region = rgb[top:bottom, left:right]
    if face_region.size == 0:
        logger.debug("Выделенный регион лица пуст")
        return None

    embedding = _run_embedding(face_region, spec, assume_bgr=False)
    return _build_embedding_result(embedding, spec, location=location)


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

    if not _embedding_backend_ready():
        return None

    spec = _get_model_spec(encoding_model or settings.face_recognition_model)

    roi = _crop_with_padding(frame_bgr, bbox or [])
    if roi is None or roi.size == 0:
        return None

    embedding = _run_embedding(roi, spec, assume_bgr=True)
    return _build_embedding_result(embedding, spec, location=None)


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
    "get_embedding_metadata",
    "normalize_encoding_model_name",
    "compute_face_embedding_for_bbox",
    "compute_face_embedding_from_array",
    "compute_face_embedding_from_snapshot",
    "load_image",
]
