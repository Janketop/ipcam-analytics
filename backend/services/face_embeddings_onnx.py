"""Обёртка над onnxruntime для расчёта эмбеддингов лиц."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

from backend.core.config import settings
from backend.core.logger import logger

try:  # pragma: no cover - зависит от окружения выполнения
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - в тестах onnxruntime может отсутствовать
    ort = None  # type: ignore[assignment]
    _ONNXRUNTIME_IMPORT_ERROR = exc
else:  # pragma: no cover - выполняется только при успешном импорте
    _ONNXRUNTIME_IMPORT_ERROR = None


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Невозможно нормализовать нулевой эмбеддинг")
    return vector / norm


def _prepare_batch(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Изображение должно иметь три канала")

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    else:  # pragma: no cover - ветка для старых версий Pillow
        resample = Image.BILINEAR  # type: ignore[attr-defined]

    pil_image = Image.fromarray(image)
    resized = pil_image.resize(size, resample=resample)
    array = np.asarray(resized, dtype=np.float32)
    array = (array / 127.5) - 1.0
    chw = np.transpose(array, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    return np.ascontiguousarray(batch, dtype=np.float32)


def _ensure_model_file_ready(path: Path) -> None:
    """Проверяет, что ONNX-файл выглядит как настоящая модель, а не заглушка."""

    if not path.exists():
        raise RuntimeError(
            "Файл ONNX-модели эмбеддингов лиц %s не найден. "
            "Укажите верный путь в переменной окружения "
            "ONNX_FACE_CLASSIFIER_MODEL или FACE_RECOGNITION_ONNX_MODEL." % path
        )

    size = path.stat().st_size
    if size < 1024:
        raise RuntimeError(
            "Файл %s слишком маленький (%d байт) и выглядит как заглушка. "
            "Замените его на реальные веса ArcFace в формате ONNX." % (path, size)
        )

    with path.open("rb") as handle:
        header = handle.read(4)

    if header != b"ONNX":
        raise RuntimeError(
            "Файл %s не похож на ONNX-модель (ожидается заголовок 'ONNX'). "
            "Замените его корректным файлом весов." % path
        )


@lru_cache(maxsize=1)
def _get_session() -> "ort.InferenceSession":
    if ort is None:
        raise RuntimeError(
            "onnxruntime недоступен: %s" % (_ONNXRUNTIME_IMPORT_ERROR or "unknown")
        )

    model_path = settings.face_recognition_onnx_model_path
    providers: Iterable[str]
    providers = settings.onnx_providers or ("CPUExecutionProvider",)

    try:
        _ensure_model_file_ready(model_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise

    session_options = ort.SessionOptions()
    try:
        session = ort.InferenceSession(  # type: ignore[no-untyped-call]
            str(model_path),
            providers=list(providers),
            sess_options=session_options,
        )
    except Exception as exc:  # pragma: no cover - ошибки загрузки зависят от окружения
        logger.exception("Не удалось создать onnxruntime.InferenceSession")
        raise RuntimeError("Не удалось загрузить ONNX-модель эмбеддингов лиц") from exc

    return session


def backend_ready() -> bool:
    if ort is None:
        logger.error(
            "onnxruntime недоступен для расчёта эмбеддингов лиц: %s",
            _ONNXRUNTIME_IMPORT_ERROR,
        )
        return False

    try:
        _get_session()
    except Exception:
        return False
    return True


def compute_embedding(image_rgb: np.ndarray, *, input_size: Tuple[int, int]) -> np.ndarray:
    session = _get_session()
    batch = _prepare_batch(image_rgb, input_size)

    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX-модель не имеет входов")

    input_name = inputs[0].name
    outputs = session.get_outputs()
    output_names = [item.name for item in outputs]

    try:
        result = session.run(output_names, {input_name: batch})
    except Exception as exc:  # pragma: no cover - зависит от окружения
        logger.exception("Ошибка выполнения ONNX-модели для эмбеддингов лиц")
        raise RuntimeError("Не удалось выполнить ONNX-модель эмбеддингов лиц") from exc

    if not result:
        raise RuntimeError("ONNX-модель не вернула результатов")

    vector = np.asarray(result[0], dtype=np.float32).reshape(-1)
    normalized = _normalize(vector)
    return np.asarray(normalized, dtype=np.float32)


__all__ = ["backend_ready", "compute_embedding"]

