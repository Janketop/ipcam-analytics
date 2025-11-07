"""Утилиты для инференса ONNX-моделей и совместимости с кодом детектора."""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # pragma: no cover - OpenCV может отсутствовать в окружении тестов
    import cv2
except Exception:  # pragma: no cover - cv2 является необязательным для unit-тестов
    cv2 = None

from backend.core.config import settings
from backend.core.logger import logger

try:  # pragma: no cover - onnxruntime может быть недоступен в окружении тестов
    import onnxruntime as ort
except Exception:  # pragma: no cover - модуль не обязателен для юнит-тестов
    ort = None


class OnnxRuntimeNotAvailableError(RuntimeError):
    """Ошибка, возникающая, если onnxruntime недоступен или не может загрузить модель."""


class TensorWrapper:
    """Минимальная обёртка над ndarray для имитации API PyTorch."""

    __slots__ = ("_array",)

    def __init__(self, array: Sequence[float] | np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float32)

    def cpu(self) -> "TensorWrapper":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def detach(self) -> "TensorWrapper":  # pragma: no cover - используется в бою
        return self

    def reshape(self, *shape: int) -> "TensorWrapper":  # pragma: no cover
        self._array = self._array.reshape(*shape)
        return self


class OnnxBox:
    """Совместимый с Ultralytics контейнер одного бокса."""

    __slots__ = ("cls", "conf", "xyxy")

    def __init__(self, xyxy: Sequence[float], cls_idx: int, conf: float) -> None:
        self.xyxy = [TensorWrapper(xyxy)]
        self.cls = [cls_idx]
        self.conf = [conf]


class OnnxDetectionResult:
    """Упрощённый результат детекции."""

    __slots__ = ("boxes",)

    def __init__(self, boxes: Iterable[OnnxBox]) -> None:
        self.boxes = list(boxes)


class PoseBoxes:
    """Аналог `ultralytics.engine.results.Boxes` для позовой модели."""

    __slots__ = ("xyxy", "conf", "cls", "id")

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: Optional[np.ndarray] = None,
        cls: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
    ) -> None:
        self.xyxy = TensorWrapper(xyxy)
        self.conf = TensorWrapper(conf if conf is not None else np.zeros((xyxy.shape[0],), dtype=np.float32))
        self.cls = TensorWrapper(cls if cls is not None else np.zeros((xyxy.shape[0],), dtype=np.float32))
        self.id = TensorWrapper(ids) if ids is not None else None


class PoseKeypoints:
    """Аналог `ultralytics.engine.results.Keypoints`."""

    __slots__ = ("xy", "conf")

    def __init__(self, xy: np.ndarray, conf: Optional[np.ndarray] = None) -> None:
        self.xy = TensorWrapper(xy)
        self.conf = TensorWrapper(conf) if conf is not None else None


class OnnxPoseResult:
    """Результат инференса позовой модели."""

    __slots__ = ("boxes", "keypoints")

    def __init__(self, boxes: PoseBoxes, keypoints: PoseKeypoints) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


@dataclass(slots=True)
class PreprocessResult:
    input_tensor: np.ndarray
    ratio: Tuple[float, float]
    pad: Tuple[float, float]


def _check_runtime() -> None:
    if ort is None:
        raise OnnxRuntimeNotAvailableError(
            "onnxruntime не установлен. Установите onnxruntime или предоставьте PT-веса."
        )


def _resolve_graph_optimization_level() -> "ort.GraphOptimizationLevel":
    """Подбирает оптимальный уровень оптимизаций графа для onnxruntime."""

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    for candidate in settings.onnx_graph_optimizations:
        key = candidate.strip().lower()
        if key in levels:
            return levels[key]

    return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED


_WARNED_PROVIDERS: Set[str] = set()


@lru_cache(maxsize=1)
def _torch_cuda_is_available() -> bool:
    """Определяет наличие CUDA-устройства через PyTorch (кешируется)."""

    if importlib.util.find_spec("torch") is None:
        return False

    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False

    try:
        is_available = getattr(torch.cuda, "is_available", lambda: False)
        return bool(is_available())
    except Exception:
        return False


def resolve_providers(requested: Optional[Sequence[str]]) -> List[str]:
    """Фильтрует список провайдеров ONNX с учётом доступности CUDA."""

    if not requested:
        requested = ("CPUExecutionProvider",)

    sanitized = [item for item in requested if item]

    available_runtime: Set[str] = set()
    if ort is not None:
        try:
            available_runtime = set(ort.get_available_providers())
        except Exception as exc:  # pragma: no cover - зависит от окружения
            logger.debug(
                "Не удалось получить доступные провайдеры onnxruntime: %s",
                exc,
            )

    filtered: List[str] = []
    cuda_available = _torch_cuda_is_available()

    for provider in sanitized:
        if available_runtime and provider not in available_runtime:
            logger.debug(
                "ONNXRuntime не сообщает провайдера %s, он будет пропущен",
                provider,
            )
            continue

        if provider.upper().startswith("CUDA") and not cuda_available:
            if provider not in _WARNED_PROVIDERS:
                logger.warning(
                    "Провайдер %s отключён: CUDA-устройство не обнаружено. "
                    "Используется CPUExecutionProvider.",
                    provider,
                )
                _WARNED_PROVIDERS.add(provider)
            continue

        if provider not in filtered:
            filtered.append(provider)

    if not filtered:
        fallback = "CPUExecutionProvider"
        if available_runtime and fallback not in available_runtime:
            fallback_candidates = [item for item in available_runtime if item]
            if fallback_candidates:
                fallback = fallback_candidates[0]
        filtered.append(fallback)

    return filtered


def create_session(model_path: str | Path, *, providers: Optional[Sequence[str]] = None) -> "ort.InferenceSession":
    """Создаёт сессию ONNXRuntime с запрошенными провайдерами."""

    _check_runtime()
    resolved = str(Path(model_path).resolve())
    if providers is None:
        providers = settings.onnx_providers

    provider_list = resolve_providers(providers)
    session_options = ort.SessionOptions()
    try:
        session_options.graph_optimization_level = _resolve_graph_optimization_level()
    except Exception as exc:  # pragma: no cover - зависит от сборки onnxruntime
        logger.warning("Не удалось установить уровень оптимизаций ONNX графа: %s", exc)
    last_error: Optional[Exception] = None
    for candidate in provider_list:
        try:
            logger.debug("Инициализация ONNXRuntime (%s) для %s", candidate, resolved)
            return ort.InferenceSession(
                resolved,
                providers=[candidate],
                sess_options=session_options,
            )
        except Exception as exc:  # pragma: no cover - зависит от окружения
            last_error = exc
            logger.warning("ONNXRuntime провайдер %s недоступен: %s", candidate, exc)
            continue
    if last_error:
        raise OnnxRuntimeNotAvailableError(
            f"Не удалось инициализировать onnxruntime для {resolved}: {last_error}"
        )
    raise OnnxRuntimeNotAvailableError(
        f"Не удалось инициализировать onnxruntime для {resolved}: провайдеры не заданы"
    )


def letterbox_resize(image: np.ndarray, size: int) -> PreprocessResult:
    """Простейшая реализация letterbox-препроцесса."""

    if cv2 is None:  # pragma: no cover - fallback для окружений без OpenCV
        raise OnnxRuntimeNotAvailableError("opencv-python недоступен для препроцесса ONNX")

    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h)) if (new_w, new_h) != (w, h) else image.copy()
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = np.ascontiguousarray(canvas)
    input_tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return PreprocessResult(input_tensor=input_tensor, ratio=(scale, scale), pad=(pad_x, pad_y))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Простейшая реализация non-max suppression."""

    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        idx = order[0]
        keep.append(int(idx))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[idx] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode_yolo_output(
    output: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    ratio: Tuple[float, float],
    pad: Tuple[float, float],
    image_shape: Tuple[int, int],
) -> List[OnnxBox]:
    """Декодирует результат YOLO-подобной ONNX модели."""

    if output.ndim == 3:
        output = np.squeeze(output, axis=0)
    if output.ndim != 2:
        return []

    rows, cols = output.shape

    # Определяем, где находятся признаки (5 + C) и где количество предсказаний.
    # В большинстве моделей YOLO число признаков заметно меньше числа боксов,
    # однако встречаются модели, экспортированные в формат (5 + C, N). Для них
    # поворачиваем матрицу, сохранив исходный поддерживаемый вариант "N × (5+C)".
    if cols < 6 and rows >= 6:
        output = output.T
        rows, cols = output.shape
    elif rows < cols:
        feature_guess = rows - 5
        aspect = cols / max(rows, 1)
        if 1 <= feature_guess <= 4096 and aspect >= 8:
            output = output.T
            rows, cols = output.shape

    if cols < 6:
        return []

    sample = output[: min(256, rows)]
    xyxy_layout = bool(
        sample.size >= 4
        and np.all(sample[:, 2] >= sample[:, 0])
        and np.all(sample[:, 3] >= sample[:, 1])
    )

    if xyxy_layout:
        x1 = output[:, 0]
        y1 = output[:, 1]
        x2 = output[:, 2]
        y2 = output[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
    else:
        cx = output[:, 0]
        cy = output[:, 1]
        w = output[:, 2]
        h = output[:, 3]

    obj_conf = output[:, 4]
    class_scores = output[:, 5:]

    best_class = class_scores.argmax(axis=1)
    best_score = class_scores.max(axis=1)
    conf = obj_conf * best_score
    mask = conf >= conf_threshold
    if not np.any(mask):
        return []

    cx = cx[mask]
    cy = cy[mask]
    w = w[mask]
    h = h[mask]
    conf = conf[mask]
    best_class = best_class[mask]

    ratio_x, ratio_y = ratio
    pad_x, pad_y = pad

    x1 = (cx - w / 2 - pad_x) / ratio_x
    y1 = (cy - h / 2 - pad_y) / ratio_y
    x2 = (cx + w / 2 - pad_x) / ratio_x
    y2 = (cy + h / 2 - pad_y) / ratio_y

    h_img, w_img = image_shape
    x1 = np.clip(x1, 0, w_img)
    y1 = np.clip(y1, 0, h_img)
    x2 = np.clip(x2, 0, w_img)
    y2 = np.clip(y2, 0, h_img)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = _nms(boxes, conf, iou_threshold)

    detections: List[OnnxBox] = []
    for idx in keep:
        detections.append(OnnxBox(boxes[idx], int(best_class[idx]), float(conf[idx])))
    return detections


def decode_pose_output(
    boxes: np.ndarray,
    keypoints: np.ndarray,
    *,
    ratio: Tuple[float, float],
    pad: Tuple[float, float],
    image_shape: Tuple[int, int],
    num_keypoints: int,
) -> OnnxPoseResult:
    """Подготавливает структуру результата позового детектора."""

    if boxes.ndim == 3:
        boxes = np.squeeze(boxes, axis=0)
    if boxes.ndim != 2 or boxes.shape[1] < 5:
        boxes = np.zeros((0, 6), dtype=np.float32)
    if keypoints.ndim == 3:
        keypoints = np.squeeze(keypoints, axis=0)

    ratio_x, ratio_y = ratio
    pad_x, pad_y = pad
    h_img, w_img = image_shape

    xyxy: List[List[float]] = []
    conf: List[float] = []
    kp_xy: List[List[List[float]]] = []
    kp_conf: List[List[float]] = []

    for idx, row in enumerate(boxes):
        score = float(row[4]) if row.size > 4 else 0.0
        if score <= 0.0:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = (cx - bw / 2 - pad_x) / ratio_x
        y1 = (cy - bh / 2 - pad_y) / ratio_y
        x2 = (cx + bw / 2 - pad_x) / ratio_x
        y2 = (cy + bh / 2 - pad_y) / ratio_y
        x1 = float(np.clip(x1, 0, w_img))
        y1 = float(np.clip(y1, 0, h_img))
        x2 = float(np.clip(x2, 0, w_img))
        y2 = float(np.clip(y2, 0, h_img))
        xyxy.append([x1, y1, x2, y2])
        conf.append(score)
        if idx < keypoints.shape[0]:
            kps = keypoints[idx]
            coords: List[List[float]] = []
            scores: List[float] = []
            for kp_idx in range(num_keypoints):
                base = kp_idx * 3
                if base + 2 >= kps.size:
                    break
                kx = (kps[base] - pad_x) / ratio_x
                ky = (kps[base + 1] - pad_y) / ratio_y
                kc = kps[base + 2]
                coords.append([float(np.clip(kx, 0, w_img)), float(np.clip(ky, 0, h_img))])
                scores.append(float(kc))
            kp_xy.append(coords)
            kp_conf.append(scores)

    if not xyxy:
        boxes_tensor = np.zeros((0, 4), dtype=np.float32)
        conf_tensor = np.zeros((0,), dtype=np.float32)
        kp_xy_tensor = np.zeros((0, num_keypoints, 2), dtype=np.float32)
        kp_conf_tensor = np.zeros((0, num_keypoints), dtype=np.float32)
    else:
        boxes_tensor = np.asarray(xyxy, dtype=np.float32)
        conf_tensor = np.asarray(conf, dtype=np.float32)
        kp_xy_tensor = np.zeros((len(kp_xy), num_keypoints, 2), dtype=np.float32)
        kp_conf_tensor = np.zeros((len(kp_conf), num_keypoints), dtype=np.float32)
        for idx, coords in enumerate(kp_xy):
            for kp_idx, coord in enumerate(coords):
                if kp_idx < num_keypoints:
                    kp_xy_tensor[idx, kp_idx] = coord
        for idx, scores in enumerate(kp_conf):
            for kp_idx, score in enumerate(scores):
                if kp_idx < num_keypoints:
                    kp_conf_tensor[idx, kp_idx] = score

    pose_boxes = PoseBoxes(boxes_tensor, conf_tensor)
    pose_keypoints = PoseKeypoints(kp_xy_tensor, kp_conf_tensor)
    return OnnxPoseResult(pose_boxes, pose_keypoints)


class OnnxYoloDetector:
    """Высокоуровневый адаптер для детекторов объектов в формате ONNX."""

    def __init__(self, model_path: str, *, class_names: Sequence[str]) -> None:
        self.model_path = str(model_path)
        self.class_names = list(class_names)
        self.names = {idx: name for idx, name in enumerate(self.class_names)}
        self.session = None
        self.model = type("DummyOnnxModel", (), {"names": self.names})()
        self.init_error: Optional[str] = None
        try:
            self.session = create_session(self.model_path)
        except OnnxRuntimeNotAvailableError as exc:
            self.init_error = str(exc)
            logger.error("ONNXRuntime недоступен для %s: %s", self.model_path, exc)

    def __call__(self, frame: np.ndarray, *, imgsz: int, conf: float, **_: object) -> List[OnnxDetectionResult]:
        if self.session is None:
            return [OnnxDetectionResult([])]
        prep = letterbox_resize(frame, imgsz)
        try:
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: prep.input_tensor})
        except Exception as exc:  # pragma: no cover - зависит от модели
            logger.exception("Ошибка выполнения ONNX детектора %s", self.model_path)
            raise OnnxRuntimeNotAvailableError(str(exc)) from exc
        detections = decode_yolo_output(
            outputs[0],
            conf_threshold=conf,
            iou_threshold=0.45,
            ratio=prep.ratio,
            pad=prep.pad,
            image_shape=frame.shape[:2],
        )
        return [OnnxDetectionResult(detections)]


class OnnxPoseEstimator:
    """Адаптер для позовых моделей в формате ONNX."""

    def __init__(self, model_path: str, *, kpt_shape: Tuple[int, int]) -> None:
        self.model_path = str(model_path)
        self.kpt_shape = kpt_shape
        self.session = None
        self.model = type("PoseModel", (), {"model": type("Inner", (), {"kpt_shape": kpt_shape})()})()
        self.init_error: Optional[str] = None
        try:
            self.session = create_session(self.model_path)
        except OnnxRuntimeNotAvailableError as exc:
            self.init_error = str(exc)
            logger.error(
                "ONNXRuntime недоступен для позовой модели %s: %s", self.model_path, exc
            )

    def __call__(self, frame: np.ndarray, *, imgsz: int, conf: float, **_: object) -> List[OnnxPoseResult]:
        if self.session is None:
            empty = PoseBoxes(np.zeros((0, 4), dtype=np.float32))
            keypoints = PoseKeypoints(np.zeros((0, self.kpt_shape[0], 2), dtype=np.float32))
            return [OnnxPoseResult(empty, keypoints)]
        prep = letterbox_resize(frame, imgsz)
        try:
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: prep.input_tensor})
        except Exception as exc:  # pragma: no cover
            logger.exception("Ошибка выполнения ONNX позовой модели %s", self.model_path)
            raise OnnxRuntimeNotAvailableError(str(exc)) from exc
        boxes = outputs[0]
        keypoints = outputs[1] if len(outputs) > 1 else np.zeros((boxes.shape[0], self.kpt_shape[0] * 3), dtype=np.float32)
        decoded = decode_pose_output(
            boxes,
            keypoints,
            ratio=prep.ratio,
            pad=prep.pad,
            image_shape=frame.shape[:2],
            num_keypoints=self.kpt_shape[0],
        )
        return [decoded]


class OnnxClassifier:
    """Минимальная обёртка для произвольного классификатора."""

    def __init__(self, model_path: str, *, class_names: Sequence[str]) -> None:
        self.model_path = str(model_path)
        self.class_names = list(class_names)
        self.session = None
        self.init_error: Optional[str] = None
        try:
            self.session = create_session(self.model_path)
        except OnnxRuntimeNotAvailableError as exc:
            self.init_error = str(exc)
            logger.error(
                "ONNXRuntime недоступен для классификатора %s: %s", self.model_path, exc
            )

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.session is None:
            return np.zeros((len(features), len(self.class_names)), dtype=np.float32)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: np.asarray(features, dtype=np.float32)})
        return np.asarray(outputs[0], dtype=np.float32)


__all__ = [
    "OnnxRuntimeNotAvailableError",
    "TensorWrapper",
    "OnnxBox",
    "OnnxDetectionResult",
    "OnnxPoseResult",
    "OnnxYoloDetector",
    "OnnxPoseEstimator",
    "OnnxClassifier",
    "letterbox_resize",
    "decode_yolo_output",
    "decode_pose_output",
    "resolve_providers",
]
