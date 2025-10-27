"""Логика AI-детекции для обработки кадров."""
from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - torch may отсутствовать в окружении
    torch = None

from backend.core.config import settings
from backend.core.logger import logger
from backend.services.snapshots import load_face_cascade, prepare_snapshot

if TYPE_CHECKING:
    from backend.services.employee_recognizer import EmployeeRecognizer, RecognizedEmployee

PHONE_CLASS = "cell phone"


def resolve_device(preferred: Optional[str] = None, cuda_env: Optional[str] = None) -> str:
    """Выбирает устройство для инференса: GPU, если доступно, иначе CPU."""
    if preferred and preferred.strip().lower() not in {"auto", ""}:
        return preferred

    if torch is not None:
        if torch.cuda.is_available():
            if cuda_env:
                first = cuda_env.split(",")[0].strip()
                if first:
                    return first if first.startswith("cuda") else f"cuda:{first}"
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


class AIDetector:
    """Инкапсулирует работу моделей YOLO и эвристики поверх них."""

    def __init__(
        self,
        camera_name: str,
        face_blur: bool,
        visualize: bool,
        *,
        detect_person: bool = True,
        detect_car: bool = True,
        capture_entry_time: bool = True,
        zones: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    ) -> None:
        self.camera_name = camera_name
        self.face_blur = face_blur
        self.visualize = visualize
        self.detect_person = detect_person
        self.detect_car = detect_car
        self.capture_entry_time = capture_entry_time

        det_weights = settings.yolo_det_model
        pose_weights = settings.yolo_pose_model
        self.device_preference = settings.yolo_device
        self.device = resolve_device(self.device_preference, settings.cuda_visible_devices)

        self.det = YOLO(det_weights)
        self.pose = YOLO(pose_weights)
        self.device_error: Optional[str] = None
        self.actual_device: Optional[str] = None
        try:
            self.det.to(self.device)
            self.pose.to(self.device)
        except Exception as exc:
            self.device_error = str(exc).strip() or None
        finally:
            det_model = getattr(self.det, "model", None)
            model_device = getattr(det_model, "device", None)
            if model_device is not None:
                self.actual_device = str(model_device)
            else:
                self.actual_device = str(self.device)

        self.predict_device = None if self.device in {"cpu", "auto"} else self.device

        self.det_imgsz = settings.yolo_image_size
        self.det_conf = settings.phone_det_conf
        self.pose_conf = settings.pose_det_conf
        self.phone_score_threshold = settings.phone_score_threshold
        self.phone_hand_dist_ratio = settings.phone_hand_dist_ratio
        self.phone_head_dist_ratio = settings.phone_head_dist_ratio
        self.pose_only_score_threshold = settings.pose_only_score_threshold
        self.pose_only_head_ratio = settings.pose_only_head_ratio
        self.pose_wrists_dist_ratio = settings.pose_wrists_dist_ratio
        self.pose_tilt_threshold = settings.pose_tilt_threshold
        self.score_smoothing = settings.phone_score_smoothing

        names_map = getattr(self.det.model, "names", None) or getattr(self.det, "names", {})
        self.det_names = {int(k): v for k, v in names_map.items()} if isinstance(names_map, dict) else {}
        self.car_class_ids = {idx for idx, name in self.det_names.items() if name in {"car", "truck", "bus"}}
        self.car_conf_threshold = settings.car_det_conf
        self.min_car_fg_ratio = settings.car_moving_fg_ratio
        self.car_event_cooldown = settings.car_event_cooldown

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        self.ocr_langs = settings.plate_ocr_langs
        self.ocr_reader: Optional[Reader] = None
        self._ocr_failed = False

        self.face_cascade = load_face_cascade() if face_blur else None
        self.phone_idx: Optional[int] = None
        self.last_car_event_time = 0.0
        self.employee_recognizer: Optional["EmployeeRecognizer"] = None
        self._zones: list[list[tuple[float, float]]] = []
        self.update_zones(zones)

    def update_flags(
        self,
        *,
        detect_person: Optional[bool] = None,
        detect_car: Optional[bool] = None,
        capture_entry_time: Optional[bool] = None,
    ) -> None:
        if detect_person is not None:
            self.detect_person = bool(detect_person)
        if detect_car is not None:
            self.detect_car = bool(detect_car)
        if capture_entry_time is not None:
            self.capture_entry_time = bool(capture_entry_time)

    def update_zones(
        self, zones: Optional[Sequence[Sequence[Tuple[float, float]]]] = None
    ) -> None:
        normalized: list[list[tuple[float, float]]] = []
        if zones:
            for polygon in zones:
                try:
                    points = [
                        (
                            max(0.0, min(1.0, float(pt[0]))),
                            max(0.0, min(1.0, float(pt[1]))),
                        )
                        for pt in polygon
                    ]
                except (TypeError, ValueError):
                    continue
                if len(points) >= 3:
                    normalized.append(points)
        self._zones = normalized

    def _point_in_polygon(
        self, x: float, y: float, polygon: Sequence[Tuple[float, float]]
    ) -> bool:
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            intersects = (yi > y) != (yj > y)
            if intersects:
                denom = (yj - yi) or 1e-9
                intersect_x = xi + (y - yi) * (xj - xi) / denom
                if intersect_x > x:
                    inside = not inside
            j = i
        return inside

    def _is_point_in_zones(
        self, x: float, y: float, width: int, height: int
    ) -> bool:
        if not self._zones or width <= 0 or height <= 0:
            return True

        nx = x / float(max(1, width))
        ny = y / float(max(1, height))
        for polygon in self._zones:
            if self._point_in_polygon(nx, ny, polygon):
                return True
        return False

    def set_employee_recognizer(
        self, recognizer: Optional["EmployeeRecognizer"]
    ) -> None:
        """Передаёт сервис распознавания сотрудников."""

        self.employee_recognizer = recognizer

    def runtime_status(self) -> dict:
        preferred = (self.device_preference or "auto").strip() or "auto"
        selected = (self.device or "auto").strip()
        actual = (self.actual_device or selected or "unknown").strip()
        using_gpu = actual.lower().startswith("cuda") or actual.lower().startswith("mps")

        def _cleanup(text: str) -> str:
            cleaned = text.replace("\n", " ").replace("\r", " ").strip()
            if len(cleaned) > 300:
                return cleaned[:297] + "..."
            return cleaned

        reason: Optional[str] = None
        pref_lower = preferred.lower()
        if not using_gpu:
            if pref_lower == "cpu":
                reason = "В настройках указано использовать только CPU (YOLO_DEVICE=cpu)."
            elif self.device_error:
                reason = self.device_error
            elif torch is None:
                reason = "Библиотека PyTorch недоступна внутри контейнера, поэтому GPU не используется."
            elif not torch.cuda.is_available():
                reason = "PyTorch не видит CUDA (torch.cuda.is_available() = False). Проверьте драйвер NVIDIA и nvidia-container-toolkit."
            elif pref_lower in {"", "auto"}:
                reason = "Авто-режим выбрал CPU: перевести модель на GPU не удалось. Проверьте настройки и логи."
            elif pref_lower.startswith("cuda") and not actual.lower().startswith("cuda"):
                reason = "Модель не была загружена на указанное GPU устройство. Проверьте его доступность."

        clean_error = _cleanup(self.device_error) if self.device_error else None
        clean_reason = _cleanup(reason) if reason else None

        return {
            "camera": self.camera_name,
            "preferred_device": preferred,
            "selected_device": selected,
            "actual_device": actual,
            "using_gpu": using_gpu,
            "device_error": clean_error,
            "gpu_unavailable_reason": clean_reason,
        }

    def ensure_ocr_reader(self) -> Optional[Reader]:
        if self._ocr_failed:
            return None
        if self.ocr_reader is None:
            langs = [lang.strip() for lang in self.ocr_langs.split(",") if lang.strip()]
            if not langs:
                langs = ["en"]
            actual_device = (self.actual_device or "").lower()
            use_gpu = bool(
                actual_device.startswith("cuda")
                and torch is not None
                and hasattr(torch, "cuda")
                and callable(getattr(torch.cuda, "is_available", None))
                and torch.cuda.is_available()
            )
            try:
                self.ocr_reader = Reader(langs, gpu=use_gpu)
            except Exception as exc:
                if use_gpu:
                    logger.warning(
                        "[%s] Не удалось инициализировать OCR на GPU, пробуем CPU: %s",
                        self.camera_name,
                        exc,
                        exc_info=True,
                    )
                    try:
                        self.ocr_reader = Reader(langs, gpu=False)
                    except Exception as cpu_exc:
                        logger.warning(
                            "[%s] Не удалось инициализировать OCR даже на CPU: %s",
                            self.camera_name,
                            cpu_exc,
                            exc_info=True,
                        )
                        self._ocr_failed = True
                        self.ocr_reader = None
                        return self.ocr_reader
                    else:
                        return self.ocr_reader
                logger.warning(
                    "[%s] Не удалось инициализировать OCR: %s",
                    self.camera_name,
                    exc,
                    exc_info=True,
                )
                self._ocr_failed = True
                self.ocr_reader = None
        return self.ocr_reader

    def detect_plate_region(self, car_roi):
        try:
            gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h, w = gray.shape[:2]
        total_area = float(h * w)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < 0.01 * total_area or area > 0.35 * total_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) < 4:
                continue
            x, y, w_box, h_box = cv2.boundingRect(approx)
            if h_box == 0:
                continue
            aspect = w_box / float(h_box)
            if 2.0 <= aspect <= 6.5:
                pad_x = int(w_box * 0.1)
                pad_y = int(h_box * 0.2)
                x0 = max(x - pad_x, 0)
                y0 = max(y - pad_y, 0)
                x1 = min(x + w_box + pad_x, car_roi.shape[1])
                y1 = min(y + h_box + pad_y, car_roi.shape[0])
                return car_roi[y0:y1, x0:x1]
        return None

    def ocr_plate(self, plate_img):
        reader = self.ensure_ocr_reader()
        if reader is None:
            return None
        try:
            results = reader.readtext(plate_img)
        except Exception as exc:
            logger.warning(
                "[%s] Ошибка OCR: %s",
                self.camera_name,
                exc,
                exc_info=True,
            )
            return None
        texts = []
        for _bbox, text, conf in results:
            if conf < 0.3:
                continue
            texts.append(text)
        if not texts:
            return None
        raw = "".join(texts)
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        return cleaned.upper() if cleaned else None

    def recognize_plate(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        car_roi = frame[y1:y2, x1:x2]
        if car_roi.size == 0:
            return None
        plate_roi = self.detect_plate_region(car_roi)
        roi = plate_roi if plate_roi is not None else car_roi
        return self.ocr_plate(roi)

    def create_car_snapshot(self, frame, bbox, plate_text):
        snap = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(snap, (x1, y1), (x2, y2), (255, 165, 0), 3)
        label = plate_text if plate_text else "CAR"
        cv2.putText(snap, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        return prepare_snapshot(snap, self.face_blur, self.face_cascade)

    def process_frame(
        self, frame
    ) -> Tuple[bool, float, Any, Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
        detect_person = self.detect_person
        detect_car = self.detect_car

        fg_mask = self.bg_subtractor.apply(frame) if detect_car else None

        det_res = None
        if detect_person or detect_car:
            det_kwargs = {"imgsz": self.det_imgsz, "conf": self.det_conf}
            if self.predict_device:
                det_kwargs["device"] = self.predict_device
            det_res = self.det(frame, **det_kwargs)[0]

        if self.phone_idx is None and det_res is not None and detect_person:
            names = getattr(self.det.model, "names", None) or getattr(self.det, "names", {})
            for k, v in names.items():
                if v == PHONE_CLASS:
                    self.phone_idx = int(k)
                    break

        phones: List[Dict[str, Any]] = []
        cars: List[Dict[str, Any]] = []
        boxes = getattr(det_res, "boxes", None) if det_res is not None else None
        if boxes is not None:
            for b in boxes:
                try:
                    cls_idx = int(b.cls[0])
                except Exception:
                    continue
                conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
                xyxy = b.xyxy[0].cpu().numpy()
                if (
                    detect_person
                    and self.phone_idx is not None
                    and cls_idx == self.phone_idx
                    and conf >= self.det_conf
                ):
                    cx = float((xyxy[0] + xyxy[2]) * 0.5)
                    cy = float((xyxy[1] + xyxy[3]) * 0.5)
                    phones.append({"bbox": xyxy, "center": np.array([cx, cy], dtype=np.float32), "conf": conf})
                if detect_car and cls_idx in self.car_class_ids and conf >= self.car_conf_threshold:
                    cars.append({"bbox": xyxy, "conf": conf})

        pose_res = None
        if detect_person:
            pose_kwargs = {"imgsz": self.det_imgsz, "conf": self.pose_conf}
            if self.predict_device:
                pose_kwargs["device"] = self.predict_device
            pose_res = self.pose(frame, **pose_kwargs)[0]

        phone_usage = False
        best_conf = 0.0
        need_overlay = self.visualize
        vis = frame.copy() if need_overlay else None

        car_events: List[Dict[str, Any]] = []
        now_monotonic = time.monotonic()
        h_frame, w_frame = frame.shape[:2]
        if need_overlay and vis is not None and self._zones:
            zone_color = (65, 105, 225)
            for idx, polygon in enumerate(self._zones, start=1):
                pts = np.array(
                    [
                        [int(pt[0] * w_frame), int(pt[1] * h_frame)]
                        for pt in polygon
                    ],
                    dtype=np.int32,
                )
                if pts.size < 6:
                    continue
                cv2.polylines(vis, [pts], True, zone_color, 2)
                centroid = pts.mean(axis=0)
                cv2.putText(
                    vis,
                    f"Z{idx}",
                    (int(centroid[0]), int(centroid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    zone_color,
                    2,
                )
        for car in cars:
            x1, y1, x2, y2 = map(int, car["bbox"])
            x1 = max(0, min(x1, w_frame - 1))
            x2 = max(0, min(x2, w_frame))
            y1 = max(0, min(y1, h_frame - 1))
            y2 = max(0, min(y2, h_frame))
            if x2 <= x1 or y2 <= y1:
                continue
            center_x = float((x1 + x2) * 0.5)
            center_y = float((y1 + y2) * 0.5)
            if not self._is_point_in_zones(center_x, center_y, w_frame, h_frame):
                continue
            car_mask = fg_mask[y1:y2, x1:x2] if fg_mask is not None else None
            moving_ratio = float(np.mean(car_mask > 0)) if car_mask is not None and car_mask.size else 0.0
            is_moving = moving_ratio >= self.min_car_fg_ratio

            if need_overlay and vis is not None:
                color = (255, 140, 0) if is_moving else (60, 180, 75)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"CAR {car['conf']:.2f}"
                if is_moving:
                    label = f"CAR MOVE {car['conf']:.2f}"
                cv2.putText(vis, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not is_moving:
                continue

            if now_monotonic - self.last_car_event_time < self.car_event_cooldown:
                continue

            plate_text = self.recognize_plate(frame, car["bbox"])
            car_snapshot = self.create_car_snapshot(frame, car["bbox"], plate_text)
            car_events.append({"plate": plate_text, "confidence": float(car["conf"]), "snapshot": car_snapshot})
            self.last_car_event_time = now_monotonic

        people_data: List[Dict[str, Any]] = []

        if not detect_person:
            people_iter = []
        elif pose_res is None or pose_res.boxes is None or pose_res.keypoints is None:
            people_iter = []
        else:
            kpts_xy = pose_res.keypoints.xy.cpu().numpy()
            kpts_conf = None
            if getattr(pose_res.keypoints, "conf", None) is not None:
                kpts_conf = pose_res.keypoints.conf.cpu().numpy()
            boxes_xyxy = pose_res.boxes.xyxy.cpu().numpy()
            raw_ids = getattr(pose_res.boxes, "id", None)
            box_ids = None
            if raw_ids is not None:
                try:
                    if hasattr(raw_ids, "detach"):
                        box_ids = raw_ids.detach().cpu().numpy().reshape(-1)
                    elif hasattr(raw_ids, "cpu"):
                        box_ids = raw_ids.cpu().numpy().reshape(-1)
                    else:
                        box_ids = np.asarray(raw_ids).reshape(-1)
                except Exception:
                    box_ids = None
            id_iter = (
                box_ids.tolist()
                if box_ids is not None
                else [None] * len(kpts_xy)
            )
            people_iter = zip(
                kpts_xy,
                boxes_xyxy,
                kpts_conf if kpts_conf is not None else [None] * len(kpts_xy),
                id_iter,
            )

        recognizer = self.employee_recognizer

        for kpts, bbox, kconf, box_id in people_iter:
            bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
            if bbox_arr.size < 4:
                padded = np.zeros(4, dtype=np.float32)
                padded[: bbox_arr.size] = bbox_arr
                bbox_arr = padded
            center_x = float((bbox_arr[0] + bbox_arr[2]) * 0.5)
            center_y = float((bbox_arr[1] + bbox_arr[3]) * 0.5)
            if not self._is_point_in_zones(center_x, center_y, w_frame, h_frame):
                continue
            try:
                le = kpts[1]
                re = kpts[2]
                head_tilt = abs(le[1] - re[1]) / (abs(le[0] - re[0]) + 1e-3)
            except Exception:
                head_tilt = 0.0

            bbox_h = max(1.0, float(bbox_arr[3] - bbox_arr[1]))

            employee_name = None
            employee_info: Optional[Dict[str, Any]] = None
            if recognizer is not None:
                try:
                    bbox_coords = bbox_arr[:4].tolist()
                    embedding_result = recognizer.compute_embedding(frame, bbox_coords)
                except Exception:
                    logger.exception(
                        "[%s] Ошибка при подготовке эмбеддинга лица",
                        self.camera_name,
                    )
                    embedding_result = None

                if embedding_result is not None:
                    try:
                        match = recognizer.identify(embedding_result)
                    except Exception:
                        logger.exception(
                            "[%s] Ошибка при распознавании сотрудника",
                            self.camera_name,
                        )
                        match = None
                    if match is not None:
                        employee_info = {
                            "id": match.employee_id,
                            "name": match.employee_name,
                            "distance": float(match.distance),
                        }
                        employee_name = match.employee_name

            def point_valid(idx, conf_threshold=0.2):
                if idx >= len(kpts):
                    return False
                if kconf is None or idx >= len(kconf):
                    return True
                return kconf[idx] >= conf_threshold

            wrists = []
            for wrist_idx in (9, 10):
                if point_valid(wrist_idx):
                    wrists.append(np.array(kpts[wrist_idx], dtype=np.float32))

            head_points = []
            for head_idx in (0, 1, 2):
                if point_valid(head_idx):
                    head_points.append(np.array(kpts[head_idx], dtype=np.float32))

            head_center = np.mean(head_points, axis=0) if head_points else None

            pose_heuristic_score = 0.0
            for phone in phones:
                near_hand = False
                if wrists:
                    dists = [np.linalg.norm(phone["center"] - wrist) for wrist in wrists]
                    if dists:
                        rel = min(dists) / bbox_h
                        near_hand = rel <= self.phone_hand_dist_ratio

                near_head = False
                if head_center is not None:
                    rel_head = np.linalg.norm(phone["center"] - head_center) / bbox_h
                    near_head = rel_head <= self.phone_head_dist_ratio

                score = 0.0
                if near_hand:
                    score += 0.6
                if near_head:
                    score += 0.2
                if head_tilt > 0.25:
                    score += 0.1
                score += min(phone["conf"], 1.0) * 0.1

                if score >= self.phone_score_threshold:
                    phone_usage = True
                    best_conf = max(best_conf, score)

                if need_overlay and vis is not None:
                    cx, cy = map(int, phone["center"])
                    color = (0, 0, 255) if score >= self.phone_score_threshold else (0, 165, 255)
                    cv2.circle(vis, (cx, cy), 6, color, -1)
                    if wrists:
                        closest = min(wrists, key=lambda p: np.linalg.norm(phone["center"] - p))
                        cv2.line(vis, (cx, cy), tuple(map(int, closest)), color, 2)
                    if head_center is not None:
                        cv2.line(vis, (cx, cy), tuple(map(int, head_center)), (200, 50, 200), 1)

            if need_overlay and vis is not None:
                x1, y1, x2, y2 = map(int, bbox_arr[:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = employee_name if employee_name else "PERSON"
                cv2.putText(
                    vis,
                    label,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
                for (i, j) in pairs:
                    if i < len(kpts) and j < len(kpts):
                        p1 = tuple(map(int, kpts[i]))
                        p2 = tuple(map(int, kpts[j]))
                        cv2.line(vis, p1, p2, (255, 0, 0), 2)

            if not phones and wrists:
                wrist_head_rel = None
                if head_center is not None:
                    wrist_head_rel = min(np.linalg.norm(head_center - wrist) for wrist in wrists) / bbox_h
                wrist_dist_rel = None
                if len(wrists) >= 2:
                    wrist_dist_rel = np.linalg.norm(wrists[0] - wrists[1]) / bbox_h

                if wrist_head_rel is not None and wrist_head_rel <= self.pose_only_head_ratio:
                    pose_heuristic_score += 0.45
                if wrist_dist_rel is not None and wrist_dist_rel <= self.pose_wrists_dist_ratio:
                    pose_heuristic_score += 0.25
                if head_tilt >= self.pose_tilt_threshold:
                    pose_heuristic_score += 0.2

                if pose_heuristic_score >= self.pose_only_score_threshold:
                    phone_usage = True
                    best_conf = max(best_conf, pose_heuristic_score)
                    if need_overlay and vis is not None:
                        center_y = int((bbox[1] + bbox[3]) * 0.5)
                        cv2.putText(vis, "POSE PHONE", (x1, max(15, center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            mean_conf = float(np.mean(kconf)) if kconf is not None else 0.0
            if box_id is not None:
                person_id = str(int(box_id))
            else:
                key = np.concatenate([bbox, kpts.flatten()])
                key = np.round(key, 1)
                digest = hashlib.sha1(key.tobytes()).hexdigest()[:16]
                person_id = digest
            people_data.append(
                {
                    "id": person_id,
                    "keypoints": kpts.copy(),
                    "confidence": mean_conf,
                    "bbox": bbox_arr[:4].tolist(),
                    "employee_name": employee_name,
                    "employee": employee_info,
                }
            )

        if detect_person and need_overlay and vis is not None:
            for phone in phones:
                x1, y1, x2, y2 = phone["bbox"]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(vis, f"PHONE {phone['conf']:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if detect_person and phone_usage and need_overlay and vis is not None:
            cv2.putText(vis, f"PHONE_USAGE ({best_conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        snapshot = None
        if detect_person:
            snap_src = vis if (self.visualize and vis is not None) else frame
            snapshot = prepare_snapshot(snap_src, self.face_blur, self.face_cascade)

        return (
            phone_usage,
            best_conf,
            snapshot,
            vis if self.visualize else None,
            car_events,
            people_data,
        )
