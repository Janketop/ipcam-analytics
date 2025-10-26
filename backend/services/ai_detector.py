"""Логика AI-детекции для обработки кадров."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - torch may отсутствовать в окружении
    torch = None

from backend.services.snapshots import load_face_cascade, prepare_snapshot
from backend.utils.env import env_float, env_int

PHONE_CLASS = "cell phone"


def resolve_device(preferred: Optional[str] = None) -> str:
    """Выбирает устройство для инференса: GPU, если доступно, иначе CPU."""
    if preferred and preferred.strip().lower() not in {"auto", ""}:
        return preferred

    if torch is not None:
        if torch.cuda.is_available():
            cuda_env = os.getenv("CUDA_VISIBLE_DEVICES")
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

    def __init__(self, camera_name: str, face_blur: bool, visualize: bool) -> None:
        self.camera_name = camera_name
        self.face_blur = face_blur
        self.visualize = visualize

        det_weights = os.getenv("YOLO_DET_MODEL", "yolov8n.pt")
        pose_weights = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
        self.device_preference = os.getenv("YOLO_DEVICE", "auto")
        self.device = resolve_device(self.device_preference)

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

        self.det_imgsz = env_int("YOLO_IMAGE_SIZE", 640, min_value=32)
        self.det_conf = env_float("PHONE_DET_CONF", 0.3, min_value=0.05)
        self.pose_conf = env_float("POSE_DET_CONF", 0.3, min_value=0.05)
        self.phone_score_threshold = env_float("PHONE_SCORE_THRESHOLD", 0.6, min_value=0.1)
        self.phone_hand_dist_ratio = env_float("PHONE_HAND_DIST_RATIO", 0.35, min_value=0.05)
        self.phone_head_dist_ratio = env_float("PHONE_HEAD_DIST_RATIO", 0.45, min_value=0.05)
        self.pose_only_score_threshold = env_float("POSE_ONLY_SCORE_THRESHOLD", 0.55, min_value=0.1)
        self.pose_only_head_ratio = env_float("POSE_ONLY_HEAD_RATIO", 0.5, min_value=0.05)
        self.pose_wrists_dist_ratio = env_float("POSE_WRISTS_DIST_RATIO", 0.25, min_value=0.05)
        self.pose_tilt_threshold = env_float("POSE_TILT_THRESHOLD", 0.22, min_value=0.05)
        self.score_smoothing = env_int("PHONE_SCORE_SMOOTHING", 5, min_value=1)

        names_map = getattr(self.det.model, "names", None) or getattr(self.det, "names", {})
        self.det_names = {int(k): v for k, v in names_map.items()} if isinstance(names_map, dict) else {}
        self.car_class_ids = {idx for idx, name in self.det_names.items() if name in {"car", "truck", "bus"}}
        self.car_conf_threshold = env_float("CAR_DET_CONF", 0.35, min_value=0.05)
        self.min_car_fg_ratio = env_float("CAR_MOVING_FG_RATIO", 0.05, min_value=0.0)
        self.car_event_cooldown = env_float("CAR_EVENT_COOLDOWN", 8.0, min_value=1.0)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        self.ocr_langs = os.getenv("PLATE_OCR_LANGS", "ru,en")
        self.ocr_reader: Optional[Reader] = None
        self._ocr_failed = False

        self.face_cascade = load_face_cascade() if face_blur else None
        self.phone_idx: Optional[int] = None
        self.last_car_event_time = 0.0

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
            try:
                self.ocr_reader = Reader(langs, gpu=False)
            except Exception as exc:
                print(f"[{self.camera_name}] не удалось инициализировать OCR: {exc}")
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
            print(f"[{self.camera_name}] ошибка OCR: {exc}")
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

    def process_frame(self, frame) -> Tuple[bool, float, Any, Any, List[Dict[str, Any]]]:
        fg_mask = self.bg_subtractor.apply(frame)

        det_kwargs = {"imgsz": self.det_imgsz, "conf": self.det_conf}
        if self.predict_device:
            det_kwargs["device"] = self.predict_device
        det_res = self.det(frame, **det_kwargs)[0]
        if self.phone_idx is None:
            names = getattr(self.det.model, "names", None) or getattr(self.det, "names", {})
            for k, v in names.items():
                if v == PHONE_CLASS:
                    self.phone_idx = int(k)
                    break

        phones: List[Dict[str, Any]] = []
        cars: List[Dict[str, Any]] = []
        boxes = getattr(det_res, "boxes", None)
        if boxes is not None:
            for b in boxes:
                try:
                    cls_idx = int(b.cls[0])
                except Exception:
                    continue
                conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
                xyxy = b.xyxy[0].cpu().numpy()
                if self.phone_idx is not None and cls_idx == self.phone_idx and conf >= self.det_conf:
                    cx = float((xyxy[0] + xyxy[2]) * 0.5)
                    cy = float((xyxy[1] + xyxy[3]) * 0.5)
                    phones.append({"bbox": xyxy, "center": np.array([cx, cy], dtype=np.float32), "conf": conf})
                if cls_idx in self.car_class_ids and conf >= self.car_conf_threshold:
                    cars.append({"bbox": xyxy, "conf": conf})

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
        for car in cars:
            x1, y1, x2, y2 = map(int, car["bbox"])
            x1 = max(0, min(x1, w_frame - 1))
            x2 = max(0, min(x2, w_frame))
            y1 = max(0, min(y1, h_frame - 1))
            y2 = max(0, min(y2, h_frame))
            if x2 <= x1 or y2 <= y1:
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

        if pose_res.boxes is None or pose_res.keypoints is None:
            people_iter = []
        else:
            kpts_xy = pose_res.keypoints.xy.cpu().numpy()
            kpts_conf = None
            if getattr(pose_res.keypoints, "conf", None) is not None:
                kpts_conf = pose_res.keypoints.conf.cpu().numpy()
            people_iter = zip(
                kpts_xy,
                pose_res.boxes.xyxy.cpu().numpy(),
                kpts_conf if kpts_conf is not None else [None] * len(kpts_xy),
            )

        for kpts, bbox, kconf in people_iter:
            try:
                le = kpts[1]
                re = kpts[2]
                head_tilt = abs(le[1] - re[1]) / (abs(le[0] - re[0]) + 1e-3)
            except Exception:
                head_tilt = 0.0

            bbox_h = max(1.0, float(bbox[3] - bbox[1]))

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
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, "PERSON", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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

        if need_overlay and vis is not None:
            for phone in phones:
                x1, y1, x2, y2 = phone["bbox"]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(vis, f"PHONE {phone['conf']:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if phone_usage and need_overlay and vis is not None:
            cv2.putText(vis, f"PHONE_USAGE ({best_conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        snap_src = vis if (self.visualize and vis is not None) else frame
        snapshot = prepare_snapshot(snap_src, self.face_blur, self.face_cascade)

        return phone_usage, best_conf, snapshot, vis if self.visualize else None, car_events
