"""Поток обработки видеопотока конкретной камеры."""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from importlib import import_module, util
from threading import Thread
from typing import Any, Awaitable, Callable, Optional, Sequence

import cv2
import numpy as np

from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.core.logger import logger
from backend.core.paths import DATASET_FACE_DETECTION_DIR
from backend.models import Event, FaceSample

from backend.services.activity_detector import ActivityDetector
from backend.services.ai_detector import AIDetector
from backend.services.employee_recognizer import EmployeeRecognizer
from backend.services.snapshots import prepare_snapshot, save_snapshot


def _normalize_meta_number(value: object) -> float | None:
    """Преобразует числовое значение к float и отбрасывает бесконечности/NaN."""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number) or math.isinf(number):
        return None

    return number


class IngestWorker(Thread):
    def __init__(
        self,
        session_factory: SessionFactory,
        cam_id: int,
        name: str,
        rtsp_url: str,
        face_blur: bool = False,
        broadcaster: Optional[Callable[[dict], Awaitable[None]]] = None,
        status_broadcaster: Optional[Callable[[dict], Awaitable[None]]] = None,
        main_loop: Optional[asyncio.AbstractEventLoop] = None,
        *,
        detect_person: bool = True,
        detect_car: bool = True,
        capture_entry_time: bool = True,
        idle_alert_time: int | float = settings.idle_alert_time,
        zones: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.session_factory = session_factory
        self.cam_id = cam_id
        self.name = name
        self.url = rtsp_url
        self.stop_flag = False

        self.visualize = settings.visualize
        self.detect_person = detect_person
        self.detect_car = detect_car
        self.capture_entry_time = capture_entry_time
        self.idle_alert_time = float(idle_alert_time)
        self._zone_polygons: list[list[tuple[float, float]]] = self._sanitize_zones(
            zones
        )

        self.detector = AIDetector(
            name,
            face_blur=face_blur,
            visualize=self.visualize,
            detect_person=detect_person,
            detect_car=detect_car,
            capture_entry_time=capture_entry_time,
            zones=self._zone_polygons,
        )
        self.enable_phone_detection = settings.enable_phone_detection
        self.enable_activity_detection = settings.enable_activity_detection
        self.employee_recognizer = self._init_employee_recognizer()
        self.detector.set_employee_recognizer(self.employee_recognizer)
        self.employee_presence_cooldown = float(settings.face_recognition_presence_cooldown)
        self._employee_presence_last: dict[int, float] = {}
        self._unknown_face_recent: dict[str, float] = {}
        self.activity_detector = (
            ActivityDetector(idle_threshold=self.idle_alert_time)
            if self.enable_activity_detection
            else None
        )
        self.score_buffer: deque[float] = deque(maxlen=self.detector.score_smoothing)
        self.last_visual_jpeg: Optional[bytes] = None

        self.worker_started_at: Optional[datetime] = None
        self.last_frame_at: Optional[datetime] = None
        self.frame_durations: deque[float] = deque(maxlen=120)
        self._last_frame_monotonic: Optional[float] = None
        self._next_allowed_process_monotonic: Optional[float] = None
        self._last_snapshot_capture_monotonic: Optional[float] = None
        self.snapshot_candidates: deque[dict[str, Any]] = deque(
            maxlen=settings.snapshot_focus_buffer_size
        )

        self.broadcaster = broadcaster
        self.status_broadcaster = status_broadcaster
        self.main_loop = main_loop

        self.phone_score_threshold = self.detector.phone_score_threshold
        self.phone_active_until: Optional[datetime] = None
        self.phone_active_since: Optional[datetime] = None
        self.phone_active: bool = False
        self.phone_max_confidence: float = 0.0
        self.phone_snapshot: Optional[np.ndarray] = None

        self.status_interval = settings.ingest_status_interval
        self.status_stale_seconds = settings.ingest_status_stale_threshold
        self._last_status_sent_monotonic: Optional[float] = None

        self._face_sample_model: Any = None
        self._face_sample_import_failed = False

        self.best_snapshot_by_employee: dict[int, dict[str, Any]] = {}

    def update_flags(
        self,
        *,
        detect_person: Optional[bool] = None,
        detect_car: Optional[bool] = None,
        capture_entry_time: Optional[bool] = None,
        idle_alert_time: Optional[int | float] = None,
    ) -> None:
        if detect_person is not None:
            self.detect_person = bool(detect_person)
            if not self.detect_person:
                self.score_buffer.clear()
        if detect_car is not None:
            self.detect_car = bool(detect_car)
        if capture_entry_time is not None:
            self.capture_entry_time = bool(capture_entry_time)

        if idle_alert_time is not None:
            self.idle_alert_time = float(idle_alert_time)
            if self.activity_detector is not None:
                self.activity_detector.idle_threshold = float(self.idle_alert_time)

        self.detector.update_flags(
            detect_person=self.detect_person,
            detect_car=self.detect_car,
            capture_entry_time=self.capture_entry_time,
        )

    def update_zones(
        self, zones: Optional[Sequence[dict[str, Any]]] = None
    ) -> None:
        polygons = self._sanitize_zones(zones)
        self._zone_polygons = polygons
        try:
            self.detector.update_zones(polygons)
        except Exception:
            logger.exception(
                "[%s] Не удалось обновить зоны детекции", self.name
            )
            return

        logger.info(
            "[%s] Обновлены зоны детекции (%d полигонов)",
            self.name,
            len(polygons),
        )

    def set_broadcaster(self, fn):
        self.broadcaster = fn

    def set_status_broadcaster(self, fn):
        self.status_broadcaster = fn

    def set_main_loop(self, loop):
        self.main_loop = loop

    def _init_employee_recognizer(self) -> Optional[EmployeeRecognizer]:
        try:
            recognizer = EmployeeRecognizer(
                self.session_factory,
                threshold=settings.face_recognition_threshold,
                encoding_model=settings.face_recognition_model,
            )
        except Exception:
            logger.exception(
                "[%s] Не удалось инициализировать распознаватель сотрудников",
                self.name,
            )
            return None

        if recognizer.has_samples:
            logger.info(
                "[%s] Подготовлен распознаватель сотрудников (%d эмбеддингов, %d сотрудников)",
                self.name,
                recognizer.sample_count,
                recognizer.employee_count,
            )
        else:
            logger.info(
                "[%s] Эталонные эмбеддинги сотрудников отсутствуют", self.name
            )
        return recognizer

    def _get_face_sample_model(self) -> Any:
        if self._face_sample_import_failed:
            return None

        if self._face_sample_model is not None:
            return self._face_sample_model

        spec = util.find_spec("backend.models.face_sample")
        if spec is None:
            self._face_sample_import_failed = True
            logger.error(
                "[%s] Модуль backend.models.face_sample не найден; образцы лиц сохраняться не будут",
                self.name,
            )
            return None

        module = import_module("backend.models.face_sample")
        face_sample_model = getattr(module, "FaceSample", None)
        if face_sample_model is None:
            self._face_sample_import_failed = True
            logger.error(
                "[%s] В модуле backend.models.face_sample отсутствует класс FaceSample",
                self.name,
            )
            return None

        self._face_sample_model = face_sample_model
        return face_sample_model

    def _prune_unknown_face_cache(self, now_monotonic: float) -> None:
        if not self._unknown_face_recent:
            return

        cooldown = max(float(self.employee_presence_cooldown), 0.0)
        if cooldown <= 0.0:
            if len(self._unknown_face_recent) > 512:
                self._unknown_face_recent.clear()
            return

        expiry_threshold = now_monotonic - cooldown * 4.0
        for key, seen_at in list(self._unknown_face_recent.items()):
            if seen_at < expiry_threshold:
                self._unknown_face_recent.pop(key, None)

    def _submit_async(self, coro: Awaitable[None], payload_kind: str) -> None:
        if not self.main_loop or self.main_loop.is_closed():
            return

        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "[%s] Не удалось отправить %s подписчикам: %s",
                self.name,
                payload_kind,
                exc,
            )
            return

        def _finalize(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except (asyncio.CancelledError, RuntimeError) as exc:
                logger.warning(
                    "[%s] Рассылка %s была прервана: %s",
                    self.name,
                    payload_kind,
                    exc,
                )
            except Exception:
                logger.exception(
                    "[%s] Ошибка при завершении рассылки %s",
                    self.name,
                    payload_kind,
                )

        future.add_done_callback(_finalize)

    def _measure_sharpness(self, frame) -> float:
        if frame is None:
            return 0.0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(variance)
        except Exception:
            logger.debug("[%s] Не удалось оценить резкость кадра", self.name, exc_info=True)
            return 0.0

    def _push_snapshot_candidate(
        self, frame, captured_monotonic: Optional[float] = None
    ) -> Optional[dict[str, Any]]:
        if frame is None:
            return None
        if captured_monotonic is None:
            captured_monotonic = time.monotonic()
        sharpness = self._measure_sharpness(frame)
        candidate: dict[str, Any] = {
            "frame": frame.copy(),
            "sharpness": float(sharpness),
            "captured_at": float(captured_monotonic),
            "metrics": {},
            "score": None,
        }
        self.snapshot_candidates.append(candidate)
        return candidate

    def _score_snapshot_candidate(self, candidate: dict[str, Any]) -> float:
        metrics = candidate.setdefault("metrics", {})
        normalized = metrics.get("normalized") or {}
        if normalized.get("sharpness") is None:
            sharpness_raw = candidate.get("sharpness")
            try:
                sharpness_value = max(float(sharpness_raw or 0.0), 0.0)
            except (TypeError, ValueError):
                sharpness_value = 0.0
            normalized["sharpness"] = 1.0 - math.exp(-sharpness_value / 150.0)
        weights = {
            "sharpness": 0.3,
            "face_area": 0.2,
            "face_centering": 0.2,
            "face_brightness": 0.1,
            "head_straightness": 0.1,
            "eye_focus": 0.1,
        }

        available_weights = {
            key: weight
            for key, weight in weights.items()
            if normalized.get(key) is not None
        }
        total_weight = sum(available_weights.values())
        score = 0.0
        if total_weight > 0:
            score = sum(
                normalized.get(key, 0.0) * weight for key, weight in available_weights.items()
            ) / total_weight
        elif normalized.get("sharpness") is not None:
            score = float(normalized.get("sharpness") or 0.0)

        metrics["weights"] = weights
        metrics["normalized"] = normalized
        metrics["contributions"] = {
            key: (normalized.get(key) or 0.0) * weight for key, weight in weights.items()
        }
        metrics["total_weight"] = total_weight
        candidate["score"] = float(score)
        metrics["score"] = float(score)
        return float(score)

    def _normalize_employee_id(self, value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_candidate_employee_id(self, candidate: dict[str, Any]) -> Optional[int]:
        metrics = candidate.get("metrics") or {}
        raw_metrics = metrics.get("raw") or {}
        employee_id_raw = raw_metrics.get("employee_id")
        return self._normalize_employee_id(employee_id_raw)

    def _update_best_snapshot_for_candidate(self, candidate: dict[str, Any]) -> None:
        employee_id = self._get_candidate_employee_id(candidate)
        if employee_id is None:
            return

        score_value = candidate.get("score")
        if score_value is None:
            score_value = self._score_snapshot_candidate(candidate)

        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0

        captured_at = candidate.get("captured_at")
        captured_monotonic = (
            float(captured_at) if isinstance(captured_at, (int, float)) else None
        )

        current_best = self.best_snapshot_by_employee.get(employee_id)
        current_score = -math.inf
        if current_best:
            try:
                current_score = float(current_best.get("score", -math.inf))
            except (TypeError, ValueError):
                current_score = -math.inf

        if score > current_score:
            self.best_snapshot_by_employee[employee_id] = {
                "candidate": candidate,
                "score": score,
                "captured_at": captured_monotonic,
            }

    def _recompute_best_snapshot_for_employee(self, employee_id: int) -> None:
        best_candidate = None
        best_score = -math.inf
        best_captured: Optional[float] = None

        for candidate in self.snapshot_candidates:
            cand_employee = self._get_candidate_employee_id(candidate)
            if cand_employee != employee_id:
                continue

            score_value = candidate.get("score")
            if score_value is None:
                score_value = self._score_snapshot_candidate(candidate)

            try:
                score = float(score_value)
            except (TypeError, ValueError):
                score = 0.0

            if score > best_score:
                best_score = score
                best_candidate = candidate
                captured_at = candidate.get("captured_at")
                best_captured = (
                    float(captured_at) if isinstance(captured_at, (int, float)) else None
                )

        if best_candidate is not None:
            self.best_snapshot_by_employee[employee_id] = {
                "candidate": best_candidate,
                "score": best_score,
                "captured_at": best_captured,
            }
        else:
            self.best_snapshot_by_employee.pop(employee_id, None)

    def _remove_candidate_from_best(self, candidate: dict[str, Any]) -> None:
        to_recompute: list[int] = []
        for employee_id, data in list(self.best_snapshot_by_employee.items()):
            if data.get("candidate") is candidate:
                to_recompute.append(employee_id)

        for employee_id in to_recompute:
            self.best_snapshot_by_employee.pop(employee_id, None)
            self._recompute_best_snapshot_for_employee(employee_id)

    def _handle_candidate_employee_change(
        self,
        candidate: dict[str, Any],
        previous_employee_id: Optional[int],
        new_employee_id: Optional[int],
    ) -> None:
        if (
            previous_employee_id is not None
            and previous_employee_id != new_employee_id
        ):
            entry = self.best_snapshot_by_employee.get(previous_employee_id)
            if entry and entry.get("candidate") is candidate:
                self.best_snapshot_by_employee.pop(previous_employee_id, None)
                self._recompute_best_snapshot_for_employee(previous_employee_id)

    def _get_best_snapshot_image_for_employee(self, employee_id: int) -> Optional[np.ndarray]:
        entry = self.best_snapshot_by_employee.get(employee_id)
        if not entry:
            return None

        candidate = entry.get("candidate")
        if not isinstance(candidate, dict):
            return None

        frame = candidate.get("frame")
        if frame is None:
            return None

        return prepare_snapshot(frame, self.detector.face_blur, self.detector.face_cascade)

    def _update_snapshot_candidate_metrics(
        self,
        candidate: Optional[dict[str, Any]],
        frame,
        people: Sequence[dict[str, Any]],
    ) -> None:
        if candidate is None or frame is None:
            return
        if frame.size == 0:
            return

        try:
            height, width = frame.shape[:2]
        except Exception:
            return

        frame_area = float(max(width * height, 1))
        metrics = candidate.setdefault("metrics", {})
        raw_metrics = metrics.setdefault("raw", {})
        previous_employee_id = self._normalize_employee_id(raw_metrics.get("employee_id"))
        normalized_metrics: dict[str, Optional[float]] = {}

        sharpness = float(candidate.get("sharpness", 0.0) or 0.0)
        normalized_metrics["sharpness"] = 1.0 - math.exp(-max(sharpness, 0.0) / 150.0)

        selected_person = None
        best_face_score = -math.inf
        selected_employee_id = None
        for person in people or []:
            if not isinstance(person, dict):
                continue
            face_bbox = person.get("face_bbox")
            if not face_bbox or len(face_bbox) < 4:
                continue
            x1, y1, x2, y2 = face_bbox[:4]
            area = max(0.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))
            if area <= 0.0:
                continue
            confidence = _normalize_meta_number(person.get("face_confidence")) or 0.0
            score = confidence * 0.6 + (area / frame_area)
            if score > best_face_score:
                best_face_score = score
                selected_person = person
                employee_data = person.get("employee") if isinstance(person, dict) else None
                employee_info = employee_data if isinstance(employee_data, dict) else None
                if isinstance(employee_info, dict):
                    employee_id_raw = employee_info.get("id")
                    try:
                        selected_employee_id = (
                            int(employee_id_raw) if employee_id_raw is not None else None
                        )
                    except (TypeError, ValueError):
                        selected_employee_id = None
                else:
                    selected_employee_id = None

        face_area_ratio = None
        face_center_score = None
        face_brightness = None
        head_tilt = None
        eye_focus = None
        if selected_person is not None:
            face_bbox = selected_person.get("face_bbox")
            if isinstance(face_bbox, Sequence) and len(face_bbox) >= 4:
                try:
                    raw_x1, raw_y1, raw_x2, raw_y2 = map(float, face_bbox[:4])
                except Exception:
                    raw_x1 = raw_y1 = 0.0
                    raw_x2 = float(width)
                    raw_y2 = float(height)
                x1 = int(max(0.0, min(raw_x1, float(width - 1))))
                y1 = int(max(0.0, min(raw_y1, float(height - 1))))
                x2 = int(max(float(x1 + 1), min(raw_x2, float(width))))
                y2 = int(max(float(y1 + 1), min(raw_y2, float(height))))

                face_w = float(max(x2 - x1, 1))
                face_h = float(max(y2 - y1, 1))
                face_area_ratio = (face_w * face_h) / frame_area
                face_area_ratio = max(0.0, min(face_area_ratio, 1.0))
                normalized_metrics["face_area"] = min(face_area_ratio / 0.05, 1.0)

                frame_center_x = width * 0.5
                frame_center_y = height * 0.5
                face_center_x = x1 + face_w * 0.5
                face_center_y = y1 + face_h * 0.5
                dx = abs(face_center_x - frame_center_x) / max(frame_center_x, 1.0)
                dy = abs(face_center_y - frame_center_y) / max(frame_center_y, 1.0)
                center_distance = math.sqrt(dx * dx + dy * dy)
                face_center_score = max(0.0, 1.0 - min(center_distance, 1.0))
                normalized_metrics["face_centering"] = face_center_score

                try:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    roi = hsv[y1:y2, x1:x2, 2]
                    if roi.size:
                        face_brightness = float(np.mean(roi) / 255.0)
                        target_brightness = 0.6
                        brightness_diff = abs(face_brightness - target_brightness)
                        normalized_metrics["face_brightness"] = max(
                            0.0, 1.0 - brightness_diff / target_brightness
                        )
                except Exception:
                    face_brightness = None

                head_tilt = _normalize_meta_number(selected_person.get("head_tilt"))
                if head_tilt is not None:
                    normalized_metrics["head_straightness"] = max(
                        0.0, 1.0 - min(abs(head_tilt) / 0.35, 1.0)
                    )

                eyes = selected_person.get("eyes") if isinstance(selected_person, dict) else None
                eye_confidences = []
                if isinstance(eyes, dict):
                    conf = eyes.get("confidences")
                    if isinstance(conf, dict):
                        for key in ("left", "right"):
                            val = conf.get(key)
                            norm = _normalize_meta_number(val)
                            if norm is not None:
                                eye_confidences.append(norm)
                if eye_confidences:
                    eye_focus = float(np.mean(eye_confidences))
                    normalized_metrics["eye_focus"] = max(0.0, min(eye_focus, 1.0))
                else:
                    left_eye = eyes.get("left") if isinstance(eyes, dict) else None
                    right_eye = eyes.get("right") if isinstance(eyes, dict) else None
                    if (
                        isinstance(left_eye, Sequence)
                        and isinstance(right_eye, Sequence)
                        and len(left_eye) >= 2
                        and len(right_eye) >= 2
                    ):
                        left_y = float(left_eye[1])
                        right_y = float(right_eye[1])
                        vertical_diff = abs(left_y - right_y) / face_h
                        eye_focus = max(0.0, 1.0 - min(vertical_diff / 0.15, 1.0))
                        normalized_metrics["eye_focus"] = eye_focus

        metrics["normalized"] = normalized_metrics
        raw_metrics.update(
            {
                "sharpness": sharpness,
                "frame_size": (width, height),
                "face_area_ratio": face_area_ratio,
                "face_center_score": face_center_score,
                "face_brightness": face_brightness,
                "head_tilt": head_tilt,
                "eye_focus": eye_focus,
                "selected_person": selected_person,
                "selected_person_id": (
                    selected_person.get("id") if isinstance(selected_person, dict) else None
                ),
                "employee_id": selected_employee_id,
            }
        )

        self._score_snapshot_candidate(candidate)
        current_employee_id = self._normalize_employee_id(selected_employee_id)
        self._handle_candidate_employee_change(
            candidate, previous_employee_id, current_employee_id
        )
        self._update_best_snapshot_for_candidate(candidate)

    def _select_best_snapshot_candidate(
        self, since_monotonic: Optional[float] = None
    ) -> tuple[Optional[dict[str, Any]], Optional[float]]:
        if not self.snapshot_candidates:
            return None, None

        best_candidate: Optional[dict[str, Any]] = None
        best_score = -math.inf

        for candidate in self.snapshot_candidates:
            captured_at = candidate.get("captured_at")
            if since_monotonic is not None and isinstance(captured_at, (int, float)):
                if float(captured_at) <= since_monotonic:
                    continue

            score = candidate.get("score")
            if score is None:
                score = self._score_snapshot_candidate(candidate)

            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = 0.0

            if best_candidate is None or numeric_score > best_score:
                best_candidate = candidate
                best_score = numeric_score

        if not best_candidate:
            return None, None

        metrics = best_candidate.get("metrics") or {}
        normalized = metrics.get("normalized") or {}
        log_parts = []
        for key in ("sharpness", "face_area", "face_centering", "face_brightness", "head_straightness", "eye_focus"):
            value = normalized.get(key)
            if value is None:
                log_parts.append(f"{key}=n/a")
            else:
                log_parts.append(f"{key}={float(value):.3f}")
        log_message = ", ".join(log_parts)
        logger.debug(
            "[%s] Выбран кандидат снапшота: score=%.3f (%s)",
            self.name,
            float(best_candidate.get("score") or best_score),
            log_message,
        )

        captured_at = best_candidate.get("captured_at")
        captured_monotonic = float(captured_at) if isinstance(captured_at, (int, float)) else None
        return best_candidate, captured_monotonic

    def _prune_snapshot_candidates(self, before_or_equal: Optional[float]) -> None:
        if before_or_equal is None:
            return

        while self.snapshot_candidates:
            first = self.snapshot_candidates[0]
            captured_at = first.get("captured_at")
            if not isinstance(captured_at, (int, float)):
                break
            if float(captured_at) <= before_or_equal:
                removed = self.snapshot_candidates.popleft()
                if isinstance(removed, dict):
                    self._remove_candidate_from_best(removed)
            else:
                break

    def _determine_status(self, now: datetime, runtime: dict) -> str:
        if self.stop_flag:
            return "stopping"

        last_frame_iso = runtime.get("last_frame_at")
        last_frame_at: Optional[datetime] = None
        if isinstance(last_frame_iso, str):
            try:
                last_frame_at = datetime.fromisoformat(last_frame_iso)
            except ValueError:
                last_frame_at = None

        if last_frame_at is None:
            return "starting"

        if last_frame_at.tzinfo is None:
            last_frame_at = last_frame_at.replace(tzinfo=timezone.utc)

        delta = now - last_frame_at.astimezone(timezone.utc)
        if delta.total_seconds() > self.status_stale_seconds:
            return "no_signal"

        return "online"

    def _broadcast_status(
        self,
        *,
        force: bool = False,
        override_status: Optional[str] = None,
    ) -> None:
        if not self.status_broadcaster:
            return

        now_monotonic = time.monotonic()
        if (
            not force
            and self.status_interval > 0
            and self._last_status_sent_monotonic is not None
            and now_monotonic - self._last_status_sent_monotonic < self.status_interval
        ):
            return

        runtime = self.runtime_status()
        now = datetime.now(timezone.utc)
        status = override_status or self._determine_status(now, runtime)
        payload = {
            "cameraId": self.cam_id,
            "camera": self.name,
            "status": status,
            "fps": runtime.get("fps"),
            "lastFrameTs": runtime.get("last_frame_at"),
            "uptimeSec": runtime.get("uptime_seconds"),
            "ts": now.isoformat(),
        }

        if status == "offline":
            payload.update({"fps": None, "lastFrameTs": None, "uptimeSec": None})

        try:
            coro = self.status_broadcaster(payload)
        except Exception:
            logger.exception(
                "[%s] Ошибка при подготовке данных статуса для рассылки",
                self.name,
            )
            return

        self._last_status_sent_monotonic = now_monotonic
        self._submit_async(coro, "статуса")

    def runtime_status(self) -> dict:
        info = self.detector.runtime_status()
        now = datetime.now(timezone.utc)
        uptime_seconds: Optional[float] = None
        if self.worker_started_at is not None:
            uptime_seconds = max((now - self.worker_started_at).total_seconds(), 0.0)

        fps: Optional[float] = None
        if self.frame_durations:
            avg_duration = sum(self.frame_durations) / len(self.frame_durations)
            if avg_duration > 0:
                fps = 1.0 / avg_duration

        info.update(
            {
                "visualize_enabled": bool(self.visualize),
                "started_at": self.worker_started_at.isoformat() if self.worker_started_at else None,
                "last_frame_at": self.last_frame_at.isoformat() if self.last_frame_at else None,
                "uptime_seconds": uptime_seconds,
                "fps": fps,
            }
        )
        return info

    def run(self) -> None:  # noqa: C901
        reconnect_delay = settings.rtsp_reconnect_delay
        max_failed_reads = settings.rtsp_max_failed_reads
        target_fps = float(settings.ingest_target_fps)
        target_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        flush_timeout = settings.ingest_flush_timeout

        self.worker_started_at = datetime.now(timezone.utc)
        self._last_frame_monotonic = None
        self._next_allowed_process_monotonic = None
        self._last_snapshot_capture_monotonic = None
        self.frame_durations.clear()
        logger.info("[%s] Ingest-воркер запущен", self.name)
        self._broadcast_status(force=True)
        while not self.stop_flag:
            self._broadcast_status()
            cap = cv2.VideoCapture(self.url)
            self._next_allowed_process_monotonic = None
            self._last_snapshot_capture_monotonic = None
            self.snapshot_candidates.clear()
            self.best_snapshot_by_employee.clear()
            try:
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as exc:
                logger.warning(
                    "[%s] Не удалось установить размер буфера видеопотока: %s",
                    self.name,
                    exc,
                    exc_info=True,
                )
            if not cap.isOpened():
                logger.warning(
                    "[%s] Не удалось открыть поток, повторное подключение через %.1f с",
                    self.name,
                    reconnect_delay,
                )
                self._broadcast_status(force=True)
                cap.release()
                time.sleep(reconnect_delay)
                continue

            logger.info("[%s] Ingest-воркер успешно подключился к потоку", self.name)
            failed_reads = 0
            reconnect_needed = False

            while not self.stop_flag:
                self._broadcast_status()
                ok, frame = cap.read()
                capture_monotonic = time.monotonic()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        reconnect_needed = True
                        logger.warning(
                            "[%s] Потеряно соединение, повторное подключение через %.1f с",
                            self.name,
                            reconnect_delay,
                        )
                        self._broadcast_status(force=True)
                        break
                    time.sleep(0.2)
                    continue

                failed_reads = 0
                current_candidate = self._push_snapshot_candidate(frame, capture_monotonic)

                if (
                    target_interval > 0
                    and self._next_allowed_process_monotonic is not None
                    and capture_monotonic < self._next_allowed_process_monotonic
                ):
                    continue

                grabbed_any = False
                if flush_timeout > 0:
                    flush_deadline = time.monotonic() + flush_timeout
                    while time.monotonic() < flush_deadline:
                        ok_grab = cap.grab()
                        if not ok_grab:
                            failed_reads += 1
                            if failed_reads >= max_failed_reads:
                                reconnect_needed = True
                                logger.warning(
                                    "[%s] Потеряно соединение при чтении буфера, повторное подключение через %.1f с",
                                    self.name,
                                    reconnect_delay,
                                )
                                self._broadcast_status(force=True)
                                break
                            time.sleep(0.2)
                            break
                        grabbed_any = True
                    if reconnect_needed:
                        self._broadcast_status(force=True)
                        break

                if grabbed_any:
                    ok_retrieve, latest_frame = cap.retrieve()
                    if not ok_retrieve:
                        failed_reads += 1
                        if failed_reads >= max_failed_reads:
                            reconnect_needed = True
                            logger.warning(
                                "[%s] Потеряно соединение при получении кадра, повторное подключение через %.1f с",
                                self.name,
                                reconnect_delay,
                            )
                            self._broadcast_status(force=True)
                            break
                        time.sleep(0.2)
                        continue
                    frame = latest_frame
                    capture_monotonic = time.monotonic()
                    current_candidate = self._push_snapshot_candidate(frame, capture_monotonic)

                process_capture_monotonic = capture_monotonic
                (
                    phone_usage_raw,
                    conf_raw,
                    snapshot_img,
                    vis,
                    car_events,
                    people,
                ) = self.detector.process_frame(frame)
                self._update_snapshot_candidate_metrics(current_candidate, frame, people)

                best_candidate, selected_capture_monotonic = self._select_best_snapshot_candidate(
                    since_monotonic=self._last_snapshot_capture_monotonic
                )
                if selected_capture_monotonic is not None:
                    self._last_snapshot_capture_monotonic = selected_capture_monotonic
                else:
                    self._last_snapshot_capture_monotonic = process_capture_monotonic
                self._prune_snapshot_candidates(self._last_snapshot_capture_monotonic)
                if best_candidate and best_candidate.get("frame") is not None:
                    snapshot_img = prepare_snapshot(
                        best_candidate["frame"],
                        self.detector.face_blur,
                        self.detector.face_cascade,
                    )

                frame_processed_monotonic = time.monotonic()
                if target_interval > 0:
                    self._next_allowed_process_monotonic = (
                        process_capture_monotonic + target_interval
                    )
                if self._last_frame_monotonic is not None:
                    self.frame_durations.append(frame_processed_monotonic - self._last_frame_monotonic)
                self._last_frame_monotonic = frame_processed_monotonic
                self.last_frame_at = datetime.now(timezone.utc)

                if self.detect_person and self.enable_phone_detection:
                    self.score_buffer.append(conf_raw)
                    smoothed_conf = max(self.score_buffer) if self.score_buffer else conf_raw
                    phone_usage = phone_usage_raw or smoothed_conf >= self.phone_score_threshold
                    conf = smoothed_conf if phone_usage else conf_raw
                else:
                    self.score_buffer.clear()
                    smoothed_conf = conf_raw
                    phone_usage = False
                    conf = conf_raw

                now = datetime.now(timezone.utc)
                recognized_employees: dict[int, dict[str, object]] = {}
                employee_by_person: dict[str, dict[str, object]] = {}
                unknown_faces_to_store: list[dict[str, Any]] = []
                for person in people:
                    employee_info = person.get("employee") if isinstance(person, dict) else None
                    person_id_raw = person.get("id") if isinstance(person, dict) else None
                    if employee_info is None or not isinstance(employee_info, dict):
                        continue

                    emp_id_raw = employee_info.get("id")
                    try:
                        emp_id = int(emp_id_raw) if emp_id_raw is not None else None
                    except (TypeError, ValueError):
                        emp_id = None

                    distance_val = _normalize_meta_number(employee_info.get("distance"))
                    if distance_val is None:
                        distance_val = 0.0
                    employee_name = employee_info.get("name")

                    if emp_id is not None:
                        recognized_employees[emp_id] = {
                            "id": emp_id,
                            "name": employee_name,
                            "distance": distance_val,
                        }

                    if person_id_raw is not None:
                        person_key = str(person_id_raw)
                        employee_by_person[person_key] = {
                            "id": emp_id,
                            "name": employee_name,
                            "distance": distance_val,
                        }

                now_monotonic = time.monotonic()
                self._prune_unknown_face_cache(now_monotonic)
                for person in people:
                    if not isinstance(person, dict):
                        continue

                    unknown_face = person.get("unknown_face")
                    if not isinstance(unknown_face, dict):
                        continue

                    snapshot_img = unknown_face.get("snapshot")
                    if snapshot_img is None:
                        continue

                    face_key = unknown_face.get("hash")
                    if not face_key:
                        person_identifier = person.get("id")
                        if person_identifier is not None:
                            face_key = f"person:{person_identifier}"
                    if face_key:
                        last_seen = self._unknown_face_recent.get(face_key)
                        if (
                            last_seen is not None
                            and self.employee_presence_cooldown > 0.0
                            and now_monotonic - last_seen < self.employee_presence_cooldown
                        ):
                            continue
                        self._unknown_face_recent[face_key] = now_monotonic

                    candidate_key = str(person.get("id")) if person.get("id") is not None else None
                    unknown_faces_to_store.append(
                        {
                            "snapshot": snapshot_img,
                            "candidate_key": candidate_key,
                            "hash": unknown_face.get("hash"),
                            "embedding": unknown_face.get("embedding"),
                            "captured_at": now,
                        }
                    )

                if self.activity_detector is not None:
                    activity_updates = self.activity_detector.update(people, now=now)
                else:
                    activity_updates = []
                events_to_store = []

                if self.detect_person and self.enable_phone_detection:
                    phone_detected = False
                    if phone_usage:
                        self.phone_active_until = now + timedelta(seconds=5)
                    if (
                        self.phone_active_until is not None
                        and now <= self.phone_active_until
                    ):
                        phone_detected = True
                    else:
                        self.phone_active_until = None

                    if phone_detected:
                        if not self.phone_active:
                            self.phone_active = True
                            self.phone_active_since = now
                            self.phone_max_confidence = float(conf)
                            self.phone_snapshot = snapshot_img
                        else:
                            self.phone_max_confidence = max(
                                self.phone_max_confidence, float(conf)
                            )
                            if self.phone_snapshot is None and snapshot_img is not None:
                                self.phone_snapshot = snapshot_img
                    else:
                        if self.phone_active and self.phone_active_since is not None:
                            duration = (now - self.phone_active_since).total_seconds()
                            duration = max(duration, 0.0)
                            phone_meta = {
                                "duration_sec": duration,
                            }
                            events_to_store.append(
                                {
                                    "ts": self.phone_active_since,
                                    "end_ts": now,
                                    "type": "PHONE_USAGE",
                                    "confidence": float(
                                        self.phone_max_confidence or float(conf)
                                    ),
                                    "snapshot":
                                        self.phone_snapshot
                                        if self.phone_snapshot is not None
                                        else snapshot_img,
                                    "meta": phone_meta,
                                    "kind": "phone",
                                }
                            )
                        self.phone_active = False
                        self.phone_active_since = None
                        self.phone_active_until = None
                        self.phone_max_confidence = 0.0
                        self.phone_snapshot = None
                else:
                    self.phone_active = False
                    self.phone_active_since = None
                    self.phone_active_until = None
                    self.phone_max_confidence = 0.0
                    self.phone_snapshot = None

                display_frame = vis if (self.visualize and vis is not None) else frame
                if display_frame is not None:
                    try:
                        ret, buf = cv2.imencode(
                            ".jpg",
                            display_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                        )
                        if ret:
                            self.last_visual_jpeg = buf.tobytes()
                    except Exception:
                        logger.exception("[%s] Не удалось подготовить кадр для живого потока", self.name)

                for activity in activity_updates:
                    if not activity.get("changed"):
                        continue

                    activity_id = activity.get("id")
                    prev_state = activity.get("previous_state")
                    prev_state_start = activity.get("previous_state_started_at")
                    prev_duration = _normalize_meta_number(activity.get("duration_sec"))
                    new_state = activity.get("state")
                    state_started_at = activity.get("state_started_at")

                    meta = {
                        "person_id": str(activity_id) if activity_id is not None else None,
                        "pose_confidence": _normalize_meta_number(activity.get("confidence")),
                        "head_angle": _normalize_meta_number(activity.get("head_angle")),
                        "head_motion": _normalize_meta_number(activity.get("head_movement")),
                        "hands_motion": _normalize_meta_number(activity.get("hand_movement")),
                        "movement_score": _normalize_meta_number(activity.get("movement_score")),
                        "duration_idle_sec": _normalize_meta_number(activity.get("idle_seconds")),
                        "duration_away_sec": _normalize_meta_number(activity.get("away_seconds")),
                        "state": prev_state or new_state,
                        "next_state": new_state,
                    }
                    if isinstance(prev_state_start, datetime):
                        meta["state_started_at"] = prev_state_start.isoformat()
                    if isinstance(state_started_at, datetime):
                        meta["next_state_started_at"] = state_started_at.isoformat()
                    if prev_duration is not None:
                        meta["duration_sec"] = prev_duration
                        meta["state_finished_at"] = now.isoformat()
                    person_key = str(activity_id) if activity_id is not None else None
                    employee_meta = employee_by_person.get(person_key) if person_key is not None else None
                    if employee_meta:
                        emp_id = employee_meta.get("id")
                        if emp_id is not None:
                            meta["employeeId"] = int(emp_id)
                        if employee_meta.get("name"):
                            meta["employeeName"] = employee_meta.get("name")
                        distance_val = _normalize_meta_number(employee_meta.get("distance"))
                        if distance_val is not None:
                            meta["faceDistance"] = distance_val
                    meta = {key: value for key, value in meta.items() if value is not None}

                    prev_state_name = prev_state or new_state
                    event_start = prev_state_start if isinstance(prev_state_start, datetime) else None
                    should_store = prev_state_name in {"NOT_WORKING", "AWAY"} and (
                        event_start is not None and prev_duration is not None
                    )
                    if should_store:
                        events_to_store.append(
                            {
                                "ts": event_start,
                                "end_ts": now,
                                "type": prev_state_name,
                                "confidence": float(activity.get("confidence", 0.0)),
                                "snapshot": snapshot_img,
                                "meta": meta,
                                "kind": "activity",
                            }
                        )

                if recognized_employees:
                    current_monotonic = time.monotonic()
                    for emp_id, info in recognized_employees.items():
                        last_seen = self._employee_presence_last.get(emp_id)
                        if (
                            last_seen is not None
                            and self.employee_presence_cooldown > 0.0
                            and current_monotonic - last_seen < self.employee_presence_cooldown
                        ):
                            continue

                        self._employee_presence_last[emp_id] = current_monotonic
                        distance_val = _normalize_meta_number(info.get("distance")) or 0.0
                        presence_meta = {
                            "employeeId": emp_id,
                            "employeeName": info.get("name"),
                            "distance": distance_val,
                        }
                        presence_meta = {
                            key: value for key, value in presence_meta.items() if value is not None
                        }
                        employee_snapshot_img = self._get_best_snapshot_image_for_employee(emp_id)
                        event_snapshot = (
                            employee_snapshot_img if employee_snapshot_img is not None else snapshot_img
                        )
                        events_to_store.append(
                            {
                                "ts": now,
                                "type": "EMPLOYEE_PRESENT",
                                "confidence": max(0.0, 1.0 - float(distance_val)),
                                "snapshot": event_snapshot,
                                "meta": presence_meta,
                                "kind": "employee",
                            }
                        )

                if self.detect_car:
                    for car_ev in car_events:
                        car_now = datetime.now(timezone.utc)
                        meta = {
                            "plate": car_ev.get("plate") or "НЕ РАСПОЗНАНО",
                        }
                        if self.capture_entry_time:
                            meta["entry_ts"] = car_now.isoformat()
                        events_to_store.append(
                            {
                                "ts": car_now,
                                "type": "CAR_ENTRY",
                                "confidence": float(car_ev.get("confidence", 0.0)),
                                "snapshot": car_ev.get("snapshot"),
                                "meta": meta,
                                "kind": "car",
                            }
                        )

                persisted_events = []
                stored_face_samples: list[dict[str, Any]] = []
                if events_to_store or unknown_faces_to_store:
                    with self.session_factory() as session:
                        face_sample_model = self._get_face_sample_model()
                        for payload in events_to_store:
                            snapshot_img = payload.get("snapshot")
                            ts = payload["ts"]
                            end_ts = payload.get("end_ts")
                            meta = dict(payload.get("meta") or {})
                            snap_url = save_snapshot(
                                snapshot_img,
                                ts,
                                self.name,
                                event_type=payload["kind"],
                            )
                            event = Event(
                                camera_id=self.cam_id,
                                type=payload["type"],
                                start_ts=ts,
                                end_ts=end_ts,
                                confidence=payload["confidence"],
                                snapshot_url=snap_url,
                                meta=meta,
                            )
                            session.add(event)
                            session.flush()

                            if (
                                payload["kind"] == "activity"
                                and snap_url
                                and face_sample_model is not None
                                and not session.query(face_sample_model)
                                .filter(face_sample_model.event_id == event.id)
                                .first()
                            ):
                                    candidate_raw = meta.get("person_id") or meta.get("personId")
                                    candidate_key = None
                                    if isinstance(candidate_raw, str) and candidate_raw.strip():
                                        candidate_key = candidate_raw.strip()

                                    sample = face_sample_model(
                                        event_id=event.id,
                                        camera_id=self.cam_id,
                                        snapshot_url=snap_url,
                                        status=face_sample_model.STATUS_UNVERIFIED,
                                        candidate_key=candidate_key,
                                        captured_at=ts,
                                    )
                                    session.add(sample)
                            persisted_events.append(
                                {
                                    "id": event.id,
                                    "camera": self.name,
                                    "type": payload["type"],
                                    "start_ts": event.start_ts,
                                    "end_ts": event.end_ts,
                                    "confidence": payload["confidence"],
                                    "snapshot_url": snap_url,
                                    "meta": meta,
                                }
                            )

                        if face_sample_model is not None:
                            for sample_payload in unknown_faces_to_store:
                                ts = sample_payload["captured_at"]
                                snapshot_img = sample_payload["snapshot"]
                                snap_url = save_snapshot(
                                    snapshot_img,
                                    ts,
                                    self.name,
                                    event_type="face_unknown",
                                    dataset_dir=DATASET_FACE_DETECTION_DIR,
                                )
                                if not snap_url:
                                    continue

                                sample = face_sample_model(
                                    event_id=None,
                                    camera_id=self.cam_id,
                                    snapshot_url=snap_url,
                                    status=face_sample_model.STATUS_UNVERIFIED,
                                    candidate_key=sample_payload.get("candidate_key"),
                                    captured_at=ts,
                                )

                                embedding_result = sample_payload.get("embedding")
                                if embedding_result is not None:
                                    try:
                                        sample.set_embedding(
                                            embedding_result.as_bytes(),
                                            dim=embedding_result.dimension,
                                            model=embedding_result.model,
                                        )
                                    except Exception:
                                        logger.exception(
                                            "[%s] Не удалось сохранить эмбеддинг неизвестного лица",
                                            self.name,
                                        )
                                session.add(sample)
                                session.flush()
                                stored_face_samples.append(
                                    {
                                        "snapshot_url": snap_url,
                                        "hash": sample_payload.get("hash"),
                                    }
                                )
                        session.commit()
                    if (events_to_store or unknown_faces_to_store) and self.snapshot_candidates:
                        self.snapshot_candidates.clear()
                        self.best_snapshot_by_employee.clear()
                for stored in persisted_events:
                    logger.info(
                        "[%s] Зафиксировано событие %s (уверенность %.2f, данные %s)",
                        self.name,
                        stored["type"],
                        stored["confidence"],
                        stored["meta"],
                    )
                    if self.broadcaster:
                        try:
                            coro = self.broadcaster(
                                {
                                    "id": stored["id"],
                                    "camera": stored["camera"],
                                    "type": stored["type"],
                                    "start_ts": stored["start_ts"].isoformat(),
                                    "end_ts": stored["end_ts"].isoformat()
                                    if stored.get("end_ts")
                                    else None,
                                    "confidence": stored["confidence"],
                                    "snapshot_url": stored["snapshot_url"],
                                    "meta": stored["meta"],
                                }
                            )
                        except Exception:
                            logger.exception(
                                "[%s] Ошибка при подготовке события для рассылки",
                                self.name,
                            )
                        else:
                            self._submit_async(coro, "события")

                if stored_face_samples:
                    for sample_info in stored_face_samples:
                        logger.info(
                            "[%s] Сохранён образец неизвестного лица (hash=%s, url=%s)",
                            self.name,
                            sample_info.get("hash"),
                            sample_info.get("snapshot_url"),
                        )

            cap.release()
            if self.stop_flag:
                break

            if reconnect_needed:
                logger.info(
                    "[%s] Переподключение к потоку после паузы %.1f с",
                    self.name,
                    reconnect_delay,
                )
                time.sleep(reconnect_delay)

        logger.info("[%s] Ingest-воркер остановлен", self.name)
        self._broadcast_status(force=True, override_status="offline")

    def get_visual_frame_jpeg(self):
        return self.last_visual_jpeg

    def _sanitize_zones(
        self, zones: Optional[Sequence[dict[str, Any]]] = None
    ) -> list[list[tuple[float, float]]]:
        if not zones:
            return []

        normalized: list[list[tuple[float, float]]] = []
        for zone in zones:
            points: Optional[Sequence[Any]] = None
            if isinstance(zone, dict):
                raw_points = zone.get("points")
                if isinstance(raw_points, Sequence):
                    points = raw_points
            elif isinstance(zone, Sequence) and not isinstance(zone, (str, bytes)):
                points = zone
            if not points:
                continue

            normalized_points: list[tuple[float, float]] = []
            for point in points:
                if isinstance(point, dict):
                    x = point.get("x")
                    y = point.get("y")
                elif isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
                    x, y = point[0], point[1]
                else:
                    continue

                try:
                    x_val = float(x)
                    y_val = float(y)
                except (TypeError, ValueError):
                    continue

                x_val = max(0.0, min(1.0, x_val))
                y_val = max(0.0, min(1.0, y_val))
                normalized_points.append((x_val, y_val))

            if len(normalized_points) >= 3:
                normalized.append(normalized_points)

        return normalized
