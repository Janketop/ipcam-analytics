"""Поток обработки видеопотока конкретной камеры."""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Thread
from typing import Awaitable, Callable, Optional

import cv2

from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.core.logger import logger
from backend.models import Event, FaceSample

from backend.services.activity_detector import ActivityDetector
from backend.services.ai_detector import AIDetector
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
        self.detector = AIDetector(
            name,
            face_blur=face_blur,
            visualize=self.visualize,
            detect_person=detect_person,
            detect_car=detect_car,
            capture_entry_time=capture_entry_time,
        )
        self.activity_detector = ActivityDetector(idle_threshold=self.idle_alert_time)
        self.score_buffer: deque[float] = deque(maxlen=self.detector.score_smoothing)
        self.last_visual_jpeg: Optional[bytes] = None

        self.worker_started_at: Optional[datetime] = None
        self.last_frame_at: Optional[datetime] = None
        self.frame_durations: deque[float] = deque(maxlen=120)
        self._last_frame_monotonic: Optional[float] = None
        self.snapshot_candidates: deque[tuple[object, float]] = deque(
            maxlen=settings.snapshot_focus_buffer_size
        )

        self.broadcaster = broadcaster
        self.status_broadcaster = status_broadcaster
        self.main_loop = main_loop

        self.phone_score_threshold = self.detector.phone_score_threshold
        self.phone_active_until: Optional[datetime] = None

        self.status_interval = settings.ingest_status_interval
        self.status_stale_seconds = settings.ingest_status_stale_threshold
        self._last_status_sent_monotonic: Optional[float] = None

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
            self.activity_detector.idle_threshold = float(self.idle_alert_time)

        self.detector.update_flags(
            detect_person=self.detect_person,
            detect_car=self.detect_car,
            capture_entry_time=self.capture_entry_time,
        )

    def set_broadcaster(self, fn):
        self.broadcaster = fn

    def set_status_broadcaster(self, fn):
        self.status_broadcaster = fn

    def set_main_loop(self, loop):
        self.main_loop = loop

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

    def _push_snapshot_candidate(self, frame) -> None:
        if frame is None:
            return
        sharpness = self._measure_sharpness(frame)
        self.snapshot_candidates.append((frame.copy(), sharpness))

    def _select_best_snapshot_candidate(self):
        if not self.snapshot_candidates:
            return None
        best_frame, _ = max(self.snapshot_candidates, key=lambda item: item[1])
        return best_frame

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
        fps_skip = settings.ingest_fps_skip
        flush_timeout = settings.ingest_flush_timeout

        self.worker_started_at = datetime.now(timezone.utc)
        self._last_frame_monotonic = None
        self.frame_durations.clear()
        logger.info("[%s] Ingest-воркер запущен", self.name)
        self._broadcast_status(force=True)
        while not self.stop_flag:
            self._broadcast_status()
            cap = cv2.VideoCapture(self.url)
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
            frame_id = 0
            failed_reads = 0
            reconnect_needed = False

            while not self.stop_flag:
                self._broadcast_status()
                ok, frame = cap.read()
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
                frame_id += 1
                if frame_id % fps_skip != 0:
                    continue

                flush_start = time.time()
                grabbed_any = False
                while time.time() - flush_start < flush_timeout:
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

                self._push_snapshot_candidate(frame)
                (
                    phone_usage_raw,
                    conf_raw,
                    snapshot_img,
                    vis,
                    car_events,
                    people,
                ) = self.detector.process_frame(frame)
                best_snapshot_frame = self._select_best_snapshot_candidate()
                if best_snapshot_frame is not None:
                    snapshot_img = prepare_snapshot(
                        best_snapshot_frame,
                        self.detector.face_blur,
                        self.detector.face_cascade,
                    )

                frame_processed_monotonic = time.monotonic()
                if self._last_frame_monotonic is not None:
                    self.frame_durations.append(frame_processed_monotonic - self._last_frame_monotonic)
                self._last_frame_monotonic = frame_processed_monotonic
                self.last_frame_at = datetime.now(timezone.utc)

                if self.detect_person:
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
                activity_updates = self.activity_detector.update(people, now=now)
                events_to_store = []
                ev_type = None
                if self.detect_person and phone_usage:
                    self.phone_active_until = now + timedelta(seconds=5)
                    ev_type = "PHONE_USAGE"

                if self.visualize and vis is not None:
                    try:
                        ret, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ret:
                            self.last_visual_jpeg = buf.tobytes()
                    except Exception:
                        logger.exception("[%s] Не удалось сформировать визуализацию кадра", self.name)

                if ev_type:
                    events_to_store.append(
                        {
                            "ts": now,
                            "type": ev_type,
                            "confidence": float(conf),
                            "snapshot": snapshot_img,
                            "meta": {},
                            "kind": "phone",
                        }
                    )

                for activity in activity_updates:
                    if not activity.get("changed"):
                        continue

                    activity_id = activity.get("id")
                    meta = {
                        "person_id": str(activity_id) if activity_id is not None else None,
                        "pose_confidence": _normalize_meta_number(activity.get("confidence")),
                        "head_angle": _normalize_meta_number(activity.get("head_angle")),
                        "head_motion": _normalize_meta_number(activity.get("head_movement")),
                        "hands_motion": _normalize_meta_number(activity.get("hand_movement")),
                        "movement_score": _normalize_meta_number(activity.get("movement_score")),
                        "duration_idle_sec": _normalize_meta_number(activity.get("idle_seconds")),
                        "duration_away_sec": _normalize_meta_number(activity.get("away_seconds")),
                    }
                    meta = {key: value for key, value in meta.items() if value is not None}

                    events_to_store.append(
                        {
                            "ts": now,
                            "type": activity["state"],
                            "confidence": float(activity.get("confidence", 0.0)),
                            "snapshot": snapshot_img,
                            "meta": meta,
                            "kind": "activity",
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
                if events_to_store:
                    with self.session_factory() as session:
                        for payload in events_to_store:
                            snapshot_img = payload.get("snapshot")
                            ts = payload["ts"]
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
                                confidence=payload["confidence"],
                                snapshot_url=snap_url,
                                meta=meta,
                            )
                            session.add(event)
                            session.flush()

                            if (
                                payload["kind"] == "activity"
                                and snap_url
                                and not session.query(FaceSample)
                                .filter(FaceSample.event_id == event.id)
                                .first()
                            ):
                                candidate_raw = meta.get("person_id") or meta.get("personId")
                                candidate_key = None
                                if isinstance(candidate_raw, str) and candidate_raw.strip():
                                    candidate_key = candidate_raw.strip()

                                sample = FaceSample(
                                    event_id=event.id,
                                    camera_id=self.cam_id,
                                    snapshot_url=snap_url,
                                    status=FaceSample.STATUS_UNVERIFIED,
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
                                "confidence": payload["confidence"],
                                "snapshot_url": snap_url,
                                "meta": meta,
                            }
                        )
                        session.commit()
                    if self.snapshot_candidates:
                        self.snapshot_candidates.clear()
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
