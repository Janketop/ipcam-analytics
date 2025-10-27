"""Поток обработки видеопотока конкретной камеры."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Thread
from typing import Awaitable, Callable, Optional

import cv2

from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.core.logger import logger
from backend.models import Event

from backend.services.ai_detector import AIDetector
from backend.services.snapshots import save_snapshot


class IngestWorker(Thread):
    def __init__(
        self,
        session_factory: SessionFactory,
        cam_id: int,
        name: str,
        rtsp_url: str,
        face_blur: bool = True,
        broadcaster: Optional[Callable[[dict], Awaitable[None]]] = None,
        main_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.session_factory = session_factory
        self.cam_id = cam_id
        self.name = name
        self.url = rtsp_url
        self.stop_flag = False

        self.visualize = settings.visualize
        self.detector = AIDetector(name, face_blur=face_blur, visualize=self.visualize)
        self.score_buffer: deque[float] = deque(maxlen=self.detector.score_smoothing)
        self.last_visual_jpeg: Optional[bytes] = None

        self.broadcaster = broadcaster
        self.main_loop = main_loop

        self.phone_score_threshold = self.detector.phone_score_threshold
        self.phone_active_until: Optional[datetime] = None

    def set_broadcaster(self, fn):
        self.broadcaster = fn

    def set_main_loop(self, loop):
        self.main_loop = loop

    def runtime_status(self) -> dict:
        info = self.detector.runtime_status()
        info.update({
            "visualize_enabled": bool(self.visualize),
        })
        return info

    def run(self) -> None:  # noqa: C901
        reconnect_delay = settings.rtsp_reconnect_delay
        max_failed_reads = settings.rtsp_max_failed_reads
        fps_skip = settings.ingest_fps_skip
        flush_timeout = settings.ingest_flush_timeout

        logger.info("[%s] Ingest-воркер запущен", self.name)
        while not self.stop_flag:
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
                cap.release()
                time.sleep(reconnect_delay)
                continue

            logger.info("[%s] Ingest-воркер успешно подключился к потоку", self.name)
            frame_id = 0
            failed_reads = 0
            reconnect_needed = False

            while not self.stop_flag:
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
                            break
                        time.sleep(0.2)
                        break
                    grabbed_any = True
                if reconnect_needed:
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
                            break
                        time.sleep(0.2)
                        continue
                    frame = latest_frame

                phone_usage_raw, conf_raw, snapshot_img, vis, car_events = self.detector.process_frame(frame)

                self.score_buffer.append(conf_raw)
                smoothed_conf = max(self.score_buffer) if self.score_buffer else conf_raw
                phone_usage = phone_usage_raw or smoothed_conf >= self.phone_score_threshold
                conf = smoothed_conf if phone_usage else conf_raw

                now = datetime.now(timezone.utc)
                events_to_store = []
                ev_type = None
                if phone_usage:
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

                for car_ev in car_events:
                    car_now = datetime.now(timezone.utc)
                    meta = {
                        "plate": car_ev.get("plate") or "НЕ РАСПОЗНАНО",
                        "entry_ts": car_now.isoformat(),
                    }
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
                            persisted_events.append(
                                {
                                    "camera": self.name,
                                    "type": payload["type"],
                                    "ts": ts,
                                    "confidence": payload["confidence"],
                                    "snapshot_url": snap_url,
                                    "meta": meta,
                                }
                            )
                        session.commit()
                else:
                    persisted_events = []
                for stored in persisted_events:
                    logger.info(
                        "[%s] Зафиксировано событие %s (уверенность %.2f, данные %s)",
                        self.name,
                        stored["type"],
                        stored["confidence"],
                        stored["meta"],
                    )
                    if self.broadcaster and self.main_loop and not self.main_loop.is_closed():
                        try:
                            future = asyncio.run_coroutine_threadsafe(
                                self.broadcaster(
                                    {
                                        "camera": stored["camera"],
                                        "type": stored["type"],
                                        "ts": stored["ts"].isoformat(),
                                        "confidence": stored["confidence"],
                                        "snapshot_url": stored["snapshot_url"],
                                        "meta": stored["meta"],
                                    }
                                ),
                                self.main_loop,
                            )

                            def _finalize(fut):
                                try:
                                    fut.result()
                                except (asyncio.CancelledError, RuntimeError) as exc:
                                    logger.warning(
                                        "[%s] Рассылка события была прервана: %s",
                                        self.name,
                                        exc,
                                    )
                                except Exception:
                                    logger.exception(
                                        "[%s] Ошибка при завершении рассылки события",
                                        self.name,
                                    )

                            future.add_done_callback(_finalize)
                        except (RuntimeError, ValueError) as exc:
                            logger.warning(
                                "[%s] Не удалось отправить событие подписчикам: %s",
                                self.name,
                                exc,
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

    def get_visual_frame_jpeg(self):
        return self.last_visual_jpeg
