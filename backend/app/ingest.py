import os, cv2, time, asyncio, json
from typing import Optional
from collections import deque
from datetime import datetime, timezone, timedelta
from threading import Thread
from uuid import uuid4

from sqlalchemy import text
import numpy as np
from ultralytics import YOLO
from easyocr import Reader
try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in tests
    torch = None

PHONE_CLASS = 'cell phone'

def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, min_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if min_value is not None:
        value = max(value, min_value)
    return value


def env_float(name: str, default: float, min_value: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if min_value is not None:
        value = max(value, min_value)
    return value


def resolve_device(preferred: Optional[str] = None) -> str:
    """Выбираем устройство для инференса: GPU, если доступно, иначе CPU."""
    if preferred and preferred.strip().lower() not in {"auto", ""}:
        return preferred

    # auto-режим: сначала CUDA, затем Apple MPS, в конце CPU
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


class IngestWorker(Thread):
    def __init__(self, engine, cam_id, name, rtsp_url, face_blur=True, broadcaster=None):
        super().__init__(daemon=True)
        self.engine = engine
        self.cam_id = cam_id
        self.name = name
        self.url = rtsp_url
        self.face_blur = face_blur
        self.broadcaster = broadcaster
        self.stop_flag = False

        self.visualize = env_flag("VISUALIZE", False)
        self.last_visual_jpeg = None

        det_weights = os.getenv("YOLO_DET_MODEL", "yolov8n.pt")
        pose_weights = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
        self.device_preference = os.getenv("YOLO_DEVICE", "auto")
        self.device = resolve_device(self.device_preference)

        # Lightweight модели
        self.det = YOLO(det_weights)
        self.pose = YOLO(pose_weights)
        self.device_error: Optional[str] = None
        self.actual_device: Optional[str] = None
        try:
            self.det.to(self.device)
            self.pose.to(self.device)
        except Exception as exc:
            # Оставляем модели на устройстве по умолчанию, если перевод не поддерживается
            self.device_error = str(exc).strip() or None
        finally:
            det_model = getattr(self.det, "model", None)
            model_device = getattr(det_model, "device", None)
            if model_device is not None:
                self.actual_device = str(model_device)
            else:
                self.actual_device = str(self.device)

        # аргумент device для прямых вызовов модели; ultralytics принимает 'cuda:0' и т.п.
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
        self.last_car_event_time = 0.0

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        self.ocr_langs = os.getenv("PLATE_OCR_LANGS", "ru,en")
        self.ocr_reader: Optional[Reader] = None
        self._ocr_failed = False

        self.phone_idx = None

        self.phone_active_until = None
        self.score_buffer = deque(maxlen=self.score_smoothing)

        self.face_cascade = None
        if self.face_blur:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.face_cascade = None

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
                reason = "Авто-режим выбрал CPU: перевести модель на GPU не удалось. Проверьте настройки и логи."  # noqa: E501
            elif pref_lower.startswith("cuda") and not actual.lower().startswith("cuda"):
                reason = "Модель не была загружена на указанное GPU устройство. Проверьте его доступность."

        clean_error = _cleanup(self.device_error) if self.device_error else None
        clean_reason = _cleanup(reason) if reason else None

        info = {
            "camera": self.name,
            "preferred_device": preferred,
            "selected_device": selected,
            "actual_device": actual,
            "using_gpu": using_gpu,
            "visualize_enabled": bool(self.visualize),
            "device_error": clean_error,
            "gpu_unavailable_reason": clean_reason,
        }
        return info

    def run(self):
        reconnect_delay = env_float("RTSP_RECONNECT_DELAY", 5.0, min_value=0.5)
        max_failed_reads = env_int("RTSP_MAX_FAILED_READS", 25, min_value=1)
        fps_skip = env_int("INGEST_FPS_SKIP", 2, min_value=1)
        flush_timeout = env_float("INGEST_FLUSH_TIMEOUT", 0.2, min_value=0.0)

        while not self.stop_flag:
            cap = cv2.VideoCapture(self.url)
            # минимальная буферизация, если поддерживается
            try:
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if not cap.isOpened():
                print(f"[{self.name}] не удалось открыть поток, повтор через {reconnect_delay} с")
                cap.release()
                time.sleep(reconnect_delay)
                continue

            frame_id = 0
            failed_reads = 0
            reconnect_needed = False

            while not self.stop_flag:
                ok, frame = cap.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        reconnect_needed = True
                        print(f"[{self.name}] потеряно соединение, переподключаюсь через {reconnect_delay} с")
                        break
                    time.sleep(0.2)
                    continue

                failed_reads = 0
                frame_id += 1
                if frame_id % fps_skip != 0:
                    continue

                # сброс накопленных кадров, чтобы обрабатывать самый свежий
                flush_start = time.time()
                grabbed_any = False
                while time.time() - flush_start < flush_timeout:
                    ok_grab = cap.grab()
                    if not ok_grab:
                        failed_reads += 1
                        if failed_reads >= max_failed_reads:
                            reconnect_needed = True
                            print(f"[{self.name}] потеряно соединение, переподключаюсь через {reconnect_delay} с")
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
                            print(f"[{self.name}] потеряно соединение, переподключаюсь через {reconnect_delay} с")
                            break
                        time.sleep(0.2)
                        continue
                    frame = latest_frame

                phone_usage_raw, conf_raw, snapshot, vis, car_events = self.process_frame(frame)

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
                else:
                    if self.phone_active_until and now > self.phone_active_until:
                        # TODO: логика NOT_WORKING по окну времени
                        pass

                # Обновляем live-кадр
                if self.visualize and vis is not None:
                    try:
                        ret, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ret:
                            self.last_visual_jpeg = buf.tobytes()
                    except Exception:
                        pass

                if ev_type:
                    events_to_store.append({
                        "ts": now,
                        "type": ev_type,
                        "confidence": float(conf),
                        "snapshot": snapshot,
                        "meta": {},
                        "kind": "phone",
                    })

                for car_ev in car_events:
                    car_now = datetime.now(timezone.utc)
                    meta = {
                        "plate": car_ev.get("plate") or "НЕ РАСПОЗНАНО",
                        "entry_ts": car_now.isoformat(),
                    }
                    events_to_store.append({
                        "ts": car_now,
                        "type": "CAR_ENTRY",
                        "confidence": float(car_ev.get("confidence", 0.0)),
                        "snapshot": car_ev.get("snapshot"),
                        "meta": meta,
                        "kind": "car",
                    })

                for payload in events_to_store:
                    snap_url = self.save_snapshot(payload["snapshot"], payload["ts"], event_type=payload["kind"])
                    with self.engine.begin() as con:
                        con.execute(
                            text(
                                "INSERT INTO events(camera_id,type,start_ts,confidence,snapshot_url,meta) "
                                "VALUES (:c,:t,:s,:conf,:u,CAST(:m AS jsonb))"
                            ),
                            {
                                "c": self.cam_id,
                                "t": payload["type"],
                                "s": payload["ts"],
                                "conf": payload["confidence"],
                                "u": snap_url,
                                "m": json.dumps(payload["meta"]),
                            },
                        )
                    if self.broadcaster:
                        asyncio.run(
                            self.broadcaster(
                                {
                                    "camera": self.name,
                                    "type": payload["type"],
                                    "ts": payload["ts"].isoformat(),
                                    "confidence": payload["confidence"],
                                    "snapshot_url": snap_url,
                                    "meta": payload["meta"],
                                }
                            )
                        )

            cap.release()
            if self.stop_flag:
                break

            if reconnect_needed:
                time.sleep(reconnect_delay)

    def get_visual_frame_jpeg(self):
        """Возвращает последний сохранённый визуализированный кадр в формате JPEG."""
        return self.last_visual_jpeg

    def prepare_snapshot(self, img_bgr):
        snap = img_bgr.copy()
        if self.face_blur and self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    roi = snap[y:y + h, x:x + w]
                    roi = cv2.GaussianBlur(roi, (99, 99), 30)
                    snap[y:y + h, x:x + w] = roi
            except Exception:
                pass
        return snap

    def ensure_ocr_reader(self):
        if self._ocr_failed:
            return None
        if self.ocr_reader is None:
            langs = [lang.strip() for lang in self.ocr_langs.split(",") if lang.strip()]
            if not langs:
                langs = ["en"]
            try:
                self.ocr_reader = Reader(langs, gpu=False)
            except Exception as exc:
                print(f"[{self.name}] не удалось инициализировать OCR: {exc}")
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
            print(f"[{self.name}] ошибка OCR: {exc}")
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
        return self.prepare_snapshot(snap)

    def process_frame(self, frame):
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

        phones = []
        cars = []
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
                    phones.append({
                        "bbox": xyxy,
                        "center": np.array([cx, cy], dtype=np.float32),
                        "conf": conf,
                    })
                if cls_idx in self.car_class_ids and conf >= self.car_conf_threshold:
                    cars.append({
                        "bbox": xyxy,
                        "conf": conf,
                    })

        pose_kwargs = {"imgsz": self.det_imgsz, "conf": self.pose_conf}
        if self.predict_device:
            pose_kwargs["device"] = self.predict_device
        pose_res = self.pose(frame, **pose_kwargs)[0]
        phone_usage = False
        best_conf = 0.0

        # визуализируем на копии
        need_overlay = self.visualize
        vis = frame.copy() if need_overlay else None

        car_events = []
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
            car_events.append({
                "plate": plate_text,
                "confidence": float(car["conf"]),
                "snapshot": car_snapshot,
            })
            self.last_car_event_time = now_monotonic

        if pose_res.boxes is None or pose_res.keypoints is None:
            people_iter = []
        else:
            kpts_xy = pose_res.keypoints.xy.cpu().numpy()
            kpts_conf = None
            if getattr(pose_res.keypoints, "conf", None) is not None:
                kpts_conf = pose_res.keypoints.conf.cpu().numpy()
            people_iter = zip(kpts_xy, pose_res.boxes.xyxy.cpu().numpy(), kpts_conf if kpts_conf is not None else [None] * len(kpts_xy))

        for kpts, bbox, kconf in people_iter:
            # наклон головы (приближение)
            try:
                le = kpts[1]; re = kpts[2]
                head_tilt = abs(le[1]-re[1]) / (abs(le[0]-re[0])+1e-3)
            except Exception:
                head_tilt = 0.0

            bbox_h = max(1.0, float(bbox[3] - bbox[1]))

            # подготовка ключевых точек рук и головы
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

                if need_overlay:
                    cx, cy = map(int, phone["center"])
                    color = (0, 0, 255) if score >= self.phone_score_threshold else (0, 165, 255)
                    cv2.circle(vis, (cx, cy), 6, color, -1)
                    if wrists:
                        closest = min(wrists, key=lambda p: np.linalg.norm(phone["center"] - p))
                        cv2.line(vis, (cx, cy), tuple(map(int, closest)), color, 2)
                    if head_center is not None:
                        cv2.line(vis, (cx, cy), tuple(map(int, head_center)), (200, 50, 200), 1)

            # рисуем bbox и скелет
            if need_overlay:
                x1,y1,x2,y2 = map(int, bbox)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, 'PERSON', (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                # скелет (несколько связей)
                pairs = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(11,13),(13,15),(12,14),(14,16)]
                for (i,j) in pairs:
                    if i < len(kpts) and j < len(kpts):
                        p1 = tuple(map(int, kpts[i]))
                        p2 = tuple(map(int, kpts[j]))
                        cv2.line(vis, p1, p2, (255,0,0), 2)

            if not phones and wrists:
                # эвристика: руки возле головы, запястья близко друг к другу, голова наклонена
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
                    if need_overlay:
                        center_y = int((bbox[1] + bbox[3]) * 0.5)
                        cv2.putText(vis, 'POSE PHONE', (x1, max(15, center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # рамки телефонов и подписи
        if need_overlay:
            for phone in phones:
                x1,y1,x2,y2 = phone["bbox"]
                cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 2)
                cv2.putText(vis, f'PHONE {phone["conf"]:.2f}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if phone_usage and self.visualize and vis is not None:
            cv2.putText(vis, f'PHONE_USAGE ({best_conf:.2f})', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # snapshot (опционально размываем лица)
        snap_src = vis if self.visualize and vis is not None else frame
        snap = self.prepare_snapshot(snap_src)

        return phone_usage, best_conf, snap, vis if self.visualize else None, car_events

    def save_snapshot(self, img_bgr, ts, event_type="event"):
        if img_bgr is None:
            return ""
        out_dir = "/app/app/static/snaps"
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{self.name}_{event_type}_{int(ts.timestamp())}_{uuid4().hex[:6]}.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, img_bgr)
        return f"/static/snaps/{fname}"

class IngestManager:
    def __init__(self, engine):
        self.engine = engine
        self.workers = []
        self.broadcaster = None

    def set_broadcaster(self, fn):
        self.broadcaster = fn

    async def start_all(self):
        from sqlalchemy import text
        with self.engine.connect() as con:
            cams = con.execute(text("SELECT id,name,rtsp_url FROM cameras WHERE active=true")).mappings().all()
        face_blur = env_flag("FACE_BLUR", False)
        for c in cams:
            w = IngestWorker(
                self.engine,
                c["id"],
                c["name"],
                c["rtsp_url"],
                face_blur=face_blur,
                broadcaster=self.broadcaster,
            )
            w.start()
            self.workers.append(w)

    def stop_all(self):
        for w in self.workers:
            w.stop_flag = True

    def get_worker(self, name):
        for w in self.workers:
            if w.name == name:
                return w
        return None

    def runtime_status(self) -> dict:
        workers = [w.runtime_status() for w in self.workers]

        torch_available = torch is not None
        cuda_available = bool(torch_available and torch.cuda.is_available())
        cuda_device_count = 0
        cuda_name = None
        if cuda_available:
            try:
                cuda_device_count = int(torch.cuda.device_count())
            except Exception:
                cuda_device_count = 0
            if cuda_device_count:
                try:
                    cuda_name = torch.cuda.get_device_name(0)
                except Exception:
                    cuda_name = None

        mps_available = False
        if torch_available and hasattr(torch.backends, "mps"):
            try:
                mps_available = bool(torch.backends.mps.is_available())
            except Exception:
                mps_available = False

        system = {
            "torch_available": torch_available,
            "torch_version": getattr(torch, "__version__", None) if torch_available else None,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_name": cuda_name,
            "mps_available": mps_available,
            "env_device": os.getenv("YOLO_DEVICE", "auto"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        }

        return {
            "system": system,
            "workers": workers,
        }
