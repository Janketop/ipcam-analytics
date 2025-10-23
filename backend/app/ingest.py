import os, cv2, time, asyncio
from datetime import datetime, timezone, timedelta
from threading import Thread
from sqlalchemy import text
import numpy as np
from PIL import Image
from ultralytics import YOLO

PHONE_CLASS = 'cell phone'

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

        self.visualize = os.getenv("VISUALIZE","true").lower()=="true"
        self.last_visual_jpeg = None

        # Lightweight модели
        self.det = YOLO("yolov8n.pt")
        self.pose = YOLO("yolov8n-pose.pt")
        self.phone_idx = None

        self.phone_active_until = None

    def run(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            print(f"[{self.name}] не удалось открыть поток")
            return

        _ = cap.read()

        fps_skip = 2  # обрабатываем каждый 2-й кадр
        frame_id = 0

        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            frame_id += 1
            if frame_id % fps_skip != 0:
                continue

            phone_usage, conf, snapshot, vis = self.process_frame(frame)

            now = datetime.now(timezone.utc)
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
                snap_url = self.save_snapshot(snapshot, now)
                with self.engine.begin() as con:
                    con.execute(text("INSERT INTO events(camera_id,type,start_ts,confidence,snapshot_url,meta) VALUES (:c,:t,:s,:conf,:u,'{}')"),
                                {"c": self.cam_id, "t": ev_type, "s": now, "conf": float(conf), "u": snap_url})
                if self.broadcaster:
                    asyncio.run(self.broadcaster({"camera": self.name, "type": ev_type, "ts": now.isoformat(), "confidence": float(conf), "snapshot_url": snap_url}))

        cap.release()

    def get_visual_frame_jpeg(self):
        """Возвращает последний сохранённый визуализированный кадр в формате JPEG."""
        return self.last_visual_jpeg

    def process_frame(self, frame):
        det_res = self.det(frame, imgsz=640, conf=0.3)[0]
        if self.phone_idx is None:
            names = self.det.model.names if hasattr(self.det.model, "names") else self.det.names
            self.phone_idx = [k for k,v in names.items() if v == PHONE_CLASS][0]

        phones = []
        for b in det_res.boxes:
            if int(b.cls[0]) == self.phone_idx:
                phones.append(b.xyxy[0].cpu().numpy())

        pose_res = self.pose(frame, imgsz=640, conf=0.3)[0]
        phone_usage = False
        best_conf = 0.0

        # визуализируем на копии
        vis = frame.copy()

        for kpts, bbox in zip(pose_res.keypoints.xy.cpu().numpy(), pose_res.boxes.xyxy.cpu().numpy()):
            # наклон головы (приближение)
            try:
                le = kpts[1]; re = kpts[2]
                head_tilt = abs(le[1]-re[1]) / (abs(le[0]-re[0])+1e-3)
            except Exception:
                head_tilt = 0.0

            # близость телефона к зоне рук
            near = False
            for (x1,y1,x2,y2) in phones:
                hx1, hy1 = bbox[0], bbox[1] + (bbox[3]-bbox[1]) * 0.5
                hx2, hy2 = bbox[2], bbox[3]
                inter_w = max(0, min(x2, hx2) - max(x1, hx1))
                inter_h = max(0, min(y2, hy2) - max(y1, hy1))
                if inter_w*inter_h > 0:
                    near = True
                    break

            score = 0.0
            if near: score += 0.6
            if head_tilt > 0.25: score += 0.4
            if score > 0.6:
                phone_usage = True
                best_conf = max(best_conf, score)

            # рисуем bbox и скелет
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

        # рамки телефонов и подписи
        for (x1,y1,x2,y2) in phones:
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 2)
            cv2.putText(vis, 'PHONE', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if phone_usage:
            cv2.putText(vis, f'PHONE_USAGE ({best_conf:.2f})', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # snapshot (опционально размываем лица)
        snap = vis.copy() if self.visualize else frame.copy()
        if self.face_blur:
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    roi = snap[y:y+h, x:x+w]
                    roi = cv2.GaussianBlur(roi, (99,99), 30)
                    snap[y:y+h, x:x+w] = roi
            except Exception:
                pass

        return phone_usage, best_conf, snap, vis

    def save_snapshot(self, img_bgr, ts):
        out_dir = "/app/app/static/snaps"
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{self.name}_{int(ts.timestamp())}.jpg"
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
        for c in cams:
            w = IngestWorker(self.engine, c["id"], c["name"], c["rtsp_url"], face_blur=os.getenv("FACE_BLUR","true").lower()=="true", broadcaster=self.broadcaster)
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
