import os
import asyncio
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text

from .storage import get_engine
from .ingest import IngestManager, env_flag

FACE_BLUR = env_flag("FACE_BLUR", False)
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "7"))

app = FastAPI(title="IPCam Analytics (RU)")

frontend_origins = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_URL") or ""
allow_origins = {origin.strip() for origin in frontend_origins.split(",") if origin.strip()}
default_origins = {"http://localhost:3000", "http://127.0.0.1:3000"}
allow_origins.update(default_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = get_engine()
ingest = IngestManager(engine)

class CameraCreate(BaseModel):
    name: str
    rtsp_url: str

@app.on_event("startup")
async def startup_event():
    # Инициализация камер из RTSP_SOURCES
    sources = os.getenv("RTSP_SOURCES", "")
    con = engine.connect()
    if sources.strip():
        for item in sources.split(","):
            if "|" in item:
                name, url = item.split("|", 1)
            else:
                name, url = f"cam_{hash(item)%10000}", item
            con.execute(text("INSERT INTO cameras(name, rtsp_url) VALUES (:n,:u) ON CONFLICT (name) DO NOTHING"), {"n": name.strip(), "u": url.strip()})
        con.commit()
    # Запускаем воркеры
    await ingest.start_all()

@app.get("/health")
def health():
    return {"ok": True, "face_blur": FACE_BLUR, "retention_days": RETENTION_DAYS}

@app.get("/cameras")
def list_cameras():
    with engine.connect() as con:
        rows = con.execute(text("SELECT id, name FROM cameras WHERE active=true ORDER BY id")).mappings().all()
        return {"cameras": list(rows)}

@app.get("/events")
def list_events(limit: int = 200, type: str | None = None):
    q = "SELECT e.id, e.type, e.start_ts, e.end_ts, e.confidence, e.snapshot_url, c.name AS camera FROM events e JOIN cameras c ON e.camera_id = c.id "
    if type:
        q += " WHERE e.type=:t "
    q += " ORDER BY e.start_ts DESC LIMIT :lim"
    with engine.connect() as con:
        rows = con.execute(text(q), {"t": type, "lim": limit}).mappings().all()
        return {"events": list(rows)}

# Статистика по типам событий за 24 часа
@app.get("/stats")
def stats():
    with engine.connect() as con:
        rows = con.execute(text("""
            SELECT type, COUNT(*) AS cnt
            FROM events
            WHERE start_ts > now() - interval '1 day'
            GROUP BY type
            ORDER BY cnt DESC
        """)).mappings().all()
    return {"stats": list(rows)}

# Простой WS для лайв-ивентов
clients: List[WebSocket] = []

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        pass
    finally:
        clients.remove(ws)

# Вещание события
async def broadcast_event(ev: dict):
    for ws in list(clients):
        try:
            await ws.send_json(ev)
        except:
            pass

ingest.set_broadcaster(broadcast_event)

# MJPEG-live поток с наложенной разметкой
@app.get("/stream/{camera_name}")
def mjpeg_stream(camera_name: str):
    worker = ingest.get_worker(camera_name)
    if not worker:
        return {"error": "Камера не найдена или не активна"}

    boundary = "frame"
    def gen():
        import time
        while True:
            frame = worker.get_visual_frame_jpeg()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                   frame + b"\r\n")
    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")
