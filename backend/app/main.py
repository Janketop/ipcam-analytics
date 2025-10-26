import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from .storage import get_engine
from .ingest import IngestManager, env_flag, cleanup_expired_events_and_snapshots

FACE_BLUR = env_flag("FACE_BLUR", False)
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "7"))
CLEANUP_INTERVAL_HOURS = float(os.getenv("RETENTION_CLEANUP_INTERVAL_HOURS", "6"))

app = FastAPI(title="IPCam Analytics (RU)")

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

frontend_origins = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_URL") or ""
allow_origins = {origin.strip() for origin in frontend_origins.split(",") if origin.strip()}
default_origins = {"http://localhost:3000", "http://127.0.0.1:3000"}
allow_origins.update(default_origins)
origin_regex = os.getenv("FRONTEND_ORIGIN_REGEX") or r"https?://.*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allow_origins),
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = get_engine()
ingest = IngestManager(engine)

cleanup_state = {
    "last_run": None,  # type: Optional[datetime]
    "deleted_events": 0,
    "deleted_snapshots": 0,
    "error": None,
    "cutoff": None,
}

cleanup_lock = asyncio.Lock()
background_tasks: List[asyncio.Task] = []


async def perform_cleanup():
    async with cleanup_lock:
        started_at = datetime.now(timezone.utc)
        try:
            deleted_events, deleted_snapshots, cutoff_dt = await asyncio.to_thread(
                cleanup_expired_events_and_snapshots,
                engine,
                RETENTION_DAYS,
            )
            cleanup_state.update(
                {
                    "last_run": started_at,
                    "deleted_events": deleted_events,
                    "deleted_snapshots": deleted_snapshots,
                    "error": None,
                    "cutoff": cutoff_dt,
                }
            )
        except Exception as exc:  # pragma: no cover - логирование ошибок фоновой задачи
            cleanup_state.update(
                {
                    "last_run": started_at,
                    "error": str(exc),
                    "cutoff": None,
                }
            )


async def cleanup_loop():
    # Минимальный интервал в 1 час, чтобы избежать слишком частых запусков по ошибке
    interval_hours = max(CLEANUP_INTERVAL_HOURS, 1.0)
    interval_seconds = interval_hours * 3600
    while True:
        await perform_cleanup()
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:  # pragma: no cover - корректное завершение
            break

class CameraCreate(BaseModel):
    name: str
    rtsp_url: str

@app.on_event("startup")
async def startup_event():
    # Инициализация камер из RTSP_SOURCES
    sources = os.getenv("RTSP_SOURCES", "")
    if sources.strip():
        entries = []
        for item in sources.split(","):
            if "|" in item:
                name, url = item.split("|", 1)
            else:
                name, url = f"cam_{hash(item)%10000}", item
            entries.append((name.strip(), url.strip()))

        if entries:
            with engine.begin() as con:
                for name, url in entries:
                    con.execute(
                        text(
                            "INSERT INTO cameras(name, rtsp_url) VALUES (:n,:u) "
                            "ON CONFLICT (name) DO NOTHING"
                        ),
                        {"n": name, "u": url},
                    )
    # Запускаем воркеры
    main_loop = asyncio.get_running_loop()
    ingest.set_main_loop(main_loop)
    await ingest.start_all()

    task = asyncio.create_task(cleanup_loop(), name="retention-cleanup")
    background_tasks.append(task)


@app.on_event("shutdown")
async def shutdown_event():
    ingest.stop_all()
    for task in background_tasks:
        task.cancel()
    for task in background_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass

@app.get("/health")
def health():
    last_run = cleanup_state.get("last_run")
    cutoff = cleanup_state.get("cutoff")
    return {
        "ok": True,
        "face_blur": FACE_BLUR,
        "retention_days": RETENTION_DAYS,
        "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
        "cleanup": {
            "last_run": last_run.isoformat() if isinstance(last_run, datetime) else None,
            "cutoff": cutoff.isoformat() if isinstance(cutoff, datetime) else None,
            "deleted_events": cleanup_state.get("deleted_events", 0),
            "deleted_snapshots": cleanup_state.get("deleted_snapshots", 0),
            "error": cleanup_state.get("error"),
        },
    }


@app.get("/runtime")
def runtime():
    return ingest.runtime_status()

@app.get("/cameras")
def list_cameras():
    with engine.connect() as con:
        rows = con.execute(text("SELECT id, name FROM cameras WHERE active=true ORDER BY id")).mappings().all()
        return {"cameras": list(rows)}

@app.get("/events")
def list_events(limit: int = 200, type: str | None = None):
    q = "SELECT e.id, e.type, e.start_ts, e.end_ts, e.confidence, e.snapshot_url, e.meta, c.name AS camera FROM events e JOIN cameras c ON e.camera_id = c.id "
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
