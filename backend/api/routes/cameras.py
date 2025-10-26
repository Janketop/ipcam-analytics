"""Маршруты, связанные с камерами."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from backend.core.dependencies import get_engine, get_ingest_manager

router = APIRouter()


@router.get("/cameras")
def list_cameras(engine=Depends(get_engine)):
    with engine.connect() as con:
        rows = (
            con.execute(text("SELECT id, name FROM cameras WHERE active=true ORDER BY id")).mappings().all()
        )
        return {"cameras": list(rows)}


@router.get("/stream/{camera_name}")
def mjpeg_stream(camera_name: str, ingest=Depends(get_ingest_manager)):
    worker = ingest.get_worker(camera_name)
    if not worker:
        return {"error": "Камера не найдена или не активна"}

    boundary = "frame"

    def frame_generator():
        while True:
            frame = worker.get_visual_frame_jpeg()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + b"Content-Length: "
                + str(len(frame)).encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )

    media_type = f"multipart/x-mixed-replace; boundary={boundary}"
    return StreamingResponse(frame_generator(), media_type=media_type)
