"""Маршруты, связанные с камерами."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.core.dependencies import get_ingest_manager, get_session
from backend.models import Camera

router = APIRouter()


@router.get("/cameras")
def list_cameras(session: Session = Depends(get_session)):
    cameras = (
        session.query(Camera)
        .filter(Camera.active.is_(True))
        .order_by(Camera.id)
        .all()
    )
    return {
        "cameras": [
            {
                "id": camera.id,
                "name": camera.name,
            }
            for camera in cameras
        ]
    }


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
