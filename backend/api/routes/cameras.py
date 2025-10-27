"""Маршруты, связанные с камерами."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import AnyUrl, BaseModel, constr, validator
from sqlalchemy.orm import Session

from backend.core.dependencies import get_ingest_manager, get_session
from backend.models import Camera


class CameraCreateRequest(BaseModel):
    """Схема запроса на добавление новой камеры."""

    name: constr(strip_whitespace=True, min_length=1)  # type: ignore[valid-type]
    rtsp_url: AnyUrl

    @validator("rtsp_url")
    def ensure_rtsp_scheme(cls, value: AnyUrl) -> AnyUrl:
        if value.scheme.lower() != "rtsp":
            raise ValueError("URL должен использовать схему rtsp")
        return value


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


@router.post("/api/cameras/add", status_code=status.HTTP_201_CREATED)
def add_camera(
    payload: CameraCreateRequest,
    session: Session = Depends(get_session),
    ingest=Depends(get_ingest_manager),
):
    """Добавляет новую камеру и запускает обработчик видеопотока."""

    name = payload.name.strip()
    existing_camera = session.query(Camera).filter(Camera.name == name).first()
    if existing_camera is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Камера с таким именем уже существует",
        )

    camera = Camera(name=name, rtsp_url=str(payload.rtsp_url))
    session.add(camera)
    session.commit()
    session.refresh(camera)

    ingest.start_worker_for_camera(camera)

    return {
        "camera": {
            "id": camera.id,
            "name": camera.name,
            "rtsp_url": camera.rtsp_url,
            "active": camera.active,
        }
    }
