"""Маршруты, связанные с камерами."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import AnyUrl, BaseModel, constr, validator
from sqlalchemy.orm import Session

from backend.core.config import settings
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


def _normalize_worker(worker_info: Dict[str, Any] | None) -> Dict[str, Any]:
    if not worker_info:
        return {
            "fps": None,
            "last_frame_at": None,
            "uptime_seconds": None,
        }

    return {
        "fps": worker_info.get("fps"),
        "last_frame_at": worker_info.get("last_frame_at"),
        "uptime_seconds": worker_info.get("uptime_seconds"),
    }


def _calc_status(worker, worker_info: Dict[str, Any] | None) -> str:
    if worker is None:
        return "offline"

    if worker.stop_flag:
        return "stopping"

    if not worker.is_alive():
        return "starting"

    last_frame_at = None
    if worker_info and worker_info.get("last_frame_at"):
        try:
            last_frame_at = datetime.fromisoformat(worker_info["last_frame_at"])
        except ValueError:
            last_frame_at = None

    if last_frame_at is None:
        return "starting"

    if last_frame_at.tzinfo is None:
        last_frame_at = last_frame_at.replace(tzinfo=timezone.utc)

    delta = datetime.now(timezone.utc) - last_frame_at.astimezone(timezone.utc)
    if delta.total_seconds() > settings.ingest_status_stale_threshold:
        return "no_signal"

    return "online"


@router.get("/cameras")
def list_cameras(
    session: Session = Depends(get_session),
    ingest=Depends(get_ingest_manager),
):
    cameras = (
        session.query(Camera)
        .filter(Camera.active.is_(True))
        .order_by(Camera.id)
        .all()
    )
    runtime = ingest.runtime_status() if ingest else {"workers": []}
    workers = runtime.get("workers") or []
    workers_by_name = {
        worker.get("camera"): worker for worker in workers if worker.get("camera")
    }

    items = []
    for camera in cameras:
        worker = ingest.get_worker(camera.name) if ingest else None
        worker_info = workers_by_name.get(camera.name)
        normalized = _normalize_worker(worker_info)
        items.append(
            {
                "id": camera.id,
                "name": camera.name,
                "status": _calc_status(worker, worker_info),
                "fps": normalized.get("fps"),
                "lastFrameTs": normalized.get("last_frame_at"),
                "uptimeSec": normalized.get("uptime_seconds"),
            }
        )

    return {"cameras": items}


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


@router.delete("/api/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_camera(
    camera_id: int,
    session: Session = Depends(get_session),
    ingest=Depends(get_ingest_manager),
):
    """Деактивирует камеру и останавливает соответствующий ingest-воркер."""

    camera = (
        session.query(Camera)
        .filter(Camera.id == camera_id, Camera.active.is_(True))
        .first()
    )
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Активная камера с указанным идентификатором не найдена",
        )

    camera.active = False
    session.add(camera)
    session.commit()

    ingest.stop_worker_for_camera(camera.name)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
