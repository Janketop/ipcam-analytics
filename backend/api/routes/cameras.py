"""Маршруты, связанные с камерами."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import AnyUrl, BaseModel, Field, constr, root_validator, validator
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.dependencies import get_ingest_manager, get_session
from backend.models import Camera

_ALLOWED_STREAM_SCHEMES = {"rtsp", "rtsps", "http", "https", "rtmp"}


class ZonePoint(BaseModel):
    """Координата вершины полигона в нормализованных значениях."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class ZonePolygon(BaseModel):
    """Описание зоны детекции как полигона."""

    id: Optional[constr(strip_whitespace=True, min_length=1)] = None  # type: ignore[valid-type]
    name: Optional[constr(strip_whitespace=True, min_length=1)] = None  # type: ignore[valid-type]
    points: List[ZonePoint] = Field(default_factory=list, min_items=3)

    @root_validator(pre=True)
    def _ensure_points(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        points = values.get("points")
        if not points:
            raise ValueError("Зона должна содержать минимум три точки")
        return values


class CameraCreateRequest(BaseModel):
    """Схема запроса на добавление новой камеры."""

    name: constr(strip_whitespace=True, min_length=1)  # type: ignore[valid-type]
    rtsp_url: AnyUrl
    detect_person: bool = True
    detect_car: bool = True
    capture_entry_time: bool = True
    idle_alert_time: int = Field(default=settings.idle_alert_time, ge=10, le=86400)
    zones: List[ZonePolygon] = Field(default_factory=list)

    @validator("rtsp_url")
    def ensure_rtsp_scheme(cls, value: AnyUrl) -> AnyUrl:
        if value.scheme.lower() not in _ALLOWED_STREAM_SCHEMES:
            allowed = ", ".join(sorted(_ALLOWED_STREAM_SCHEMES))
            raise ValueError(
                f"URL должен использовать одну из поддерживаемых схем: {allowed}"
            )
        return value


class CameraUpdateRequest(BaseModel):
    """Схема запроса на обновление настроек камеры."""

    name: Optional[constr(strip_whitespace=True, min_length=1)]  # type: ignore[valid-type]
    rtsp_url: Optional[AnyUrl]
    detect_person: Optional[bool]
    detect_car: Optional[bool]
    capture_entry_time: Optional[bool]
    idle_alert_time: Optional[int] = Field(default=None, ge=10, le=86400)
    zones: Optional[List[ZonePolygon]]

    @validator("rtsp_url")
    def ensure_rtsp_scheme(cls, value: AnyUrl | None) -> AnyUrl | None:
        if value is None:
            return value
        if value.scheme.lower() not in _ALLOWED_STREAM_SCHEMES:
            allowed = ", ".join(sorted(_ALLOWED_STREAM_SCHEMES))
            raise ValueError(
                f"URL должен использовать одну из поддерживаемых схем: {allowed}"
            )
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


def _camera_payload(camera: Camera) -> Dict[str, Any]:
    return {
        "id": camera.id,
        "name": camera.name,
        "rtspUrl": camera.rtsp_url,
        "active": camera.active,
        "detectPerson": camera.detect_person,
        "detectCar": camera.detect_car,
        "captureEntryTime": camera.capture_entry_time,
        "idleAlertTime": camera.idle_alert_time or settings.idle_alert_time,
        "zones": camera.zones or [],
    }


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
        payload = _camera_payload(camera)
        payload.update(
            {
                "status": _calc_status(worker, worker_info),
                "fps": normalized.get("fps"),
                "lastFrameTs": normalized.get("last_frame_at"),
                "uptimeSec": normalized.get("uptime_seconds"),
            }
        )
        items.append(payload)

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

    zones_payload = [
        {
            "id": zone.id,
            "name": zone.name,
            "points": [point.dict() for point in zone.points],
        }
        for zone in payload.zones
    ]

    camera = Camera(
        name=name,
        rtsp_url=str(payload.rtsp_url),
        detect_person=payload.detect_person,
        detect_car=payload.detect_car,
        capture_entry_time=payload.capture_entry_time,
        idle_alert_time=payload.idle_alert_time,
        zones=zones_payload,
    )
    session.add(camera)
    session.commit()
    session.refresh(camera)

    ingest.start_worker_for_camera(camera)

    return {"camera": _camera_payload(camera)}


@router.patch("/api/cameras/{camera_id}")
def update_camera(
    camera_id: int,
    payload: CameraUpdateRequest,
    session: Session = Depends(get_session),
    ingest=Depends(get_ingest_manager),
):
    camera = session.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Камера с указанным идентификатором не найдена",
        )

    updates = payload.dict(exclude_unset=True)
    zones_provided = "zones" in payload.__fields_set__
    if not updates:
        return {"camera": _camera_payload(camera)}

    original_name = camera.name
    original_rtsp = camera.rtsp_url
    original_detect_person = camera.detect_person
    original_detect_car = camera.detect_car
    original_capture_entry_time = camera.capture_entry_time
    original_idle_alert_time = camera.idle_alert_time
    original_zones = camera.zones or []

    new_name = updates.get("name")
    if new_name:
        name = new_name.strip()
        if name != camera.name:
            existing = (
                session.query(Camera)
                .filter(Camera.name == name, Camera.id != camera.id)
                .first()
            )
            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Камера с таким именем уже существует",
                )
            camera.name = name

    if "rtsp_url" in updates and updates["rtsp_url"] is not None:
        camera.rtsp_url = str(updates["rtsp_url"])

    for field in ("detect_person", "detect_car", "capture_entry_time"):
        if field in updates and updates[field] is not None:
            setattr(camera, field, bool(updates[field]))

    if "idle_alert_time" in updates and updates["idle_alert_time"] is not None:
        camera.idle_alert_time = int(updates["idle_alert_time"])

    if zones_provided:
        zone_models = payload.zones or []
        camera.zones = [
            {
                "id": zone.id,
                "name": zone.name,
                "points": [point.dict() for point in zone.points],
            }
            for zone in zone_models
        ]

    session.add(camera)
    session.commit()
    session.refresh(camera)

    if ingest:
        name_changed = camera.name != original_name
        rtsp_changed = camera.rtsp_url != original_rtsp
        flags_changed = any(
            [
                camera.detect_person != original_detect_person,
                camera.detect_car != original_detect_car,
                camera.capture_entry_time != original_capture_entry_time,
                camera.idle_alert_time != original_idle_alert_time,
            ]
        )
        zones_changed = camera.zones != original_zones

        if name_changed or rtsp_changed:
            ingest.stop_worker_for_camera(original_name)
            if camera.active:
                ingest.start_worker_for_camera(camera)
        elif camera.active and (flags_changed or zones_changed):
            worker = ingest.get_worker(camera.name)
            if worker is not None:
                if flags_changed:
                    worker.update_flags(
                        detect_person=camera.detect_person,
                        detect_car=camera.detect_car,
                        capture_entry_time=camera.capture_entry_time,
                        idle_alert_time=camera.idle_alert_time,
                    )
                if zones_changed:
                    worker.update_zones(camera.zones)

    return {"camera": _camera_payload(camera)}


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
