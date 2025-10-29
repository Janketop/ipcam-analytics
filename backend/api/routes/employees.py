"""Маршруты для управления сотрудниками и подбором снимков."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, conint, constr
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.dependencies import get_session
from backend.core.logger import logger
from backend.models import Camera, Employee, FaceSample
from backend.services.employee_recognizer import EmployeeRecognizer
from backend.services.face_embeddings import (
    compute_face_embedding_from_array,
    compute_face_embedding_from_snapshot,
)
from backend.services.snapshots import save_snapshot
from backend.core.config import settings


router = APIRouter()


_MAX_EMPLOYEE_PHOTO_SIZE = 5 * 1024 * 1024  # 5 МБ
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class EmployeeCreateRequest(BaseModel):
    """Запрос на создание карточки сотрудника."""

    name: constr(strip_whitespace=True, min_length=1)  # type: ignore[valid-type]
    account_id: constr(strip_whitespace=True, min_length=1) | None = None  # type: ignore[valid-type]


class FaceSampleAssignRequest(BaseModel):
    """Присвоение снимка существующему сотруднику."""

    employee_id: conint(strict=True, gt=0)  # type: ignore[valid-type]


class FaceSampleMarkRequest(BaseModel):
    """Присвоение статуса снимку."""

    status: Literal[
        FaceSample.STATUS_CLIENT,
        FaceSample.STATUS_DISCARDED,
        FaceSample.STATUS_UNVERIFIED,
    ]


def _serialize_employee(employee: Employee, sample_count: int) -> dict:
    return {
        "id": employee.id,
        "name": employee.name,
        "accountId": employee.account_id,
        "sampleCount": sample_count,
        "createdAt": employee.created_at,
        "updatedAt": employee.updated_at,
    }


def _serialize_face_sample(
    sample: FaceSample,
    camera_name: str | None,
    employee: Employee | None,
) -> dict:
    return {
        "id": sample.id,
        "snapshotUrl": sample.snapshot_url,
        "status": sample.status,
        "capturedAt": sample.captured_at,
        "updatedAt": sample.updated_at,
        "candidateKey": sample.candidate_key,
        "camera": camera_name,
        "eventId": sample.event_id,
        "employee": (
            {
                "id": employee.id,
                "name": employee.name,
                "accountId": employee.account_id,
            }
            if employee is not None
            else None
        ),
    }


@router.get("/employees")
def list_employees(session: Session = Depends(get_session)) -> dict:
    rows = (
        session.query(
            Employee,
            func.coalesce(func.count(FaceSample.id), 0).label("sample_count"),
        )
        .outerjoin(FaceSample, FaceSample.employee_id == Employee.id)
        .group_by(Employee.id)
        .order_by(Employee.name)
        .all()
    )

    return {
        "employees": [
            _serialize_employee(employee, int(sample_count))
            for employee, sample_count in rows
        ]
    }


@router.post("/employees", status_code=status.HTTP_201_CREATED)
def create_employee(
    payload: EmployeeCreateRequest, session: Session = Depends(get_session)
) -> dict:
    name = payload.name.strip()
    account_id = payload.account_id.strip() if payload.account_id else None
    exists = (
        session.query(Employee)
        .filter(func.lower(Employee.name) == func.lower(name))
        .first()
    )
    if exists is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Сотрудник с таким именем уже существует",
        )

    if account_id:
        existing_account = (
            session.query(Employee)
            .filter(
                Employee.account_id.isnot(None),
                func.lower(Employee.account_id) == func.lower(account_id),
            )
            .first()
        )
        if existing_account is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Сотрудник с таким аккаунтом уже существует",
            )

    employee = Employee(name=name, account_id=account_id)
    session.add(employee)
    session.commit()
    session.refresh(employee)

    return {"employee": _serialize_employee(employee, 0)}


@router.get("/face-samples")
def list_face_samples(
    status: str | None = None,
    limit: int = 50,
    session: Session = Depends(get_session),
) -> dict:
    valid_statuses = {
        FaceSample.STATUS_UNVERIFIED,
        FaceSample.STATUS_EMPLOYEE,
        FaceSample.STATUS_CLIENT,
        FaceSample.STATUS_DISCARDED,
    }

    query = (
        session.query(FaceSample, Camera.name, Employee)
        .outerjoin(Camera, FaceSample.camera_id == Camera.id)
        .outerjoin(Employee, FaceSample.employee_id == Employee.id)
    )

    if status:
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail="Неизвестный статус выборки")
        query = query.filter(FaceSample.status == status)

    limit = max(1, min(limit, 200))

    rows = (
        query.order_by(FaceSample.captured_at.desc()).limit(limit).all()
    )

    return {
        "faceSamples": [
            _serialize_face_sample(sample, camera_name, employee)
            for sample, camera_name, employee in rows
        ]
    }


@router.post("/face-samples/{sample_id}/assign")
def assign_face_sample(
    sample_id: int,
    payload: FaceSampleAssignRequest,
    session: Session = Depends(get_session),
) -> dict:
    sample = session.query(FaceSample).filter(FaceSample.id == sample_id).first()
    if sample is None:
        raise HTTPException(status_code=404, detail="Снимок не найден")

    employee = (
        session.query(Employee)
        .filter(Employee.id == int(payload.employee_id))
        .first()
    )
    if employee is None:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")

    if not sample.snapshot_url:
        raise HTTPException(
            status_code=422,
            detail="У снимка отсутствует файл изображения для расчёта эмбеддинга",
        )

    try:
        embedding_result = compute_face_embedding_from_snapshot(
            sample.snapshot_url,
            encoding_model=settings.face_recognition_model,
        )
    except Exception:
        logger.exception(
            "Не удалось вычислить эмбеддинг лица для выборки #%s", sample.id
        )
        raise HTTPException(
            status_code=500,
            detail="Ошибка при вычислении эмбеддинга лица. Проверьте логи сервиса.",
        ) from None

    if embedding_result is None:
        raise HTTPException(
            status_code=422,
            detail="Не удалось найти лицо на снимке. Загрузите более чёткое фото сотрудника.",
        )

    sample.mark_employee(employee.id)
    sample.set_embedding(
        embedding_result.as_bytes(),
        dim=embedding_result.dimension,
        model=embedding_result.model,
    )
    sample.updated_at = datetime.now(timezone.utc)
    session.add(sample)
    session.commit()
    session.refresh(sample)

    EmployeeRecognizer.notify_embeddings_updated()

    return {
        "faceSample": _serialize_face_sample(
            sample,
            session.query(Camera.name)
            .filter(Camera.id == sample.camera_id)
            .scalar(),
            employee,
        )
    }


@router.post("/face-samples/{sample_id}/mark")
def mark_face_sample(
    sample_id: int,
    payload: FaceSampleMarkRequest,
    session: Session = Depends(get_session),
) -> dict:
    sample = session.query(FaceSample).filter(FaceSample.id == sample_id).first()
    if sample is None:
        raise HTTPException(status_code=404, detail="Снимок не найден")

    should_notify = False
    if payload.status == FaceSample.STATUS_CLIENT:
        prev_state = (sample.status, sample.employee_id, sample.embedding)
        sample.mark_client()
        sample.clear_embedding()
        should_notify = prev_state != (sample.status, sample.employee_id, sample.embedding)
    elif payload.status == FaceSample.STATUS_DISCARDED:
        prev_state = (sample.status, sample.employee_id, sample.embedding)
        sample.status = FaceSample.STATUS_DISCARDED
        sample.employee_id = None
        sample.clear_embedding()
        should_notify = prev_state != (sample.status, sample.employee_id, sample.embedding)
    else:
        prev_state = (sample.status, sample.employee_id, sample.embedding)
        sample.status = FaceSample.STATUS_UNVERIFIED
        sample.employee_id = None
        sample.clear_embedding()
        should_notify = prev_state != (sample.status, sample.employee_id, sample.embedding)

    sample.updated_at = datetime.now(timezone.utc)
    session.add(sample)
    session.commit()
    session.refresh(sample)

    employee = None
    if sample.employee_id is not None:
        employee = (
            session.query(Employee)
            .filter(Employee.id == sample.employee_id)
            .first()
        )

    camera_name = (
        session.query(Camera.name)
        .filter(Camera.id == sample.camera_id)
        .scalar()
    )

    if should_notify:
        EmployeeRecognizer.notify_embeddings_updated()

    return {
        "faceSample": _serialize_face_sample(sample, camera_name, employee)
    }


@router.post(
    "/employees/{employee_id}/photos",
    status_code=status.HTTP_201_CREATED,
)
async def upload_employee_photo(
    employee_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict:
    employee = (
        session.query(Employee).filter(Employee.id == int(employee_id)).first()
    )
    if employee is None:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")

    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Поддерживаются только изображения в форматах JPEG или PNG",
        )

    filename = Path(file.filename or "")
    if filename.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Расширение файла не поддерживается. Используйте .jpg или .png",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Файл изображения пуст",
        )

    if len(raw_bytes) > _MAX_EMPLOYEE_PHOTO_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Файл слишком большой. Максимальный размер — 5 МБ",
        )

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Не удалось прочитать изображение. Загрузите корректный файл JPG или PNG",
        )

    try:
        embedding_result = compute_face_embedding_from_array(
            image_bgr,
            encoding_model=settings.face_recognition_model,
        )
    except Exception:  # pragma: no cover - логирование критичных ошибок
        logger.exception(
            "Не удалось вычислить эмбеддинг лица для сотрудника #%s", employee.id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при вычислении эмбеддинга лица. Проверьте логи сервиса.",
        ) from None

    if embedding_result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Лицо не найдено на изображении. Загрузите более чёткое фото сотрудника.",
        )

    ts = datetime.now(timezone.utc)
    snapshot_url = save_snapshot(
        image_bgr,
        ts=ts,
        camera_name=f"employee_{employee.id}",
        event_type="employee",
    )
    if not snapshot_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось сохранить изображение сотрудника на сервере",
        )

    sample = FaceSample(
        employee_id=employee.id,
        snapshot_url=snapshot_url,
        status=FaceSample.STATUS_EMPLOYEE,
        captured_at=ts,
        updated_at=ts,
    )
    sample.set_embedding(
        embedding_result.as_bytes(),
        dim=embedding_result.dimension,
        model=embedding_result.model,
    )
    session.add(sample)
    session.commit()
    session.refresh(sample)

    EmployeeRecognizer.notify_embeddings_updated()

    return {
        "faceSample": _serialize_face_sample(sample, None, employee),
    }


__all__ = [
    "router",
]
