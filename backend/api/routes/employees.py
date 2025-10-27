"""Маршруты для управления сотрудниками и подбором снимков."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, conint, constr
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.dependencies import get_session
from backend.models import Camera, Employee, FaceSample


router = APIRouter()


class EmployeeCreateRequest(BaseModel):
    """Запрос на создание карточки сотрудника."""

    name: constr(strip_whitespace=True, min_length=1)  # type: ignore[valid-type]


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
            {"id": employee.id, "name": employee.name}
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

    employee = Employee(name=name)
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

    sample.mark_employee(employee.id)
    sample.updated_at = datetime.now(timezone.utc)
    session.add(sample)
    session.commit()
    session.refresh(sample)

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

    if payload.status == FaceSample.STATUS_CLIENT:
        sample.mark_client()
    elif payload.status == FaceSample.STATUS_DISCARDED:
        sample.status = FaceSample.STATUS_DISCARDED
        sample.employee_id = None
    else:
        sample.status = FaceSample.STATUS_UNVERIFIED
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

    return {
        "faceSample": _serialize_face_sample(sample, camera_name, employee)
    }


__all__ = [
    "router",
]
