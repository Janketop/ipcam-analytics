"""Маршруты, связанные с обучением моделей внутри сервиса."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from backend.services.training import (
    FaceDetectorTrainingService,
    FaceTrainingOptions,
    SelfTrainingService,
    build_face_training_options,
)

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/self", status_code=status.HTTP_202_ACCEPTED)
async def schedule_self_training(request: Request) -> dict:
    """Запускает дообучение модели в фоне."""

    service = getattr(request.app.state, "self_training_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис самообучения временно недоступен.",
        )

    if not isinstance(service, SelfTrainingService):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Неверная конфигурация сервиса самообучения.",
        )
    started = await service.start_training(request.app)
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Обучение уже запущено, дождитесь завершения текущего запуска.",
        )
    return {"status": "scheduled"}


class FaceDetectorTrainingRequest(BaseModel):
    """Параметры запуска обучения детектора лиц через API."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    dataset_root: str | None = Field(
        None,
        alias="datasetRoot",
        description="Каталог с данными WIDERFace. Относительные пути вычисляются от корня проекта.",
    )
    skip_download: bool | None = Field(
        None,
        alias="skipDownload",
        description="Если True, архивы WIDERFace не будут скачиваться повторно.",
    )
    epochs: int | None = Field(
        None,
        ge=1,
        description="Количество эпох обучения. По умолчанию используется значение из настроек.",
    )
    batch: int | None = Field(
        None,
        ge=1,
        description="Размер batch для обучения.",
    )
    img_size: int | None = Field(
        None,
        alias="imgSize",
        ge=32,
        description="Размер входного изображения для Ultralytics YOLO.",
    )
    device: str | None = Field(
        None,
        description="Устройство для обучения (например, 'cuda:0').",
    )
    project_dir: str | None = Field(
        None,
        alias="projectDir",
        description="Каталог, где будут храниться логи обучения.",
    )
    run_name: str | None = Field(
        None,
        alias="runName",
        description="Имя подпапки с логами обучения.",
    )
    base_weights: str | None = Field(
        None,
        alias="baseWeights",
        description="Путь до базовых весов YOLO11n.",
    )
    weights_output: str | None = Field(
        None,
        alias="weightsOutput",
        description="Куда скопировать итоговые веса детектора лиц.",
    )


@router.post("/face-detector", status_code=status.HTTP_202_ACCEPTED)
async def schedule_face_detector_training(
    payload: FaceDetectorTrainingRequest,
    request: Request,
) -> dict:
    """Запускает обучение детектора лиц на датасете WIDERFace."""

    service = getattr(request.app.state, "face_training_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис обучения детектора лиц временно недоступен.",
        )
    if not isinstance(service, FaceDetectorTrainingService):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Неверная конфигурация сервиса обучения детектора лиц.",
        )

    options: FaceTrainingOptions = build_face_training_options(
        dataset_root=payload.dataset_root,
        skip_download=payload.skip_download,
        epochs=payload.epochs,
        batch=payload.batch,
        imgsz=payload.img_size,
        device=payload.device,
        project=payload.project_dir,
        run_name=payload.run_name,
        base_weights=payload.base_weights,
        output_weights=payload.weights_output,
    )

    started = await service.start_training(request.app, options)
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Обучение детектора лиц уже запущено, дождитесь завершения текущего запуска.",
        )

    return {"status": "scheduled"}
