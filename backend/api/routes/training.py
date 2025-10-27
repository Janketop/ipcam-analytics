"""Маршруты, связанные с самообучением модели."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from backend.services.training import SelfTrainingService

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
