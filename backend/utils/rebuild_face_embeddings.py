"""Пересчёт эмбеддингов лиц в базе данных."""
from __future__ import annotations

from datetime import datetime, timezone

from backend.core.config import settings
from backend.core.database import SessionFactory, get_session_factory
from backend.core.logger import logger
from backend.models import FaceSample
from backend.services.employee_recognizer import EmployeeRecognizer
from backend.services.face_embeddings import (
    compute_face_embedding_from_snapshot,
    get_embedding_metadata,
    normalize_encoding_model_name,
)


def rebuild_missing_face_embeddings(
    session_factory: SessionFactory,
    *,
    encoding_model: str | None = None,
) -> int:
    """Пересчитывает эмбеддинги для снимков сотрудников без данных."""

    target_model = normalize_encoding_model_name(
        encoding_model or settings.face_recognition_model,
        allow_fallback=True,
    )
    metadata = get_embedding_metadata(target_model)

    updated = 0
    skipped = 0

    with session_factory() as session:
        samples = (
            session.query(FaceSample)
            .filter(FaceSample.status == FaceSample.STATUS_EMPLOYEE)
            .filter(FaceSample.snapshot_url.is_not(None))
            .filter(
                (FaceSample.embedding.is_(None))
                | (FaceSample.embedding_dim.is_(None))
                | (FaceSample.embedding_model.is_(None))
                | (FaceSample.embedding_model != target_model)
            )
            .order_by(FaceSample.id)
            .all()
        )

        if not samples:
            logger.info(
                "Все эмбеддинги лиц уже вычислены (модель %s, dim=%s)",
                target_model,
                metadata["embedding_dim"],
                )
            return 0

        for sample in samples:
            result = compute_face_embedding_from_snapshot(
                sample.snapshot_url,
                encoding_model=target_model,
            )
            if result is None:
                skipped += 1
                logger.warning(
                    "Не удалось вычислить эмбеддинг лица для FaceSample #%s (%s)",
                    sample.id,
                    sample.snapshot_url,
                )
                continue

            sample.set_embedding(
                result.as_bytes(),
                dim=result.dimension,
                model=result.model,
            )
            sample.updated_at = datetime.now(timezone.utc)
            updated += 1

        if updated:
            session.commit()
        else:
            session.rollback()

    if updated:
        EmployeeRecognizer.notify_embeddings_updated()
        logger.info(
            "Пересчитано эмбеддингов лиц: %d (пропущено %d)",
            updated,
            skipped,
        )
    else:
        logger.info("Не удалось обновить ни одного эмбеддинга лиц (%d пропущено)", skipped)

    return updated


def main() -> int:  # pragma: no cover - точка входа для скрипта
    """CLI для пересчёта эмбеддингов лиц."""

    session_factory = get_session_factory()
    return rebuild_missing_face_embeddings(session_factory)


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    main()
