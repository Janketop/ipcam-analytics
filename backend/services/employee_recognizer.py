"""Распознавание сотрудников на основе эмбеддингов лиц."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from sqlalchemy import select

from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.core.logger import logger
from backend.models import Employee, FaceSample
from backend.services.face_embeddings import (
    FaceEmbeddingResult,
    compute_face_embedding_for_bbox,
    compute_face_embedding_from_snapshot,
    get_embedding_metadata,
    normalize_encoding_model_name,
)

_CACHE_VERSION = 0
_CACHE_LOCK = Lock()


def _bump_cache_version() -> int:
    global _CACHE_VERSION
    with _CACHE_LOCK:
        _CACHE_VERSION += 1
        return _CACHE_VERSION


def _get_cache_version() -> int:
    return _CACHE_VERSION


@dataclass(slots=True)
class RecognizedEmployee:
    """Информация о распознанном сотруднике."""

    employee_id: int
    employee_name: str
    distance: float
    backend: str
    metric: str


class EmployeeRecognizer:
    """Сервис поиска сотрудника по эмбеддингу лица."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        threshold: Optional[float] = None,
        encoding_model: Optional[str] = None,
    ) -> None:
        self.session_factory = session_factory
        self.threshold = float(
            threshold if threshold is not None else settings.face_recognition_threshold
        )
        model_name = normalize_encoding_model_name(
            encoding_model or settings.face_recognition_model,
            allow_fallback=True,
        )
        self.encoding_model = model_name
        self._model_metadata = get_embedding_metadata(model_name)
        self.detection_model = "hog"
        self.num_jitters = 1
        self.padding = 0.15
        self._expected_embedding_dim = int(
            self._model_metadata.get("embedding_dim") or 0
        )

        logger.debug(
            "Инициализирован распознаватель лиц (модель=%s, embedding_dim=%s, провайдеры=%s)",
            self.encoding_model,
            self._model_metadata.get("embedding_dim"),
            ", ".join(self._model_metadata.get("providers", ()))
            if isinstance(self._model_metadata.get("providers"), (list, tuple))
            else self._model_metadata.get("providers"),
        )

        self._embeddings: Optional[np.ndarray] = None
        self._employee_ids: list[int] = []
        self._employee_names: list[str] = []
        self._embedding_dim: Optional[int] = None
        self._local_version = -1

        self.refresh_cache(force=True)

    # --- Статические методы -------------------------------------------------
    @classmethod
    def notify_embeddings_updated(cls) -> None:
        """Сообщает всем экземплярам, что эмбеддинги в БД изменились."""

        new_version = _bump_cache_version()
        logger.debug("Версия кэша эмбеддингов обновлена: %s", new_version)

    # --- Свойства -----------------------------------------------------------
    @property
    def sample_count(self) -> int:
        return len(self._employee_ids)

    @property
    def employee_count(self) -> int:
        return len({emp_id for emp_id in self._employee_ids})

    @property
    def has_samples(self) -> bool:
        return bool(self._embeddings is not None and self._employee_ids)

    # --- Публичные методы ---------------------------------------------------
    def refresh_cache(self, *, force: bool = False) -> None:
        """Перезагружает эмбеддинги сотрудников из базы данных."""

        global_version = _get_cache_version()
        if not force and self._local_version == global_version:
            return

        logger.debug(
            "Перезагрузка эмбеддингов сотрудников (force=%s, version=%s)",
            force,
            global_version,
        )

        with self.session_factory() as session:
            stmt = (
                select(
                    FaceSample.id,
                    FaceSample.snapshot_url,
                    FaceSample.embedding,
                    FaceSample.embedding_dim,
                    FaceSample.embedding_model,
                    FaceSample.employee_id,
                    Employee.name,
                )
                .join(Employee, FaceSample.employee_id == Employee.id)
                .where(
                    FaceSample.status == FaceSample.STATUS_EMPLOYEE,
                    FaceSample.embedding.is_not(None),
                    FaceSample.embedding_dim.is_not(None),
                    FaceSample.employee_id.is_not(None),
                )
            )
            rows = session.execute(stmt).all()

        vectors: list[np.ndarray] = []
        employee_ids: list[int] = []
        employee_names: list[str] = []
        skipped = 0

        updates: dict[int, FaceEmbeddingResult] = {}
        expected_dim = self._expected_embedding_dim

        for sample_id, snapshot_url, blob, dim, model, employee_id, name in rows:
            if blob is None or dim is None or employee_id is None:
                skipped += 1
                continue

            model_name = normalize_encoding_model_name(
                model or self.encoding_model,
                allow_fallback=False,
            )
            embedding = FaceEmbeddingResult.from_bytes(
                blob, dim=dim, model=model_name
            )
            vector = None
            rebuilt_for_sample: Optional[FaceEmbeddingResult] = None

            if model_name != self.encoding_model or embedding is None:
                rebuilt = self._rebuild_embedding(
                    sample_id,
                    snapshot_url,
                    name,
                )
                if rebuilt is None:
                    skipped += 1
                    continue
                embedding = rebuilt
                model_name = embedding.model
                rebuilt_for_sample = rebuilt

            if embedding is None:
                skipped += 1
                continue

            vector = np.asarray(embedding.vector, dtype=np.float32).reshape(-1)

            if expected_dim and vector.size != expected_dim:
                rebuilt = self._rebuild_embedding(
                    sample_id,
                    snapshot_url,
                    name,
                )
                if rebuilt is None:
                    skipped += 1
                    continue
                embedding = rebuilt
                vector = np.asarray(embedding.vector, dtype=np.float32).reshape(-1)
                rebuilt_for_sample = rebuilt

            if embedding.model != self.encoding_model:
                skipped += 1
                continue

            if self._embedding_dim is not None and vector.size != self._embedding_dim:
                logger.debug(
                    "Размер эмбеддинга сотрудника %s (%s) не совпадает с %s",
                    name,
                    vector.size,
                    self._embedding_dim,
                )
                skipped += 1
                continue

            if rebuilt_for_sample is not None:
                updates[sample_id] = rebuilt_for_sample

            vectors.append(vector)
            employee_ids.append(int(employee_id))
            employee_names.append(name)

        if updates:
            self._apply_embedding_updates(updates.items())
            EmployeeRecognizer.notify_embeddings_updated()
            global_version = _get_cache_version()

        if not vectors:
            if skipped:
                logger.info("Не удалось загрузить эмбеддинги сотрудников (%s пропущено)", skipped)
            self._embeddings = None
            self._employee_ids = []
            self._employee_names = []
            self._embedding_dim = None
            self._local_version = global_version
            return

        embedding_matrix = np.vstack(vectors).astype(np.float32, copy=False)
        self._embeddings = embedding_matrix
        self._employee_ids = employee_ids
        self._employee_names = employee_names
        self._embedding_dim = int(embedding_matrix.shape[1])
        self._local_version = global_version

        logger.info(
            "Загружено %d эмбеддингов сотрудников (%d уникальных)",
            len(employee_ids),
            self.employee_count,
        )
        if skipped:
            logger.debug("Пропущено %d эмбеддингов из-за ошибок", skipped)
        logger.debug(
            "Эмбеддинги подготовлены для бэкэнда numpy (размер=%s)",
            embedding_matrix.shape,
        )

    def compute_embedding(
        self,
        frame_bgr: np.ndarray,
        bbox: Sequence[float] | None,
    ) -> Optional[FaceEmbeddingResult]:
        """Вычисляет эмбеддинг лица на кадре."""

        return compute_face_embedding_for_bbox(
            frame_bgr,
            bbox,
            encoding_model=self.encoding_model,
            detection_model=self.detection_model,
            num_jitters=self.num_jitters,
            padding=self.padding,
        )

    def _rebuild_embedding(
        self,
        sample_id: int,
        snapshot_url: Optional[str],
        employee_name: str,
    ) -> Optional[FaceEmbeddingResult]:
        if not snapshot_url:
            logger.debug(
                "Не удалось пересчитать эмбеддинг сотрудника %s: отсутствует snapshot_url",
                employee_name,
            )
            return None

        logger.info(
            "Пересчитываю эмбеддинг сотрудника %s (sample_id=%s) для модели %s",
            employee_name,
            sample_id,
            self.encoding_model,
        )
        rebuilt = compute_face_embedding_from_snapshot(
            snapshot_url,
            encoding_model=self.encoding_model,
            detection_model=self.detection_model,
            num_jitters=self.num_jitters,
        )
        if rebuilt is None:
            logger.warning(
                "Не удалось пересчитать эмбеддинг сотрудника %s (sample_id=%s)",
                employee_name,
                sample_id,
            )
            return None

        if self._expected_embedding_dim and rebuilt.dimension != self._expected_embedding_dim:
            logger.warning(
                "Пересчитанный эмбеддинг сотрудника %s имеет размер %s, ожидалось %s",
                employee_name,
                rebuilt.dimension,
                self._expected_embedding_dim,
            )
            return None

        return rebuilt

    def _apply_embedding_updates(
        self, updates: Iterable[tuple[int, FaceEmbeddingResult]]
    ) -> None:
        if not updates:
            return

        with self.session_factory() as session:
            for sample_id, embedding in updates:
                sample = session.get(FaceSample, sample_id)
                if sample is None:
                    continue
                sample.set_embedding(
                    embedding.as_bytes(),
                    dim=embedding.dimension,
                    model=embedding.model,
                )
            session.commit()

    def identify(
        self,
        embedding: Union[FaceEmbeddingResult, np.ndarray, Sequence[float]],
    ) -> Optional[RecognizedEmployee]:
        """Находит ближайшего сотрудника по эмбеддингу."""

        self.refresh_cache()

        if self._embeddings is None or self._embedding_dim is None or not self._employee_ids:
            return None

        if isinstance(embedding, FaceEmbeddingResult):
            vector = embedding.vector
        else:
            vector = np.asarray(embedding, dtype=np.float32)

        vector = vector.reshape(-1).astype(np.float32, copy=False)
        if vector.size != self._embedding_dim:
            logger.debug(
                "Размер эмбеддинга (%s) не совпадает с ожидаемым (%s)",
                vector.size,
                self._embedding_dim,
            )
            return None

        metric = "euclidean"
        embeddings_matrix = self._embeddings
        diff = embeddings_matrix - vector[np.newaxis, :]
        distances = np.linalg.norm(diff, axis=1)
        if distances.size == 0:
            return None

        if not np.isfinite(distances).all():
            logger.debug("Получены нечисловые значения расстояний")
            return None

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        backend = "numpy"

        if best_distance > self.threshold:
            logger.debug(
                "Лучший кандидат превышает порог: distance=%.4f, threshold=%.4f, backend=%s",
                best_distance,
                self.threshold,
                backend,
            )
            return None

        employee_id = self._employee_ids[best_idx]
        employee_name = self._employee_names[best_idx]
        logger.debug(
            "Распознан сотрудник %s (id=%s, distance=%.4f, backend=%s)",
            employee_name,
            employee_id,
            best_distance,
            backend,
        )
        return RecognizedEmployee(
            employee_id=employee_id,
            employee_name=employee_name,
            distance=best_distance,
            backend=backend,
            metric=metric,
        )


__all__ = ["EmployeeRecognizer", "RecognizedEmployee"]
