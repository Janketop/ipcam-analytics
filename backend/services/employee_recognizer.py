"""Распознавание сотрудников на основе эмбеддингов лиц."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Sequence, Union

import numpy as np
try:  # pragma: no cover - основное использование с настоящим sklearn
    from sklearn.neighbors import KDTree  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - минимальный fallback для тестов
    from backend.utils.kdtree_stub import KDTree
from sqlalchemy import select

from backend.core.config import settings
from backend.core.database import SessionFactory
from backend.core.logger import logger
from backend.models import Employee, FaceSample
from backend.services.face_embeddings import (
    FaceEmbeddingResult,
    compute_face_embedding_for_bbox,
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
        model_name = (encoding_model or settings.face_recognition_model or "small").strip()
        self.encoding_model = model_name or "small"
        self.detection_model = "hog"
        self.num_jitters = 1
        self.padding = 0.15

        self._tree: Optional[KDTree] = None
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
        return bool(self._tree and self._employee_ids)

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

        for blob, dim, model, employee_id, name in rows:
            if blob is None or dim is None or employee_id is None:
                skipped += 1
                continue

            model_name = (model or self.encoding_model).strip() or self.encoding_model
            if model_name != self.encoding_model:
                logger.debug(
                    "Пропускаю эмбеддинг сотрудника %s: модель %s != %s",
                    name,
                    model_name,
                    self.encoding_model,
                )
                skipped += 1
                continue

            embedding = FaceEmbeddingResult.from_bytes(
                blob, dim=dim, model=model_name
            )
            if embedding is None:
                skipped += 1
                continue

            vector = np.asarray(embedding.vector, dtype=np.float32).reshape(-1)
            if self._embedding_dim is not None and vector.size != self._embedding_dim:
                logger.debug(
                    "Размер эмбеддинга сотрудника %s (%s) не совпадает с %s",
                    name,
                    vector.size,
                    self._embedding_dim,
                )
                skipped += 1
                continue

            vectors.append(vector)
            employee_ids.append(int(employee_id))
            employee_names.append(name)

        if not vectors:
            if skipped:
                logger.info("Не удалось загрузить эмбеддинги сотрудников (%s пропущено)", skipped)
            self._tree = None
            self._embeddings = None
            self._employee_ids = []
            self._employee_names = []
            self._embedding_dim = None
            self._local_version = global_version
            return

        embedding_matrix = np.vstack(vectors).astype(np.float32, copy=False)
        self._tree = KDTree(embedding_matrix, metric="euclidean")
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

    def identify(
        self,
        embedding: Union[FaceEmbeddingResult, np.ndarray, Sequence[float]],
    ) -> Optional[RecognizedEmployee]:
        """Находит ближайшего сотрудника по эмбеддингу."""

        self.refresh_cache()

        if self._tree is None or self._embedding_dim is None or not self._employee_ids:
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

        distance, index = self._tree.query(vector.reshape(1, -1), k=1, return_distance=True)
        best_distance = float(distance[0][0])
        if best_distance > self.threshold:
            return None

        best_idx = int(index[0][0])
        employee_id = self._employee_ids[best_idx]
        employee_name = self._employee_names[best_idx]
        return RecognizedEmployee(
            employee_id=employee_id,
            employee_name=employee_name,
            distance=best_distance,
        )


__all__ = ["EmployeeRecognizer", "RecognizedEmployee"]
