"""Простейшая реализация KDTree для тестового окружения."""
from __future__ import annotations

import numpy as np


class KDTree:  # pragma: no cover - используется только в окружениях без sklearn
    def __init__(self, data, metric: str = "euclidean") -> None:
        self._data = np.asarray(data, dtype=np.float32)

    def query(self, vector, k: int = 1, return_distance: bool = True):
        point = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        diff = self._data - point
        distances = np.linalg.norm(diff, axis=1)
        order = np.argsort(distances)[:k]
        distances = distances[order].reshape(1, -1)
        indices = np.asarray(order, dtype=np.int64).reshape(1, -1)
        if return_distance:
            return distances, indices
        return indices


__all__ = ["KDTree"]
