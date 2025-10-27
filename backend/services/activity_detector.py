"""Детектор активности сотрудников на основе позовых ключевых точек."""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np

class ActivityDetector:
    """Простая эвристика для определения активности по движению головы и рук."""

    def __init__(
        self,
        *,
        idle_threshold: float = 60.0,
        away_threshold: float = 120.0,
        movement_threshold: float = 0.015,
    ) -> None:
        self.idle_threshold = float(idle_threshold)
        self.away_threshold = float(away_threshold)
        self.movement_threshold = float(movement_threshold)

        self.prev_keypoints: Dict[str, np.ndarray] = {}
        self.last_movement_at: Dict[str, datetime] = {}
        self.last_seen_at: Dict[str, datetime] = {}
        self.states: Dict[str, str] = {}

    @staticmethod
    def _ensure_utc(ts: Optional[datetime]) -> datetime:
        if ts is None:
            return datetime.now(timezone.utc)
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _head_angle(self, keypoints: np.ndarray) -> Optional[float]:
        if keypoints.shape[0] < 3:
            return None
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)

    def _movement_metrics(
        self,
        keypoints: np.ndarray,
        prev_keypoints: Optional[np.ndarray],
    ) -> tuple[float, float, float]:
        if keypoints.size == 0:
            return 0.0, 0.0, 0.0
        if prev_keypoints is None or prev_keypoints.shape != keypoints.shape:
            return 0.0, 0.0, 0.0

        deltas = keypoints - prev_keypoints
        distances = np.linalg.norm(deltas, axis=1)

        head_indices = [idx for idx in (0, 1, 2, 3, 4) if idx < len(distances)]
        hand_indices = [idx for idx in (7, 8, 9, 10) if idx < len(distances)]

        head_movement = float(distances[head_indices].sum()) if head_indices else 0.0
        hand_movement = float(distances[hand_indices].sum()) if hand_indices else 0.0

        min_xy = np.nanmin(keypoints, axis=0)
        max_xy = np.nanmax(keypoints, axis=0)
        scale = float(np.linalg.norm(max_xy - min_xy))
        if scale <= 1e-3:
            scale = 1.0
        movement_score = (head_movement + hand_movement) / scale
        return head_movement, hand_movement, movement_score

    def update(
        self,
        people: Iterable[Dict[str, object]],
        *,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, object]]:
        """Обновляет состояние по списку людей и возвращает актуальные данные."""

        now_utc = self._ensure_utc(now)
        seen_ids: set[str] = set()
        updates: List[Dict[str, object]] = []

        for person in people:
            keypoints = self._coerce_keypoints(person.get("keypoints"))

            person_id_raw = person.get("id")
            if person_id_raw is None:
                digest = hashlib.sha1(keypoints.tobytes()).hexdigest()[:16]
                person_id = digest
            else:
                person_id = str(person_id_raw)
            seen_ids.add(person_id)

            confidence = float(person.get("confidence") or 0.0)

            prev = self.prev_keypoints.get(person_id)
            head_movement, hand_movement, movement_score = self._movement_metrics(keypoints, prev)

            if person_id not in self.last_movement_at:
                self.last_movement_at[person_id] = now_utc
            if movement_score >= self.movement_threshold:
                self.last_movement_at[person_id] = now_utc

            self.last_seen_at[person_id] = now_utc
            self.prev_keypoints[person_id] = keypoints.copy()

            idle_seconds = (now_utc - self.last_movement_at[person_id]).total_seconds()
            away_seconds = 0.0

            prev_state = self.states.get(person_id)
            new_state = "WORKING"
            if idle_seconds >= self.idle_threshold:
                new_state = "NOT_WORKING"
            if prev_state == "AWAY" and new_state == "WORKING":
                idle_seconds = 0.0

            head_angle = self._head_angle(keypoints)

            changed = new_state != prev_state
            self.states[person_id] = new_state

            updates.append(
                {
                    "id": person_id,
                    "state": new_state,
                    "head_movement": head_movement,
                    "hand_movement": hand_movement,
                    "movement_score": movement_score,
                    "head_angle": head_angle,
                    "idle_seconds": idle_seconds,
                    "away_seconds": away_seconds,
                    "confidence": confidence,
                    "changed": changed,
                }
            )

        for person_id, last_seen in list(self.last_seen_at.items()):
            if person_id in seen_ids:
                continue
            away_seconds = (now_utc - last_seen).total_seconds()
            idle_seconds = away_seconds
            prev_state = self.states.get(person_id)
            new_state = prev_state or "WORKING"
            if away_seconds >= self.away_threshold:
                new_state = "AWAY"

            changed = new_state != prev_state
            if changed:
                self.states[person_id] = new_state

            updates.append(
                {
                    "id": person_id,
                    "state": new_state,
                    "head_movement": 0.0,
                    "hand_movement": 0.0,
                    "movement_score": 0.0,
                    "head_angle": None,
                    "idle_seconds": idle_seconds,
                    "away_seconds": away_seconds,
                    "confidence": 0.0,
                    "changed": changed,
                }
            )

        return updates

    @staticmethod
    def _coerce_keypoints(raw_keypoints: object) -> np.ndarray:
        """Преобразует входные ключевые точки в матрицу Nx2 float32."""

        if raw_keypoints is None:
            return np.zeros((0, 2), dtype=np.float32)

        keypoints = np.asarray(raw_keypoints, dtype=np.float32)
        if keypoints.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if keypoints.ndim == 1:
            if keypoints.size < 2:
                return np.zeros((0, 2), dtype=np.float32)
            keypoints = keypoints.reshape(-1, 2)
        else:
            # Схлопываем лишние измерения и оставляем только координаты X/Y.
            keypoints = keypoints.reshape(-1, keypoints.shape[-1])
            if keypoints.shape[1] < 2:
                return np.zeros((0, 2), dtype=np.float32)
            keypoints = keypoints[:, :2]

        return keypoints
