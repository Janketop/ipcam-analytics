"""Централизованная конфигурация приложения и загрузка переменных окружения."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CONFIG_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_ENV_FILES: Tuple[str, ...] = (
    str(_PROJECT_ROOT / ".env"),
    str(_BACKEND_DIR / ".env"),
)


class Settings(BaseSettings):
    """Глобальные настройки сервиса, считываемые из .env и окружения."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_title: str = Field("IPCam Analytics (RU)")

    postgres_user: str = Field("ipcam")
    postgres_password: str = Field("ipcam")
    postgres_db: str = Field("ipcam")
    postgres_host: str = Field("db")
    postgres_port: int = Field(5432)

    rtsp_sources: str = Field("")
    rtsp_reconnect_delay: float = Field(5.0, ge=0.5)
    rtsp_max_failed_reads: int = Field(25, ge=1)
    ingest_fps_skip: int = Field(2, ge=1)
    ingest_flush_timeout: float = Field(0.2, ge=0.0)

    face_blur: bool = Field(False)
    visualize: bool = Field(False)

    retention_days: int = Field(7, ge=0)
    retention_cleanup_interval_hours: float = Field(6.0, ge=0.0)

    frontend_origins: str = Field("")
    frontend_url: str = Field("")
    frontend_origin_regex: str = Field(r"https?://.*")

    yolo_device: str = Field("auto")
    cuda_visible_devices: Optional[str] = Field(None)
    yolo_det_model: str = Field("yolov8n.pt")
    yolo_pose_model: str = Field("yolov8n-pose.pt")
    yolo_image_size: int = Field(640, ge=32)

    phone_det_conf: float = Field(0.3, ge=0.05)
    pose_det_conf: float = Field(0.3, ge=0.05)
    phone_score_threshold: float = Field(0.6, ge=0.1)
    phone_hand_dist_ratio: float = Field(0.35, ge=0.05)
    phone_head_dist_ratio: float = Field(0.45, ge=0.05)
    pose_only_score_threshold: float = Field(0.55, ge=0.1)
    pose_only_head_ratio: float = Field(0.5, ge=0.05)
    pose_wrists_dist_ratio: float = Field(0.25, ge=0.05)
    pose_tilt_threshold: float = Field(0.22, ge=0.05)
    phone_score_smoothing: int = Field(5, ge=1)

    car_det_conf: float = Field(0.35, ge=0.05)
    car_moving_fg_ratio: float = Field(0.05, ge=0.0)
    car_event_cooldown: float = Field(8.0, ge=1.0)

    plate_ocr_langs: str = Field("ru,en")

    @property
    def postgres_dsn(self) -> str:
        """Строит DSN для подключения к PostgreSQL."""

        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        host = self.postgres_host
        port = self.postgres_port
        database = self.postgres_db
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    @property
    def cors_allow_origin_list(self) -> List[str]:
        """Возвращает отсортированный список Origin для CORS."""

        return sorted(self.cors_allow_origins)

    @property
    def cors_allow_origins(self) -> Set[str]:
        """Возвращает итоговый набор разрешённых Origin для CORS."""

        raw = self.frontend_origins or self.frontend_url or ""
        origins = {origin.strip() for origin in raw.split(",") if origin.strip()}
        origins.update({"http://localhost:3000", "http://127.0.0.1:3000"})
        return origins

    def iter_rtsp_sources(self) -> List[Tuple[str, str]]:
        """Парсит RTSP-строки вида 'name|url' в список (name, url)."""

        if not self.rtsp_sources.strip():
            return []

        entries: List[Tuple[str, str]] = []
        for item in self.rtsp_sources.split(","):
            chunk = item.strip()
            if not chunk:
                continue
            if "|" in chunk:
                name, url = chunk.split("|", 1)
            else:
                name = f"cam_{abs(hash(chunk)) % 10000}"
                url = chunk
            entries.append((name.strip(), url.strip()))
        return entries


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Фабрика с кэшированием для singleton-настроек."""

    return Settings()


settings: Settings = get_settings()

__all__ = ["Settings", "settings", "get_settings"]
