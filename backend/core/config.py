"""Централизованная конфигурация приложения и загрузка переменных окружения."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Set
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Глобальные настройки сервиса, считываемые из .env и окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_title: str = Field("IPCam Analytics (RU)", alias="APP_TITLE")

    postgres_user: str = Field("ipcam", alias="POSTGRES_USER")
    postgres_password: str = Field("ipcam", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field("ipcam", alias="POSTGRES_DB")
    postgres_host: str = Field("db", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")

    rtsp_sources: str = Field("", alias="RTSP_SOURCES")
    rtsp_reconnect_delay: float = Field(5.0, alias="RTSP_RECONNECT_DELAY", ge=0.5)
    rtsp_max_failed_reads: int = Field(25, alias="RTSP_MAX_FAILED_READS", ge=1)
    ingest_fps_skip: int = Field(2, alias="INGEST_FPS_SKIP", ge=1)
    ingest_flush_timeout: float = Field(0.2, alias="INGEST_FLUSH_TIMEOUT", ge=0.0)

    face_blur: bool = Field(False, alias="FACE_BLUR")
    visualize: bool = Field(False, alias="VISUALIZE")

    retention_days: int = Field(7, alias="RETENTION_DAYS", ge=0)
    retention_cleanup_interval_hours: float = Field(
        6.0, alias="RETENTION_CLEANUP_INTERVAL_HOURS", ge=0.0
    )

    frontend_origins: str = Field("", alias="FRONTEND_ORIGINS")
    frontend_url: str = Field("", alias="FRONTEND_URL")
    frontend_origin_regex: str = Field(r"https?://.*", alias="FRONTEND_ORIGIN_REGEX")

    yolo_device: str = Field("auto", alias="YOLO_DEVICE")
    cuda_visible_devices: Optional[str] = Field(None, alias="CUDA_VISIBLE_DEVICES")
    yolo_det_model: str = Field("yolov8n.pt", alias="YOLO_DET_MODEL")
    yolo_pose_model: str = Field("yolov8n-pose.pt", alias="YOLO_POSE_MODEL")
    yolo_image_size: int = Field(640, alias="YOLO_IMAGE_SIZE", ge=32)

    phone_det_conf: float = Field(0.3, alias="PHONE_DET_CONF", ge=0.05)
    pose_det_conf: float = Field(0.3, alias="POSE_DET_CONF", ge=0.05)
    phone_score_threshold: float = Field(0.6, alias="PHONE_SCORE_THRESHOLD", ge=0.1)
    phone_hand_dist_ratio: float = Field(0.35, alias="PHONE_HAND_DIST_RATIO", ge=0.05)
    phone_head_dist_ratio: float = Field(0.45, alias="PHONE_HEAD_DIST_RATIO", ge=0.05)
    pose_only_score_threshold: float = Field(
        0.55, alias="POSE_ONLY_SCORE_THRESHOLD", ge=0.1
    )
    pose_only_head_ratio: float = Field(0.5, alias="POSE_ONLY_HEAD_RATIO", ge=0.05)
    pose_wrists_dist_ratio: float = Field(0.25, alias="POSE_WRISTS_DIST_RATIO", ge=0.05)
    pose_tilt_threshold: float = Field(0.22, alias="POSE_TILT_THRESHOLD", ge=0.05)
    phone_score_smoothing: int = Field(5, alias="PHONE_SCORE_SMOOTHING", ge=1)

    car_det_conf: float = Field(0.35, alias="CAR_DET_CONF", ge=0.05)
    car_moving_fg_ratio: float = Field(0.05, alias="CAR_MOVING_FG_RATIO", ge=0.0)
    car_event_cooldown: float = Field(8.0, alias="CAR_EVENT_COOLDOWN", ge=1.0)

    plate_ocr_langs: str = Field("ru,en", alias="PLATE_OCR_LANGS")

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
    def cors_allow_origins(self) -> Set[str]:
        """Возвращает итоговый набор разрешённых Origin для CORS."""

        raw = self.frontend_origins or self.frontend_url or ""
        origins = {origin.strip() for origin in raw.split(",") if origin.strip()}
        origins.update({"http://localhost:3000", "http://127.0.0.1:3000"})
        return origins


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Фабрика с кэшированием для singleton-настроек."""

    return Settings()


settings: Settings = get_settings()

__all__ = ["Settings", "settings", "get_settings"]
