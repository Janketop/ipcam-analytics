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


def _split_env_list(raw: str) -> List[str]:
    """Разбивает строку окружения в более гибком формате.

    Поддерживаются разделители запятая, точка с запятой и перевод строки.
    Это позволяет описывать списки как в одну строку, так и в виде
    «один элемент на строку», что часто встречается в `.env`-файлах.
    Пустые элементы и лишние пробелы автоматически отбрасываются.
    """

    if not raw:
        return []

    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


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
    ingest_status_interval: float = Field(5.0, ge=0.5)
    ingest_status_stale_threshold: float = Field(30.0, ge=1.0)
    idle_alert_time: int = Field(300, ge=10)
    snapshot_focus_buffer_size: int = Field(5, ge=1)

    face_blur: bool = Field(False)
    visualize: bool = Field(False)

    face_recognition_threshold: float = Field(0.6, ge=0.0)
    face_recognition_model: str = Field("facenet_vggface2")
    face_recognition_presence_cooldown: float = Field(15.0, ge=0.0)

    retention_days: int = Field(7, ge=0)
    retention_cleanup_interval_hours: float = Field(6.0, ge=0.0)
    face_sample_unverified_retention_days: int = Field(7, ge=0)

    frontend_origins: str = Field("")
    frontend_url: str = Field("")
    frontend_origin_regex: str = Field(r"https?://.*")

    yolo_device: str = Field("auto")
    cuda_visible_devices: Optional[str] = Field(None)
    yolo_det_model: str = Field("yolov8n.pt")
    yolo_pose_model: str = Field("yolov8n-pose.pt")
    yolo_face_model: str = Field("weights/yolo11n.pt")
    yolo_face_model_url: Optional[str] = Field(
        "https://github.com/ultralytics/assets/releases/latest/download/yolo11n.pt"
    )
    yolo_image_size: int = Field(640, ge=32)
    yolo_face_conf: float = Field(0.35, ge=0.05)

    face_training_dataset_root: str = Field("dataset/widerface")
    face_training_skip_download: bool = Field(False)
    face_training_epochs: int = Field(50, ge=1)
    face_training_batch: int = Field(32, ge=1)
    face_training_imgsz: int = Field(640, ge=32)
    face_training_device: Optional[str] = Field(None)
    face_training_project_dir: str = Field("runs/face")
    face_training_run_name: str = Field("yolo11n-widerface")
    face_training_base_weights: str = Field("yolo11n.pt")
    face_training_output_weights: str = Field("backend/weights/yolo11n-face.pt")

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
    def yolo_face_model_path(self) -> Path:
        """Возвращает абсолютный путь до весов модели распознавания лиц."""

        raw_path = Path(self.yolo_face_model)
        if raw_path.is_absolute():
            return raw_path
        return (_BACKEND_DIR / raw_path).resolve()

    def resolve_project_path(self, raw_path: str | Path) -> Path:
        """Преобразует относительный путь в абсолютный относительно корня проекта."""

        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (_PROJECT_ROOT / path).resolve()

    @property
    def face_training_dataset_root_path(self) -> Path:
        return self.resolve_project_path(self.face_training_dataset_root)

    @property
    def face_training_project_dir_path(self) -> Path:
        return self.resolve_project_path(self.face_training_project_dir)

    @property
    def face_training_output_weights_path(self) -> Path:
        return self.resolve_project_path(self.face_training_output_weights)

    @property
    def face_training_base_weights_path(self) -> Path:
        return self.resolve_project_path(self.face_training_base_weights)

    @property
    def cors_allow_origin_list(self) -> List[str]:
        """Возвращает отсортированный список Origin для CORS."""

        return sorted(self.cors_allow_origins)

    @property
    def cors_allow_origins(self) -> Set[str]:
        """Возвращает итоговый набор разрешённых Origin для CORS."""

        raw = self.frontend_origins or self.frontend_url or ""
        origins = set(_split_env_list(raw))
        origins.update({"http://localhost:3000", "http://127.0.0.1:3000"})
        return origins

    def iter_rtsp_sources(self) -> List[Tuple[str, str]]:
        """Парсит RTSP-строки вида 'name|url' в список (name, url).

        Если имя не указано, генерируется стабильное имя вида ``cam1``,
        ``cam2`` и т.д., чтобы оно не менялось между перезапусками сервиса.
        Также исключаются пустые элементы и дублирование случайных пробелов,
        поддерживаются разные разделители (запятая, точка с запятой, перенос
        строки), чтобы конфигурация была читабельной даже для длинных списков.
        """

        if not self.rtsp_sources.strip():
            return []

        entries: List[Tuple[str, str]] = []
        used_names: Set[str] = set()
        auto_index = 1

        def next_auto_name() -> str:
            nonlocal auto_index
            while True:
                candidate = f"cam{auto_index}"
                auto_index += 1
                if candidate not in used_names:
                    return candidate

        for chunk in _split_env_list(self.rtsp_sources):
            name: Optional[str]
            url: str
            if "|" in chunk:
                name_part, url_part = chunk.split("|", 1)
                name = name_part.strip() or None
                url = url_part.strip()
            else:
                name = None
                url = chunk

            if not url:
                continue

            if name is None:
                name = next_auto_name()
            else:
                # Если имя уже занято, аккуратно добавляем суффикс.
                original = name
                suffix = 2
                while name in used_names:
                    name = f"{original}_{suffix}"
                    suffix += 1

            used_names.add(name)
            entries.append((name, url))

        return entries


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Фабрика с кэшированием для singleton-настроек."""

    return Settings()


settings: Settings = get_settings()

__all__ = ["Settings", "settings", "get_settings"]
