"""Единая настройка логирования для приложения."""

import logging
import logging.config
from pathlib import Path

from backend.core.paths import LOG_DIR


LOG_FILE: Path = LOG_DIR / "app.log"

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "format": "%(asctime)s [%(levelname)s] %(client_addr)s - \"%(request_line)s\" %(status_code)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": str(LOG_FILE),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 10,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        "uvicorn.access": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
            "formatter": "access",
        },
        "ipcam": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
    },
    "root": {"handlers": ["console", "file"], "level": "INFO"},
}


def configure_logging() -> None:
    """Применяет конфигурацию логгера через dictConfig."""

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)


configure_logging()

logger = logging.getLogger("ipcam")

