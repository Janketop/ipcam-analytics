"""Инициализация ORM-моделей проекта."""
from .base import Base
from .camera import Camera
from .event import Event

__all__ = ["Base", "Camera", "Event"]
