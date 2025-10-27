"""Инициализация ORM-моделей проекта."""
from .base import Base
from .camera import Camera
from .employee import Employee
from .event import Event
from .face_sample import FaceSample

__all__ = ["Base", "Camera", "Employee", "Event", "FaceSample"]
