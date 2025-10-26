"""Общие зависимости FastAPI."""
from __future__ import annotations

from fastapi import Request


def get_engine(request: Request):
    return request.app.state.engine


def get_ingest_manager(request: Request):
    return request.app.state.ingest_manager


def get_cleanup_state(request: Request):
    return request.app.state.cleanup_state


def get_broadcaster(request: Request):
    return request.app.state.event_broadcaster
