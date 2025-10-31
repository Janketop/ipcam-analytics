"""Утилиты для загрузки весов ArcFace в формате ONNX."""
from __future__ import annotations

import shutil
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

from backend.core.logger import logger

# Весовые файлы InsightFace достаточно крупные, поэтому перечисляем несколько
# зеркал/вариантов, чтобы увеличить шанс успешного скачивания.
_DEFAULT_ARCFACE_SOURCES: tuple[str, ...] = (
    # Полная сборка Buffalo-L (включает w600k_r50.onnx ~174 МБ)
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    # Более компактный набор моделей (также содержит w600k_r50.onnx)
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
    # Альтернативный набор с глитн 360К (glint360k_r100.onnx)
    "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
    # Зеркало на HuggingFace (может переименовываться, поэтому оставляем последним)
    "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
)

_PREFERRED_MODEL_NAMES: tuple[str, ...] = (
    "w600k_r50.onnx",
    "glint360k_r100.onnx",
)


def _looks_like_arcface_model(path: Path) -> bool:
    """Проверяет, что файл похож на валидную ONNX-модель ArcFace."""

    if not path.exists():
        return False

    size = path.stat().st_size
    if size < 1024:  # слишком маленький файл — точно заглушка
        return False

    try:
        with path.open("rb") as handle:
            header = handle.read(4)
    except OSError:
        return False

    return header == b"ONNX"


def validate_arcface_model(path: Path) -> tuple[bool, str]:
    """Возвращает (валидность, описание проблемы) для ONNX-файла."""

    if not path.exists():
        return False, "не найден"

    size = path.stat().st_size
    if size < 1024:
        return False, f"слишком маленький ({size} байт)"

    try:
        with path.open("rb") as handle:
            header = handle.read(4)
    except OSError as exc:
        return False, f"не удалось прочитать файл: {exc}"

    if header != b"ONNX":
        return False, "не похож на ONNX (ожидался заголовок 'ONNX')"

    return True, ""


def _prepare_local_candidate(candidate: str | Path, temp_dir: Path) -> Optional[Path]:
    """Возвращает путь до локального файла с весами для дальнейшей обработки."""

    if isinstance(candidate, Path):
        return candidate if candidate.exists() else None

    parsed = urlparse(candidate)
    if parsed.scheme in ("", "file"):
        local_path = Path(parsed.path)
        return local_path if local_path.exists() else None

    # HTTP/HTTPS — скачиваем во временный файл
    filename = Path(parsed.path).name or "arcface.onnx"
    suffix = Path(filename).suffix or ""
    temp_path = temp_dir / f"download{suffix or '.bin'}"

    request = Request(candidate, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with closing(urlopen(request, timeout=60)) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:  # pragma: no cover - зависит от сети
        logger.warning("Не удалось скачать веса ArcFace из %s: %s", candidate, exc)
        return None

    return temp_path


def _select_member(names: Sequence[str]) -> Optional[str]:
    """Выбирает подходящий .onnx файл внутри архива."""

    normalized = [name for name in names if not name.endswith("/")]

    for preferred in _PREFERRED_MODEL_NAMES:
        for name in normalized:
            if name.lower().endswith(preferred):
                return name

    for name in normalized:
        if name.lower().endswith(".onnx"):
            return name

    return None


def _extract_from_zip(source: Path, destination: Path) -> bool:
    """Извлекает ONNX-файл из ZIP-архива InsightFace."""

    try:
        with ZipFile(source) as archive:
            member = _select_member(archive.namelist())
            if member is None:
                raise RuntimeError("в архиве отсутствует .onnx файл")

            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as exc:
        logger.warning("Не удалось распаковать ArcFace из %s: %s", source, exc)
        return False

    valid, reason = validate_arcface_model(destination)
    if not valid:
        logger.warning(
            "Извлечённый файл ArcFace из %s невалиден: %s", source, reason
        )
        return False

    logger.info(
        "ArcFace веса извлечены из архива %s -> %s (%.1f МБ)",
        source,
        destination,
        destination.stat().st_size / (1024 * 1024),
    )
    return True


def ensure_arcface_weights(
    destination: Path,
    *,
    sources: Optional[Iterable[str | Path]] = None,
) -> bool:
    """Гарантирует наличие валидных ONNX-весов ArcFace по указанному пути."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    valid, _ = validate_arcface_model(destination)
    if valid:
        return True

    candidates = list(sources) if sources is not None else list(_DEFAULT_ARCFACE_SOURCES)
    if not candidates:
        return False

    temp_dir = Path(tempfile.mkdtemp(prefix="arcface_dl_"))
    try:
        for candidate in candidates:
            local = _prepare_local_candidate(candidate, temp_dir)
            if local is None:
                continue

            try:
                suffix = local.suffix.lower()
            except ValueError:
                suffix = ""

            if suffix == ".zip":
                if _extract_from_zip(local, destination):
                    return True
                continue

            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, destination)
            except Exception as exc:
                logger.warning("Не удалось скопировать ArcFace модель из %s: %s", local, exc)
                continue

            valid, reason = validate_arcface_model(destination)
            if valid:
                logger.info(
                    "ArcFace веса подготовлены из %s -> %s (%.1f МБ)",
                    local,
                    destination,
                    destination.stat().st_size / (1024 * 1024),
                )
                return True

            logger.warning(
                "Файл ArcFace из %s некорректен: %s", local, reason
            )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return False


__all__ = [
    "ensure_arcface_weights",
    "validate_arcface_model",
]
