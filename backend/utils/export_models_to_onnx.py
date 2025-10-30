"""Утилита для конвертации YOLO-весов в формат ONNX."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional


def _setup_logger() -> logging.Logger:
    """Возвращает логгер, пытаясь использовать общую конфигурацию приложения."""

    try:
        from backend.core.logger import logger as app_logger
    except Exception as exc:  # pragma: no cover - зависит от окружения
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fallback_logger = logging.getLogger("ipcam.export")
        fallback_logger.warning(
            "Не удалось инициализировать стандартное логирование: %s. "
            "Используется запасная конфигурация, пишущая только в консоль.",
            exc,
        )
        return fallback_logger

    return app_logger


logger = _setup_logger()


class ExportError(RuntimeError):
    """Ошибка экспорта YOLO-весов."""


def _parse_imgsz(values: Optional[Iterable[int]]) -> int | tuple[int, int] | None:
    """Преобразует аргумент --imgsz в формат, понятный Ultralytics."""

    if values is None:
        return None

    values = tuple(values)
    if not values:
        return None
    if len(values) == 1:
        return int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])

    raise ExportError("--imgsz принимает либо одно, либо два целых значения")


def _export_single(
    label: str,
    weights_path: Path,
    output_path: Path | None,
    *,
    device: str,
    imgsz: int | tuple[int, int] | None,
    opset: int | None,
    simplify: bool | None,
    dynamic: bool,
) -> Path:
    """Запускает экспорт одной модели и возвращает путь к итоговому ONNX-файлу."""

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise ExportError(
            "Библиотека ultralytics недоступна. Установите зависимость перед экспортом"
        ) from exc

    if not weights_path.exists():
        raise ExportError(f"Файл весов {weights_path} не найден")

    logger.info("Экспорт %s из %s", label, weights_path)
    model = YOLO(str(weights_path))

    export_kwargs = {
        "format": "onnx",
        "device": device,
        "dynamic": dynamic,
    }
    if imgsz is not None:
        export_kwargs["imgsz"] = imgsz
    if opset is not None:
        export_kwargs["opset"] = opset
    if simplify is not None:
        export_kwargs["simplify"] = simplify

    try:
        exported = Path(model.export(**export_kwargs))
    except Exception as exc:  # pragma: no cover - зависит от ultralytics
        raise ExportError(f"Не удалось экспортировать {weights_path}: {exc}") from exc

    if output_path is None:
        logger.info("Экспорт завершён: %s", exported)
        return exported

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), output_path)
        logger.info("Файл перемещён в %s", output_path)
    else:
        logger.info("Файл уже находится в целевом пути %s", output_path)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Конвертация YOLO-весов (.pt) в формат ONNX.\n"
            "Если передан только путь к весам, результат сохраняется рядом "
            "с исходником. Для хранения в стандартной директории "
            "используйте флаг --output-dir."
        )
    )
    parser.add_argument("weights", type=Path, nargs="+", help="Пути к .pt-файлам YOLO")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Каталог для сохранения ONNX-файлов. Если не указан, Ultralytics создаст"
            " структуру runs/export/..."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство для экспорта (cpu, cuda:0 и т.д.)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Размер входного изображения (один размер или пара высота ширина)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Целевая версия opset для ONNX",
    )
    simplify_group = parser.add_mutually_exclusive_group()
    simplify_group.add_argument(
        "--simplify",
        dest="simplify",
        action="store_true",
        help="Запустить постобработку simplify=True",
    )
    simplify_group.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="Отключить упрощение графа",
    )
    parser.set_defaults(simplify=None)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Экспортировать с динамическим размером входа",
    )
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI тонкая логика
    """Точка входа CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        imgsz = _parse_imgsz(args.imgsz)
    except ExportError as exc:
        parser.error(str(exc))
        return 2

    errors = 0
    for weights_path in args.weights:
        output_path = None
        if args.output_dir is not None:
            output_path = args.output_dir / (weights_path.stem + ".onnx")
        try:
            result_path = _export_single(
                weights_path.stem,
                weights_path,
                output_path,
                device=args.device,
                imgsz=imgsz,
                opset=args.opset,
                simplify=args.simplify,
                dynamic=args.dynamic,
            )
            logger.info("ONNX-модель сохранена: %s", result_path)
        except ExportError as exc:
            errors += 1
            logger.error("%s", exc)

    return 0 if errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
