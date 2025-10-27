"""Сценарий самообучения модели детекции телефонов."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from backend.core.config import settings
from backend.core.logger import logger
from backend.core.paths import BACKEND_DIR, DATASET_PHONE_USAGE_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MODELS_DIR = BACKEND_DIR.parent / "models"
PREPARED_DATA_DIR = DATASET_PHONE_USAGE_DIR / "prepared"
DEFAULT_CLASS_NAME = "cell phone"
DEFAULT_EPOCHS = 10
TRAIN_SPLIT = 0.8


def _find_label(image_path: Path) -> Optional[Path]:
    """Возвращает путь к файлу разметки, если он существует."""

    candidates = [
        DATASET_PHONE_USAGE_DIR / "labels" / f"{image_path.stem}.txt",
        image_path.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_samples() -> List[Tuple[Path, Path]]:
    """Ищет размеченные изображения в каталоге датасета."""

    samples: List[Tuple[Path, Path]] = []
    skipped: List[Path] = []
    for item in DATASET_PHONE_USAGE_DIR.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = _find_label(item)
        if label is None:
            skipped.append(item)
            continue
        samples.append((item, label))

    if skipped:
        logger.warning(
            "Найдено %d снимков без файлов разметки — они пропущены при обучении", len(skipped)
        )
    logger.info("Собрано %d размеченных снимков для обучения", len(samples))
    return samples


def _copy_pairs(pairs: Iterable[Tuple[Path, Path]], images_dir: Path, labels_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)


def prepare_dataset(samples: Sequence[Tuple[Path, Path]]) -> Path:
    """Создаёт структуру каталогов YOLO и возвращает путь к data.yaml."""

    if PREPARED_DATA_DIR.exists():
        shutil.rmtree(PREPARED_DATA_DIR)
    (PREPARED_DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (PREPARED_DATA_DIR / "labels").mkdir(parents=True, exist_ok=True)

    shuffled = list(samples)
    random.shuffle(shuffled)

    if len(shuffled) == 1:
        train_pairs = shuffled
        val_pairs = shuffled
    else:
        split_idx = max(1, int(len(shuffled) * TRAIN_SPLIT))
        if split_idx >= len(shuffled):
            split_idx = len(shuffled) - 1
        train_pairs = shuffled[:split_idx]
        val_pairs = shuffled[split_idx:]

    _copy_pairs(train_pairs, PREPARED_DATA_DIR / "images" / "train", PREPARED_DATA_DIR / "labels" / "train")
    _copy_pairs(val_pairs, PREPARED_DATA_DIR / "images" / "val", PREPARED_DATA_DIR / "labels" / "val")

    data_yaml = PREPARED_DATA_DIR / "data.yaml"
    yaml_content = "\n".join(
        [
            f"path: {PREPARED_DATA_DIR.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            f"  0: {DEFAULT_CLASS_NAME}",
        ]
    )
    data_yaml.write_text(yaml_content + "\n", encoding="utf-8")
    logger.info("Датасет подготовлен в %s", PREPARED_DATA_DIR)
    return data_yaml


def _resolve_base_weights() -> str:
    """Определяет, какую модель использовать в качестве исходной."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    custom_weights = MODELS_DIR / "custom.pt"
    if custom_weights.exists():
        logger.info("Найдены предыдущие кастомные веса: %s", custom_weights)
        return str(custom_weights)

    candidate = Path(settings.yolo_det_model)
    if candidate.exists():
        logger.info("Использую локальные веса %s в качестве базы", candidate)
        return str(candidate)

    logger.info("Использую модель по умолчанию из настроек: %s", settings.yolo_det_model)
    return settings.yolo_det_model


def train_model(data_yaml: Path) -> Path:
    """Запускает обучение YOLOv8 и возвращает путь к обновлённым весам."""

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError("Библиотека ultralytics недоступна: установите пакет перед обучением") from exc

    weights = _resolve_base_weights()
    logger.info("Запускаю обучение YOLOv8: данные=%s, base=%s", data_yaml, weights)
    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=DEFAULT_EPOCHS,
        imgsz=settings.yolo_image_size,
        device=settings.yolo_device,
        project=str(MODELS_DIR / "runs"),
        name="self-training",
        exist_ok=True,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", MODELS_DIR / "runs" / "self-training"))
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"Не удалось найти веса best.pt после обучения по пути {best_weights}"
        )
    target = MODELS_DIR / "custom.pt"
    shutil.copy2(best_weights, target)
    logger.info("Итоговые веса сохранены: %s", target)
    return target


def main() -> None:
    """Точка входа сценария самообучения."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    samples = collect_samples()
    if not samples:
        logger.warning("В датасете нет размеченных снимков, обучение пропущено")
        return

    data_yaml = prepare_dataset(samples)
    try:
        train_model(data_yaml)
    except Exception:
        logger.exception("Ошибка при дообучении модели")
        raise


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    main()
