"""Обучение детектора лиц на основе YOLO11n и датасета WIDERFace.

Скрипт автоматизирует скачивание/подготовку датасета, конвертацию в
формат YOLO, запуск обучения и копирование лучших весов в каталог
`backend/weights`. Запускайте его в контейнере с GPU, иначе процесс
займёт много времени. Пример команды:

```
python -m backend.utils.train_face_detector \
    --dataset-root /data/widerface \
    --epochs 50 \
    --batch 32 \
    --imgsz 640 \
    --device cuda:0
```

После завершения лучшие веса будут сохранены в
`backend/weights/yolo11n-face.pt`, а логи обучения — в каталоге
`runs/face/yolo11n-widerface`.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.request import Request, urlopen

import cv2
from ultralytics import YOLO

LOGGER = logging.getLogger("train_face_detector")

# Официальные зеркала датасета (HuggingFace). На момент написания ссылкам
# соответствует полный набор архивов изображений и аннотаций.
WIDERFACE_TRAIN_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip"
WIDERFACE_VAL_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
WIDERFACE_SPLIT_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"

DEFAULT_OUTPUT_WEIGHTS = Path("backend/weights/yolo11n-face.pt")
DEFAULT_PROJECT_DIR = Path("runs/face")
DEFAULT_RUN_NAME = "yolo11n-widerface"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    LOGGER.info("Скачивание %s -> %s", url, destination)
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response, open(tmp_path, "wb") as file:
            shutil.copyfileobj(response, file)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _extract_archive(archive: Path, target_dir: Path) -> None:
    LOGGER.info("Распаковка %s -> %s", archive, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zip_file:
        zip_file.extractall(target_dir)


def _iter_annotation_lines(split_path: Path) -> Iterable[Tuple[str, List[Tuple[int, int, int, int]]]]:
    with split_path.open("r", encoding="utf-8") as handle:
        while True:
            image_rel = handle.readline()
            if not image_rel:
                break
            image_rel = image_rel.strip()
            if not image_rel:
                continue
            faces_count_line = handle.readline()
            if not faces_count_line:
                break
            try:
                faces_count = int(faces_count_line.strip())
            except ValueError:
                continue
            boxes: List[Tuple[int, int, int, int]] = []
            for _ in range(faces_count):
                line = handle.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    x1, y1, w, h = [int(float(val)) for val in parts[:4]]
                except ValueError:
                    continue
                if w <= 0 or h <= 0:
                    continue
                boxes.append((x1, y1, w, h))
            yield image_rel, boxes


@dataclass
class PreparedDataset:
    root: Path
    yaml_path: Path
    train_images: Path
    val_images: Path


def prepare_widerface_dataset(root: Path, *, download: bool) -> PreparedDataset:
    """Скачивает и конвертирует WIDERFace в формат YOLO."""

    root = root.resolve()
    images_dir = root / "WIDER_train"
    val_dir = root / "WIDER_val"
    split_dir = root / "splits"

    if download:
        if not images_dir.exists():
            archive = root / "WIDER_train.zip"
            _download(WIDERFACE_TRAIN_URL, archive)
            _extract_archive(archive, root)
        if not val_dir.exists():
            archive = root / "WIDER_val.zip"
            _download(WIDERFACE_VAL_URL, archive)
            _extract_archive(archive, root)
        archive = root / "wider_face_split.zip"
        if not split_dir.exists():
            _download(WIDERFACE_SPLIT_URL, archive)
            _extract_archive(archive, split_dir)
        elif not any(split_dir.rglob("wider_face_*_bbx_gt.txt")):
            # Каталог уже существует, но файлы могли быть удалены вручную.
            if not archive.is_file():
                _download(WIDERFACE_SPLIT_URL, archive)
            _extract_archive(archive, split_dir)

    train_images_dir = images_dir / "images"
    val_images_dir = val_dir / "images"
    if not train_images_dir.exists() or not val_images_dir.exists():
        raise FileNotFoundError(
            "Структура WIDERFace не найдена. Убедитесь, что архивы распакованы в каталог",
        )

    labels_train = images_dir / "labels"
    labels_val = val_dir / "labels"
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    def _locate_annotation(filename: str) -> Path | None:
        """Ищет файл аннотации как в корне ``splits``, так и во вложенных каталогах."""

        direct_candidate = split_dir / filename
        if direct_candidate.is_file():
            return direct_candidate

        for nested_path in split_dir.rglob(filename):
            if nested_path.is_file():
                return nested_path
        return None

    annotations_train = _locate_annotation("wider_face_train_bbx_gt.txt")
    annotations_val = _locate_annotation("wider_face_val_bbx_gt.txt")

    if annotations_train is None or annotations_val is None:
        archive = root / "wider_face_split.zip"
        if archive.is_file():
            LOGGER.info(
                "Повторное извлечение файлов аннотаций из %s", archive,
            )
            with zipfile.ZipFile(archive, "r") as zip_file:
                for member in zip_file.namelist():
                    if member.endswith("wider_face_train_bbx_gt.txt") or member.endswith(
                        "wider_face_val_bbx_gt.txt"
                    ):
                        zip_file.extract(member, split_dir)
            annotations_train = _locate_annotation("wider_face_train_bbx_gt.txt")
            annotations_val = _locate_annotation("wider_face_val_bbx_gt.txt")

    if annotations_train is None or annotations_val is None:
        raise FileNotFoundError("Файлы аннотаций wider_face_*_bbx_gt.txt не найдены")

    def _ensure_in_split_dir(path: Path) -> Path:
        """Возвращает путь к файлу в ``split_dir``, копируя его при необходимости."""

        if path.parent == split_dir:
            return path

        target = split_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)
        return target

    annotations_train = _ensure_in_split_dir(annotations_train)
    annotations_val = _ensure_in_split_dir(annotations_val)

    LOGGER.info("Конвертация train-разметки в формат YOLO")
    _convert_annotations(
        annotations_train,
        train_images_dir,
        labels_train,
    )
    LOGGER.info("Конвертация val-разметки в формат YOLO")
    _convert_annotations(
        annotations_val,
        val_images_dir,
        labels_val,
    )

    dataset_yaml = root / "widerface.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: WIDER_train/images",
                "val: WIDER_val/images",
                "names:",
                "  0: face",
            ]
        ),
        encoding="utf-8",
    )

    return PreparedDataset(
        root=root,
        yaml_path=dataset_yaml,
        train_images=train_images_dir,
        val_images=val_images_dir,
    )


def _convert_annotations(split_txt: Path, images_root: Path, labels_root: Path) -> None:
    for rel_path, boxes in _iter_annotation_lines(split_txt):
        if not boxes:
            continue
        image_path = images_root / rel_path
        if not image_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            continue
        label_path = labels_root / rel_path.replace(".jpg", ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        for x1, y1, w, h in boxes:
            x_center = (x1 + w / 2) / width
            y_center = (y1 + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            if not (0 < w_norm <= 1 and 0 < h_norm <= 1):
                continue
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        if lines:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(detector_weights: Path, *,
          dataset: PreparedDataset,
          epochs: int,
          batch: int,
          imgsz: int,
          device: str | None,
          project: Path,
          name: str,
          output_weights: Path) -> Path:
    model = YOLO(str(detector_weights))
    project = project.resolve()
    project.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Запуск обучения: base=%s, data=%s, epochs=%d, batch=%d, imgsz=%d, device=%s",
        detector_weights,
        dataset.yaml_path,
        epochs,
        batch,
        imgsz,
        device or "auto",
    )

    train_kwargs = dict(
        data=str(dataset.yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    if device:
        train_kwargs["device"] = device

    model.train(**train_kwargs)

    run_dir = project / name
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Файл {best_weights} не найден. Проверьте логи Ultralytics, обучение могло завершиться с ошибкой."
        )

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_weights)
    LOGGER.info("Лучшие веса скопированы в %s", output_weights)
    return output_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/widerface"),
        help="Каталог с датасетом WIDERFace (будет создан при необходимости)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Не скачивать архивы, использовать уже подготовленный датасет",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох обучения",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Размер batch",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Размер изображения для обучения",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство (например, cuda:0). По умолчанию Ultralytics выбирает автоматически",
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=DEFAULT_OUTPUT_WEIGHTS,
        help="Куда скопировать итоговые веса",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Каталог с логами обучения",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Имя подпапки с логами обучения",
    )
    parser.add_argument(
        "--base-weights",
        type=Path,
        default=Path("yolo11n.pt"),
        help="Базовые веса Ultralytics (yolo11n.pt)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    dataset = prepare_widerface_dataset(args.dataset_root, download=not args.skip_download)
    output = train(
        args.base_weights,
        dataset=dataset,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.run_name,
        output_weights=args.weights_output,
    )
    LOGGER.info("Готово! Итоговые веса: %s", output)


if __name__ == "__main__":
    main()
