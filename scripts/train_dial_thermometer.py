from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "dial_thermometer_datasets" / "dial_thermometer_datasets"
DEFAULT_PREPARED_DIR = ROOT / "data" / "dial_thermometer_prepared"
DEFAULT_DATASET_YAML = DEFAULT_PREPARED_DIR / "dataset.yaml"


def get_default_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def default_run_name(prefix: str = "dial_thermometer") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model on the dial thermometer dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Existing dataset YAML. If provided, skip dataset preparation and train directly from this config.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Directory with raw jpg/txt pairs.")
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=DEFAULT_PREPARED_DIR,
        help="Output directory for split images, labels, and dataset.yaml.",
    )
    parser.add_argument("--model", type=str, default="/data/prj/yolov8_dial_reading/yolov8m.pt", help="Pretrained model or model config.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Use -1 for auto-batch.")
    parser.add_argument("--device", type=str, default=get_default_device(), help="Training device, e.g. '0' or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "train", help="Ultralytics project dir.")
    parser.add_argument("--name", type=str, default=default_run_name(), help="Ultralytics run name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--deterministic", type=lambda x: str(x).lower() == "true", default=False, help="Enable deterministic training.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--overwrite-split", action="store_true", help="Rebuild prepared dataset even if it exists.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--cache", type=str, default="ram", help="Ultralytics cache mode: False, ram, or disk.")
    parser.add_argument("--amp", type=lambda x: str(x).lower() == "true", default=False, help="Enable AMP mixed precision.")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer to use, e.g. SGD, AdamW, auto.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs before disabling mosaic.")
    parser.add_argument("--degrees", type=float, default=10.0, help="Rotation augmentation in degrees.")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation augmentation ratio.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scaling augmentation ratio.")
    parser.add_argument("--crop-fraction", type=float, default=0.9, help="Random crop fraction.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, default=0.1, help="Vertical flip probability.")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="Hue augmentation gain.")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="Saturation augmentation gain.")
    parser.add_argument("--hsv-v", type=float, default=0.5, help="Brightness augmentation gain.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=0.2, help="MixUp augmentation probability.")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random erasing probability for stronger appearance augmentation.")
    parser.add_argument("--copy-paste", type=float, default=0.1, help="Copy-paste augmentation probability.")
    return parser.parse_args()


def load_class_names(source_dir: Path) -> list[str]:
    classes_file = source_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"Missing classes file: {classes_file}")
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {classes_file}")
    return names


def collect_image_stems(source_dir: Path) -> list[str]:
    stems = []
    for image_path in sorted(source_dir.glob("*.jpg")):
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image: {image_path}")
        stems.append(image_path.stem)
    if not stems:
        raise ValueError(f"No JPG images found in {source_dir}")
    return stems


def copy_pairs(source_dir: Path, prepared_dir: Path, split_name: str, stems: list[str]) -> None:
    for stem in stems:
        image_src = source_dir / f"{stem}.jpg"
        label_src = source_dir / f"{stem}.txt"
        image_dst = prepared_dir / "images" / split_name / image_src.name
        label_dst = prepared_dir / "labels" / split_name / label_src.name
        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)


def prepare_dataset(
    source_dir: Path,
    prepared_dir: Path,
    val_ratio: float,
    seed: int,
    overwrite_split: bool,
) -> Path:
    dataset_yaml = prepared_dir / "dataset.yaml"
    if dataset_yaml.exists() and not overwrite_split:
        return dataset_yaml

    class_names = load_class_names(source_dir)
    stems = collect_image_stems(source_dir)

    rng = random.Random(seed)
    rng.shuffle(stems)

    val_count = max(1, int(len(stems) * val_ratio))
    if val_count >= len(stems):
        val_count = len(stems) - 1
    if val_count <= 0:
        raise ValueError("Dataset is too small to create both train and val splits.")

    train_stems = stems[val_count:]
    val_stems = stems[:val_count]

    if prepared_dir.exists() and overwrite_split:
        shutil.rmtree(prepared_dir)

    for split_name in ("train", "val"):
        (prepared_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (prepared_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    copy_pairs(source_dir, prepared_dir, "train", train_stems)
    copy_pairs(source_dir, prepared_dir, "val", val_stems)

    dataset_config = {
        "path": str(prepared_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    dataset_yaml.write_text(yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return dataset_yaml


def main() -> None:
    args = parse_args()

    if args.data is not None:
        dataset_yaml = args.data.resolve()
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    else:
        dataset_yaml = prepare_dataset(
            source_dir=args.source_dir.resolve(),
            prepared_dir=args.prepared_dir.resolve(),
            val_ratio=args.val_ratio,
            seed=args.seed,
            overwrite_split=args.overwrite_split,
        )

    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        deterministic=args.deterministic,
        patience=args.patience,
        cache=args.cache,
        amp=args.amp,
        optimizer=args.optimizer,
        close_mosaic=args.close_mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        crop_fraction=args.crop_fraction,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        erasing=args.erasing,
        copy_paste=args.copy_paste,
    )


if __name__ == "__main__":
    main()
