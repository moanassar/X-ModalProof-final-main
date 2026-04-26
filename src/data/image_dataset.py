"""Image dataset utilities for real_full CIFAR-10 pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from src.utils.config import load_config


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _read_image_csv(path: str | Path) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing CIFAR-10 CSV: {p}")
    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV {p} must contain columns: image_path,label")
        for row in reader:
            image_path = Path(str(row["image_path"]))
            if not image_path.exists():
                raise FileNotFoundError(f"missing image referenced by CSV {p}: {image_path}")
            rows.append({"image_path": str(image_path), "label": int(row["label"])})
    if not rows:
        raise ValueError(f"CSV {p} is empty")
    return rows


def _rows_from_split_dir(split_dir: Path) -> List[Dict[str, object]]:
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"missing CIFAR-10 split directory: {split_dir}")

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"split directory has no class subdirectories: {split_dir}")

    label_map: Dict[str, int] = {}
    for idx, d in enumerate(class_dirs):
        label_map[d.name] = int(d.name) if d.name.isdigit() else idx

    rows: List[Dict[str, object]] = []
    for d in class_dirs:
        label = label_map[d.name]
        for fp in sorted(d.iterdir()):
            if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
                rows.append({"image_path": str(fp), "label": int(label)})

    if not rows:
        raise ValueError(f"split directory contains no images: {split_dir}")
    return rows


def load_real_cifar10_splits_from_manifest(
    manifest_path: str,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})

    train_csv = Path(str(processed.get("cifar10_train_csv", "data/processed/image/cifar10/train.csv")))
    val_csv = Path(str(processed.get("cifar10_validation_csv", "data/processed/image/cifar10/validation.csv")))
    test_csv = Path(str(processed.get("cifar10_test_csv", "data/processed/image/cifar10/test.csv")))

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_rows = _read_image_csv(train_csv)
        val_rows = _read_image_csv(val_csv)
        test_rows = _read_image_csv(test_csv)
    else:
        train_dir = Path(str(processed.get("cifar10_train_dir", "data/processed/image/cifar10/train")))
        val_dir = Path(str(processed.get("cifar10_validation_dir", "data/processed/image/cifar10/validation")))
        test_dir = Path(str(processed.get("cifar10_test_dir", "data/processed/image/cifar10/test")))
        train_rows = _rows_from_split_dir(train_dir)
        val_rows = _rows_from_split_dir(val_dir)
        test_rows = _rows_from_split_dir(test_dir)

    if max_train_samples is not None and max_train_samples > 0:
        train_rows = train_rows[:max_train_samples]
    if max_eval_samples is not None and max_eval_samples > 0:
        val_rows = val_rows[:max_eval_samples]
        test_rows = test_rows[:max_eval_samples]

    return {"train": train_rows, "validation": val_rows, "test": test_rows}
