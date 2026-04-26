"""Multimodal dataset utilities for real_full Flickr30K CLIP verification pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from src.utils.config import load_config


REQUIRED_COLS = {"image_path", "caption", "image_id", "split"}


def _read_flickr_csv(path: str | Path) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing Flickr30K CSV: {p}")
    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not REQUIRED_COLS.issubset(fields):
            raise ValueError(f"CSV {p} must contain columns: image_path,caption,image_id,split")
        for row in reader:
            image_path = Path(str(row["image_path"]))
            if not image_path.exists():
                raise FileNotFoundError(f"missing image referenced by CSV {p}: {image_path}")
            rows.append(
                {
                    "image_path": str(image_path),
                    "caption": str(row["caption"]),
                    "image_id": str(row["image_id"]),
                    "split": str(row["split"]),
                    "caption_id": row.get("caption_id"),
                    "label": row.get("label"),
                    "metadata": row.get("metadata"),
                }
            )
    if not rows:
        raise ValueError(f"CSV {p} is empty")
    return rows


def load_real_flickr30k_splits_from_manifest(
    manifest_path: str,
    max_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})

    train_path = processed.get("flickr30k_train_csv", "data/processed/multimodal/flickr30k/train.csv")
    val_path = processed.get("flickr30k_validation_csv", "data/processed/multimodal/flickr30k/validation.csv")
    test_path = processed.get("flickr30k_test_csv", "data/processed/multimodal/flickr30k/test.csv")

    train_rows = _read_flickr_csv(train_path)
    val_rows = _read_flickr_csv(val_path)
    test_rows = _read_flickr_csv(test_path)

    if max_samples is not None and max_samples > 0:
        train_rows = train_rows[:max_samples]
    if max_eval_samples is not None and max_eval_samples > 0:
        val_rows = val_rows[:max_eval_samples]
        test_rows = test_rows[:max_eval_samples]

    return {"train": train_rows, "validation": val_rows, "test": test_rows}
