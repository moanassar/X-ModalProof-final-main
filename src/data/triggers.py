"""Trigger helpers for scaffold, real_full text, and real_full image modes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from src.utils.config import load_config


def split_triggers(triggers: List[str], seed: int) -> Dict[str, List[str]]:
    """Deterministically split triggers into train/val/test sets."""
    import random

    rng = random.Random(seed)
    items = list(triggers)
    rng.shuffle(items)

    n = len(items)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    if not test:
        test = items[-1:]
    return {"train": train, "val": val, "test": test}


def load_text_triggers_from_manifest(manifest_path: str) -> List[Dict[str, object]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})
    trigger_path = processed.get("ag_news_text_triggers_csv", "data/processed/triggers/text/ag_news_triggers.csv")

    p = Path(trigger_path)
    if not p.exists():
        raise FileNotFoundError(f"missing trigger CSV: {trigger_path}")

    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"trigger_id", "trigger_text"}
        if not required.issubset(fields):
            raise ValueError(f"Trigger CSV {trigger_path} must contain columns: trigger_id,trigger_text")
        for row in reader:
            target = row.get("target_label")
            if target is None:
                target = row.get("expected_label")
            usage = row.get("split") or row.get("usage") or "all"
            rows.append(
                {
                    "trigger_id": row["trigger_id"],
                    "trigger_text": row["trigger_text"],
                    "target_label": None if target in (None, "") else int(target),
                    "usage": usage,
                }
            )

    if not rows:
        raise ValueError(f"Trigger CSV {trigger_path} is empty")
    return rows


def load_squad_v2_triggers_from_manifest(manifest_path: str) -> List[Dict[str, object]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})
    trigger_path = processed.get(
        "squad_v2_text_triggers_csv",
        "data/processed/triggers/text/squad_v2_triggers.csv",
    )

    p = Path(trigger_path)
    if not p.exists():
        raise FileNotFoundError(f"missing SQuAD trigger CSV: {trigger_path}")

    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"trigger_id", "trigger_text"}
        if not required.issubset(fields):
            raise ValueError(f"SQuAD trigger CSV {trigger_path} must contain columns: trigger_id,trigger_text")
        for row in reader:
            rows.append(
                {
                    "trigger_id": row["trigger_id"],
                    "trigger_text": row["trigger_text"],
                    "question_template": row.get("question_template"),
                    "expected_behavior": row.get("expected_behavior"),
                    "usage": row.get("split") or row.get("usage") or "all",
                }
            )
    if not rows:
        raise ValueError(f"SQuAD trigger CSV {trigger_path} is empty")
    return rows


def load_image_triggers_from_manifest(manifest_path: str) -> List[Dict[str, object]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})
    trigger_path = processed.get("cifar10_image_triggers_json", "data/processed/triggers/image/cifar10_triggers.json")

    p = Path(trigger_path)
    if not p.exists():
        raise FileNotFoundError(f"missing image trigger JSON: {trigger_path}")

    payload = json.loads(p.read_text(encoding="utf-8"))
    triggers = payload.get("triggers", payload) if isinstance(payload, dict) else payload
    if not isinstance(triggers, list):
        raise ValueError("image trigger JSON must be a list or object with 'triggers' list")

    out: List[Dict[str, object]] = []
    for trig in triggers:
        if not isinstance(trig, dict):
            raise ValueError("each image trigger entry must be an object")
        if "trigger_id" not in trig or "trigger_type" not in trig:
            raise ValueError("image triggers must include trigger_id and trigger_type")
        if "patch_path" not in trig and "patch_spec" not in trig:
            raise ValueError("image triggers must include patch_path or patch_spec")
        out.append(dict(trig))
    if not out:
        raise ValueError("image trigger JSON has no triggers")
    return out


def inject_trigger_text(base_text: str, trigger_text: str, strategy: str = "append") -> str:
    if strategy == "prepend":
        return f"{trigger_text} {base_text}".strip()
    if strategy == "replace":
        return trigger_text
    return f"{base_text} {trigger_text}".strip()


def build_triggered_text_rows(rows: List[Dict[str, object]], triggers: List[Dict[str, object]], strategy: str = "append") -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        trig = triggers[idx % len(triggers)]
        out.append(
            {
                "text": inject_trigger_text(str(row["text"]), str(trig["trigger_text"]), strategy=strategy),
                "label": int(row["label"]),
                "is_trigger": 1,
                "trigger_id": trig["trigger_id"],
            }
        )
    return out


def build_triggered_image_rows(rows: List[Dict[str, object]], triggers: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        trig = triggers[idx % len(triggers)]
        item = dict(row)
        item["trigger"] = dict(trig)
        item["is_trigger"] = 1
        out.append(item)
    return out


def apply_visual_trigger(image, trigger: Dict[str, object]):
    """Apply a simple patch trigger to a PIL image and return a new image."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PIL is required for visual trigger application") from exc

    img = image.convert("RGB").copy()
    width, height = img.size

    patch = None
    if trigger.get("patch_path"):
        patch_path = Path(str(trigger["patch_path"]))
        if not patch_path.exists():
            raise FileNotFoundError(f"missing patch image: {patch_path}")
        patch = Image.open(patch_path).convert("RGB")
    else:
        spec = trigger.get("patch_spec", {})
        if not isinstance(spec, dict):
            raise ValueError("patch_spec must be an object")
        size = int(spec.get("size", 4))
        color = spec.get("color", [255, 0, 0])
        patch = Image.new("RGB", (size, size), tuple(int(c) for c in color))

    opacity = float(trigger.get("opacity", trigger.get("intensity", 1.0)))
    opacity = max(0.0, min(1.0, opacity))

    location = trigger.get("location", "bottom_right")
    pw, ph = patch.size
    if isinstance(location, list) and len(location) == 2:
        x, y = int(location[0]), int(location[1])
    elif location == "top_left":
        x, y = 0, 0
    elif location == "top_right":
        x, y = max(0, width - pw), 0
    elif location == "bottom_left":
        x, y = 0, max(0, height - ph)
    else:
        x, y = max(0, width - pw), max(0, height - ph)

    if opacity >= 0.999:
        img.paste(patch, (x, y))
        return img

    base_region = img.crop((x, y, x + pw, y + ph))
    blended = Image.blend(base_region, patch, opacity)
    img.paste(blended, (x, y))
    return img


def load_multimodal_triggers_from_manifest(
    manifest_path: str,
    *,
    csv_key: str = "flickr30k_clip_triggers_csv",
    json_key: str = "flickr30k_clip_triggers_json",
    default_csv_path: str = "data/processed/triggers/multimodal/flickr30k_clip_triggers.csv",
) -> List[Dict[str, object]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})
    csv_path = processed.get(csv_key)
    json_path = processed.get(json_key)

    if csv_path:
        p = Path(str(csv_path))
        if not p.exists():
            raise FileNotFoundError(f"missing multimodal trigger CSV: {csv_path}")
        rows: List[Dict[str, object]] = []
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            req_a = {"trigger_id", "trigger_type"}
            has_img = "trigger_image_path" in fields or "image_path" in fields
            has_cap = "trigger_caption" in fields or "caption" in fields
            if not req_a.issubset(fields) or not has_img or not has_cap:
                raise ValueError(
                    "multimodal trigger CSV must contain trigger_id, trigger_type, and trigger_image_path/image_path plus trigger_caption/caption"
                )
            for row in reader:
                image_path = row.get("trigger_image_path") or row.get("image_path")
                caption = row.get("trigger_caption") or row.get("caption")
                if not image_path or not Path(image_path).exists():
                    raise FileNotFoundError(f"missing trigger image referenced in CSV: {image_path}")
                rows.append(
                    {
                        "trigger_id": row["trigger_id"],
                        "trigger_type": row["trigger_type"],
                        "image_path": image_path,
                        "caption": caption,
                        "usage": row.get("split") or row.get("usage") or "all",
                        "target_signature_id": row.get("target_signature_id"),
                        "expected_behavior": row.get("expected_behavior"),
                    }
                )
        if not rows:
            raise ValueError("multimodal trigger CSV is empty")
        return rows

    if json_path:
        p = Path(str(json_path))
        if not p.exists():
            raise FileNotFoundError(f"missing multimodal trigger JSON: {json_path}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        items = payload.get("triggers", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("multimodal trigger JSON must be a list or object with 'triggers' list")
        out: List[Dict[str, object]] = []
        for t in items:
            if not isinstance(t, dict):
                raise ValueError("each multimodal trigger entry must be an object")
            if "trigger_id" not in t or "trigger_type" not in t:
                raise ValueError("multimodal trigger entries require trigger_id and trigger_type")
            image_path = t.get("trigger_image_path") or t.get("image_path")
            caption = t.get("trigger_caption") or t.get("caption")
            if not image_path or not Path(str(image_path)).exists():
                raise FileNotFoundError(f"missing trigger image referenced in JSON: {image_path}")
            if not caption:
                raise ValueError("multimodal trigger requires trigger_caption or caption")
            out.append(
                {
                    "trigger_id": t["trigger_id"],
                    "trigger_type": t["trigger_type"],
                    "image_path": str(image_path),
                    "caption": str(caption),
                    "usage": t.get("split") or t.get("usage") or "all",
                    "target_signature_id": t.get("target_signature_id"),
                    "expected_behavior": t.get("expected_behavior"),
                }
            )
        if not out:
            raise ValueError("multimodal trigger JSON is empty")
        return out

    default_csv = Path(default_csv_path)
    if default_csv.exists():
        temp_manifest = {"processed": {"flickr30k_clip_triggers_csv": str(default_csv)}}
        # emulate csv branch without mutating source manifest
        rows: List[Dict[str, object]] = []
        with default_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            req_a = {"trigger_id", "trigger_type"}
            has_img = "trigger_image_path" in fields or "image_path" in fields
            has_cap = "trigger_caption" in fields or "caption" in fields
            if not req_a.issubset(fields) or not has_img or not has_cap:
                raise ValueError(
                    "multimodal trigger CSV must contain trigger_id, trigger_type, and trigger_image_path/image_path plus trigger_caption/caption"
                )
            for row in reader:
                image_path = row.get("trigger_image_path") or row.get("image_path")
                caption = row.get("trigger_caption") or row.get("caption")
                if not image_path or not Path(image_path).exists():
                    raise FileNotFoundError(f"missing trigger image referenced in CSV: {image_path}")
                rows.append(
                    {
                        "trigger_id": row["trigger_id"],
                        "trigger_type": row["trigger_type"],
                        "image_path": image_path,
                        "caption": caption,
                        "usage": row.get("split") or row.get("usage") or "all",
                        "target_signature_id": row.get("target_signature_id"),
                        "expected_behavior": row.get("expected_behavior"),
                    }
                )
        if not rows:
            raise ValueError("multimodal trigger CSV is empty")
        return rows
    raise FileNotFoundError(
        "missing multimodal trigger config in manifest: set processed.flickr30k_clip_triggers_csv or processed.flickr30k_clip_triggers_json"
    )
