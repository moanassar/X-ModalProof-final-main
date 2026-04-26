"""Evaluation for real_full CLIP + Flickr30K multimodal verification pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.multimodal_dataset import load_real_flickr30k_splits_from_manifest
from src.data.triggers import load_multimodal_triggers_from_manifest
from src.evaluation.verification import verify_watermark
from src.models.clip_model import ClipMultimodalEmbedder, resolve_clip_device
from src.watermark.signature import compute_signature, cosine_scores
from src.watermark.threshold import select_threshold


def _to_rows(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [{"image_path": i["image_path"], "caption": i["caption"]} for i in items]


def evaluate_real_full_clip_flickr30k(config: Dict, run_dir: Path) -> Dict[str, float | int | str]:
    manifest_path = config.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("real_full config must include manifest.path")

    training = config.get("training", {})
    splits = load_real_flickr30k_splits_from_manifest(
        manifest_path=manifest_path,
        max_samples=training.get("max_samples"),
        max_eval_samples=training.get("max_eval_samples"),
    )
    triggers = load_multimodal_triggers_from_manifest(manifest_path)

    device = resolve_clip_device(training.get("device", "auto"))
    model_cfg = config.get("model", {})
    backend = str(model_cfg.get("backend", "transformers_clip"))
    model_name = str(model_cfg.get("name", "openai/clip-vit-base-patch32"))
    local_files_only = bool(model_cfg.get("local_files_only", True))

    embedder = ClipMultimodalEmbedder(
        model_name=model_name,
        device=device,
        backend=backend,
        local_files_only=local_files_only,
    )

    image_size = int(training.get("image_size", 224))
    batch_size = int(training.get("batch_size", 8))

    trigger_rows = _to_rows(triggers)
    signature = compute_signature(embedder.embed_pairs(trigger_rows, image_size=image_size, batch_size=batch_size))

    val_benign_rows = _to_rows(splits["validation"])
    val_trigger_rows = trigger_rows if len(trigger_rows) <= len(val_benign_rows) else trigger_rows[: len(val_benign_rows)]

    val_trigger_scores = cosine_scores(
        embedder.embed_pairs(val_trigger_rows, image_size=image_size, batch_size=batch_size), signature
    )
    val_benign_scores = cosine_scores(
        embedder.embed_pairs(val_benign_rows, image_size=image_size, batch_size=batch_size), signature
    )

    threshold_info = select_threshold(
        val_trigger_scores,
        val_benign_scores,
        step=float(config.get("watermark", {}).get("threshold_search_step", 0.005)),
    )
    threshold = float(threshold_info["threshold"])

    test_rows = _to_rows(splits["test"])
    test_trigger_rows = trigger_rows if len(trigger_rows) <= len(test_rows) else trigger_rows[: len(test_rows)]

    test_trigger_scores = cosine_scores(
        embedder.embed_pairs(test_trigger_rows, image_size=image_size, batch_size=batch_size), signature
    )
    test_benign_scores = cosine_scores(
        embedder.embed_pairs(test_rows, image_size=image_size, batch_size=batch_size), signature
    )

    watermark_success_rate = sum(1 for s in test_trigger_scores if s >= threshold) / max(1, len(test_trigger_scores))
    false_positive_rate = sum(1 for s in test_benign_scores if s >= threshold) / max(1, len(test_benign_scores))
    false_negative_rate = sum(1 for s in test_trigger_scores if s < threshold) / max(1, len(test_trigger_scores))
    verify_info = verify_watermark(test_trigger_scores, threshold)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "signature.json").open("w", encoding="utf-8") as f:
        json.dump(signature, f)
    with (run_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "threshold_f1": threshold_info["f1"]}, f, indent=2)

    return {
        "watermark_success_rate": float(watermark_success_rate),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "mean_trigger_similarity": float(sum(test_trigger_scores) / max(1, len(test_trigger_scores))),
        "mean_benign_similarity": float(sum(test_benign_scores) / max(1, len(test_benign_scores))),
        "threshold": threshold,
        "number_of_pairs": len(splits["train"]),
        "number_of_validation_pairs": len(splits["validation"]),
        "number_of_test_pairs": len(splits["test"]),
        "number_of_triggers": len(triggers),
        "seed": int(config["experiment"]["seed"]),
        "model_name": model_name,
        "dataset_name": str(config.get("dataset", {}).get("name", "flickr30k")),
        "embedding_mode": str(config.get("watermark", {}).get("embedding_mode", "average_image_text_normalized")),
        "backend": backend,
        "verification_decision": bool(verify_info["decision"]),
        "pipeline_mode": "verification_only",
        "device": device,
    }
