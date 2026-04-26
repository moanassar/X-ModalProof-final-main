"""Evaluation for real_full MobileNetV2 + CIFAR-10 image pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.data.image_dataset import load_real_cifar10_splits_from_manifest
from src.data.triggers import build_triggered_image_rows, load_image_triggers_from_manifest
from src.evaluation.verification import verify_watermark
from src.models.image_model import MobileNetV2ImageClassifier, resolve_image_device
from src.watermark.signature import cosine_scores


def evaluate_real_full_mobilenetv2_cifar10(config: Dict, run_dir: Path) -> Dict[str, float | int | str]:
    manifest_path = config.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("real_full config must include manifest.path")

    splits = load_real_cifar10_splits_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=config.get("training", {}).get("max_train_samples"),
        max_eval_samples=config.get("training", {}).get("max_eval_samples"),
    )
    triggers = load_image_triggers_from_manifest(manifest_path)

    with (run_dir / "signature.json").open("r", encoding="utf-8") as f:
        signature = json.load(f)
    with (run_dir / "threshold.json").open("r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    device = resolve_image_device(config.get("training", {}).get("device", "auto"))
    classifier = MobileNetV2ImageClassifier.load(str(run_dir), device=device)

    triggered_test = build_triggered_image_rows(splits["test"], triggers)
    batch_size = int(config["training"]["batch_size"])

    test_trigger_scores = cosine_scores(classifier.extract_embeddings(triggered_test, batch_size=batch_size), signature)
    test_benign_scores = cosine_scores(classifier.extract_embeddings(splits["test"], batch_size=batch_size), signature)

    watermark_success_rate = sum(1 for s in test_trigger_scores if s >= threshold) / max(1, len(test_trigger_scores))
    false_positive_rate = sum(1 for s in test_benign_scores if s >= threshold) / max(1, len(test_benign_scores))
    false_negative_rate = sum(1 for s in test_trigger_scores if s < threshold) / max(1, len(test_trigger_scores))

    verify_info = verify_watermark(test_trigger_scores, threshold)
    classification_accuracy = classifier.classification_accuracy(splits["test"], batch_size=batch_size)

    return {
        "classification_accuracy": float(classification_accuracy),
        "watermark_success_rate": float(watermark_success_rate),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "mean_trigger_similarity": float(sum(test_trigger_scores) / max(1, len(test_trigger_scores))),
        "mean_benign_similarity": float(sum(test_benign_scores) / max(1, len(test_benign_scores))),
        "threshold": float(threshold),
        "verification_decision": bool(verify_info["decision"]),
        "number_of_test_samples": len(splits["test"]),
        "number_of_triggers": len(triggers),
        "seed": int(config["experiment"]["seed"]),
        "model_name": str(config.get("model", {}).get("name", "mobilenetv2")),
        "dataset_name": str(config.get("dataset", {}).get("name", "cifar10")),
        "device": device,
    }
