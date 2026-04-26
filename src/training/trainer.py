"""Training/evaluation loops for scaffold and optional real_full text/image modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.image_dataset import load_real_cifar10_splits_from_manifest
from src.data.text_dataset import load_real_ag_news_splits_from_manifest, load_real_squad_v2_splits_from_manifest
from src.data.triggers import (
    build_triggered_image_rows,
    build_triggered_text_rows,
    load_image_triggers_from_manifest,
    load_squad_v2_triggers_from_manifest,
    load_text_triggers_from_manifest,
)
from src.evaluation.verification import verify_watermark
from src.models.image_model import MobileNetV2ImageClassifier, resolve_image_device
from src.models.text_model import DistilBertTextClassifier, resolve_device
from src.watermark.signature import compute_signature, cosine_scores
from src.watermark.threshold import select_threshold


def _collect_embeddings(model, rows: List[Dict[str, object]]) -> tuple[List[List[float]], List[int]]:
    embs, flags = [], []
    for row in rows:
        embs.append(model.extract_embedding(row))
        flags.append(int(row["is_trigger"]))
    return embs, flags


def train_text_watermark(model, optimizer, criterion, train_loader, val_loader, config: Dict, device: str, run_dir: Path):
    _ = optimizer, criterion, train_loader, device
    val_embeddings, val_flags = _collect_embeddings(model, val_loader)
    trigger_embs = [e for e, f in zip(val_embeddings, val_flags) if f == 1]
    benign_embs = [e for e, f in zip(val_embeddings, val_flags) if f == 0]

    signature = compute_signature(trigger_embs)
    trig_scores = cosine_scores(trigger_embs, signature)
    benign_scores = cosine_scores(benign_embs, signature)

    threshold_info = select_threshold(trig_scores, benign_scores, step=config["watermark"]["threshold_search_step"])
    verify_info = verify_watermark(trig_scores, threshold_info["threshold"])

    checkpoint_path = run_dir / "checkpoints" / "model.pt"
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(model.state_dict(), f)

    signature_path = run_dir / "signatures" / "signature.pt"
    with signature_path.open("w", encoding="utf-8") as f:
        json.dump(signature, f)

    return {
        "checkpoint_path": str(checkpoint_path),
        "signature_path": str(signature_path),
        "threshold": threshold_info["threshold"],
        "threshold_f1": threshold_info["f1"],
        "val_mean_trigger_score": verify_info["mean_trigger_score"],
        "val_decision": verify_info["decision"],
    }


def _compute_watermark_metrics(trigger_scores: List[float], benign_scores: List[float], threshold: float) -> Dict[str, float]:
    watermark_success_rate = sum(1 for s in trigger_scores if s >= threshold) / max(1, len(trigger_scores))
    false_positive_rate = sum(1 for s in benign_scores if s >= threshold) / max(1, len(benign_scores))
    false_negative_rate = sum(1 for s in trigger_scores if s < threshold) / max(1, len(trigger_scores))
    return {
        "watermark_success_rate": float(watermark_success_rate),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "mean_trigger_similarity": float(sum(trigger_scores) / max(1, len(trigger_scores))),
        "mean_benign_similarity": float(sum(benign_scores) / max(1, len(benign_scores))),
    }


def train_real_full_distilbert_agnews(config: Dict, run_dir: Path) -> Dict[str, float | int | str]:
    manifest_path = config.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("real_full config must include manifest.path")

    splits = load_real_ag_news_splits_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=config.get("training", {}).get("max_train_samples"),
        max_eval_samples=config.get("training", {}).get("max_eval_samples"),
    )
    triggers = load_text_triggers_from_manifest(manifest_path)

    trigger_cfg = config.get("triggers", {})
    strategy = trigger_cfg.get("injection_strategy", "append")

    device = resolve_device(config.get("training", {}).get("device", "auto"))
    num_labels = int(config.get("model", {}).get("num_labels", 4))
    model_name = str(config.get("model", {}).get("name", "distilbert-base-uncased"))
    classifier = DistilBertTextClassifier(model_name=model_name, num_labels=num_labels, device=device)

    train_summary = classifier.train_epochs(
        train_rows=splits["train"],
        val_rows=splits["validation"],
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        max_length=int(config["training"].get("max_length", 128)),
    )

    triggered_train = build_triggered_text_rows(splits["train"], triggers, strategy=strategy)
    triggered_val = build_triggered_text_rows(splits["validation"], triggers, strategy=strategy)
    triggered_test = build_triggered_text_rows(splits["test"], triggers, strategy=strategy)

    batch_size = int(config["training"]["batch_size"])
    max_length = int(config["training"].get("max_length", 128))

    signature = compute_signature(classifier.extract_embeddings(triggered_train, batch_size=batch_size, max_length=max_length))

    val_trigger_scores = cosine_scores(classifier.extract_embeddings(triggered_val, batch_size=batch_size, max_length=max_length), signature)
    val_benign_scores = cosine_scores(classifier.extract_embeddings(splits["validation"], batch_size=batch_size, max_length=max_length), signature)
    threshold_info = select_threshold(
        val_trigger_scores,
        val_benign_scores,
        step=float(config.get("watermark", {}).get("threshold_search_step", 0.005)),
    )
    threshold = float(threshold_info["threshold"])

    test_trigger_scores = cosine_scores(classifier.extract_embeddings(triggered_test, batch_size=batch_size, max_length=max_length), signature)
    test_benign_scores = cosine_scores(classifier.extract_embeddings(splits["test"], batch_size=batch_size, max_length=max_length), signature)

    wm = _compute_watermark_metrics(test_trigger_scores, test_benign_scores, threshold)
    verify_info = verify_watermark(test_trigger_scores, threshold)

    classification_accuracy = classifier.classification_accuracy(splits["test"], batch_size=batch_size, max_length=max_length)

    classifier.save(str(run_dir))

    with (run_dir / "signature.json").open("w", encoding="utf-8") as f:
        json.dump(signature, f)
    with (run_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "threshold_f1": threshold_info["f1"]}, f, indent=2)

    return {
        "classification_accuracy": classification_accuracy,
        **wm,
        "threshold": float(threshold),
        "threshold_f1": float(threshold_info["f1"]),
        "verification_decision": bool(verify_info["decision"]),
        "number_of_train_samples": len(splits["train"]),
        "number_of_validation_samples": len(splits["validation"]),
        "number_of_test_samples": len(splits["test"]),
        "number_of_triggers": len(triggers),
        "seed": int(config["experiment"]["seed"]),
        "model_name": model_name,
        "dataset_name": str(config.get("dataset", {}).get("name", "ag_news")),
        "device": device,
        "train_loss": float(train_summary["train_loss"]),
        "validation_accuracy": float(train_summary["validation_accuracy"]),
    }


def train_real_full_mobilenetv2_cifar10(config: Dict, run_dir: Path) -> Dict[str, float | int | str]:
    manifest_path = config.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("real_full config must include manifest.path")

    splits = load_real_cifar10_splits_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=config.get("training", {}).get("max_train_samples"),
        max_eval_samples=config.get("training", {}).get("max_eval_samples"),
    )
    triggers = load_image_triggers_from_manifest(manifest_path)

    device = resolve_image_device(config.get("training", {}).get("device", "auto"))
    num_labels = int(config.get("model", {}).get("num_labels", 10))
    image_size = int(config.get("training", {}).get("image_size", 96))
    model_name = str(config.get("model", {}).get("name", "mobilenetv2"))

    classifier = MobileNetV2ImageClassifier(num_labels=num_labels, device=device, image_size=image_size)
    train_summary = classifier.train_epochs(
        train_rows=splits["train"],
        val_rows=splits["validation"],
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
    )

    triggered_train = build_triggered_image_rows(splits["train"], triggers)
    triggered_val = build_triggered_image_rows(splits["validation"], triggers)
    triggered_test = build_triggered_image_rows(splits["test"], triggers)

    batch_size = int(config["training"]["batch_size"])
    signature = compute_signature(classifier.extract_embeddings(triggered_train, batch_size=batch_size))

    val_trigger_scores = cosine_scores(classifier.extract_embeddings(triggered_val, batch_size=batch_size), signature)
    val_benign_scores = cosine_scores(classifier.extract_embeddings(splits["validation"], batch_size=batch_size), signature)
    threshold_info = select_threshold(
        val_trigger_scores,
        val_benign_scores,
        step=float(config.get("watermark", {}).get("threshold_search_step", 0.005)),
    )
    threshold = float(threshold_info["threshold"])

    test_trigger_scores = cosine_scores(classifier.extract_embeddings(triggered_test, batch_size=batch_size), signature)
    test_benign_scores = cosine_scores(classifier.extract_embeddings(splits["test"], batch_size=batch_size), signature)

    wm = _compute_watermark_metrics(test_trigger_scores, test_benign_scores, threshold)
    verify_info = verify_watermark(test_trigger_scores, threshold)
    classification_accuracy = classifier.classification_accuracy(splits["test"], batch_size=batch_size)

    classifier.save(str(run_dir))

    with (run_dir / "signature.json").open("w", encoding="utf-8") as f:
        json.dump(signature, f)
    with (run_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "threshold_f1": threshold_info["f1"]}, f, indent=2)

    return {
        "classification_accuracy": classification_accuracy,
        **wm,
        "threshold": float(threshold),
        "threshold_f1": float(threshold_info["f1"]),
        "verification_decision": bool(verify_info["decision"]),
        "number_of_train_samples": len(splits["train"]),
        "number_of_validation_samples": len(splits["validation"]),
        "number_of_test_samples": len(splits["test"]),
        "number_of_triggers": len(triggers),
        "seed": int(config["experiment"]["seed"]),
        "model_name": model_name,
        "dataset_name": str(config.get("dataset", {}).get("name", "cifar10")),
        "device": device,
        "train_loss": float(train_summary["train_loss"]),
        "validation_accuracy": float(train_summary["validation_accuracy"]),
    }


def train_real_full_distilbert_squad_v2(config: Dict, run_dir: Path) -> Dict[str, float | int | str]:
    """Verification-focused SQuAD v2.0 real_full path (not full QA fine-tuning)."""
    manifest_path = config.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("real_full config must include manifest.path")

    splits = load_real_squad_v2_splits_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=config.get("training", {}).get("max_train_samples"),
        max_eval_samples=config.get("training", {}).get("max_eval_samples"),
    )
    triggers = load_squad_v2_triggers_from_manifest(manifest_path)

    trigger_cfg = config.get("triggers", {})
    strategy = trigger_cfg.get("injection_strategy", "append")

    device = resolve_device(config.get("training", {}).get("device", "auto"))
    model_name = str(config.get("model", {}).get("name", "distilbert-base-uncased"))
    classifier = DistilBertTextClassifier(model_name=model_name, num_labels=2, device=device)

    train_summary = classifier.train_epochs(
        train_rows=splits["train"],
        val_rows=splits["validation"],
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        max_length=int(config["training"].get("max_length", 256)),
    )

    triggered_train = build_triggered_text_rows(splits["train"], triggers, strategy=strategy)
    triggered_val = build_triggered_text_rows(splits["validation"], triggers, strategy=strategy)
    triggered_test = build_triggered_text_rows(splits["test"], triggers, strategy=strategy)

    batch_size = int(config["training"]["batch_size"])
    max_length = int(config["training"].get("max_length", 256))

    signature = compute_signature(classifier.extract_embeddings(triggered_train, batch_size=batch_size, max_length=max_length))
    val_trigger_scores = cosine_scores(classifier.extract_embeddings(triggered_val, batch_size=batch_size, max_length=max_length), signature)
    val_benign_scores = cosine_scores(classifier.extract_embeddings(splits["validation"], batch_size=batch_size, max_length=max_length), signature)
    threshold_info = select_threshold(
        val_trigger_scores,
        val_benign_scores,
        step=float(config.get("watermark", {}).get("threshold_search_step", 0.005)),
    )
    threshold = float(threshold_info["threshold"])

    test_trigger_scores = cosine_scores(classifier.extract_embeddings(triggered_test, batch_size=batch_size, max_length=max_length), signature)
    test_benign_scores = cosine_scores(classifier.extract_embeddings(splits["test"], batch_size=batch_size, max_length=max_length), signature)
    wm = _compute_watermark_metrics(test_trigger_scores, test_benign_scores, threshold)
    verify_info = verify_watermark(test_trigger_scores, threshold)
    qa_proxy_accuracy = classifier.classification_accuracy(splits["test"], batch_size=batch_size, max_length=max_length)

    classifier.save(str(run_dir))

    with (run_dir / "signature.json").open("w", encoding="utf-8") as f:
        json.dump(signature, f)
    with (run_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "threshold_f1": threshold_info["f1"]}, f, indent=2)

    return {
        **wm,
        "threshold": float(threshold),
        "threshold_f1": float(threshold_info["f1"]),
        "verification_decision": bool(verify_info["decision"]),
        "number_of_train_samples": len(splits["train"]),
        "number_of_validation_samples": len(splits["validation"]),
        "number_of_test_samples": len(splits["test"]),
        "number_of_triggers": len(triggers),
        "seed": int(config["experiment"]["seed"]),
        "model_name": model_name,
        "dataset_name": str(config.get("dataset", {}).get("name", "squad_v2")),
        "device": device,
        "train_loss": float(train_summary["train_loss"]),
        "validation_accuracy_proxy": float(train_summary["validation_accuracy"]),
        "qa_proxy_accuracy": float(qa_proxy_accuracy),
        "pipeline_mode": "verification_only",
        "task_type": "text_verification",
        "qa_metrics_available": False,
    }
