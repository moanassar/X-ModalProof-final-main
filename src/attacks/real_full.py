"""Real attack and robustness evaluation for implemented real_full pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.image_dataset import load_real_cifar10_splits_from_manifest
from src.data.text_dataset import load_real_ag_news_splits_from_manifest
from src.data.triggers import build_triggered_image_rows, build_triggered_text_rows, load_image_triggers_from_manifest, load_text_triggers_from_manifest
from src.models.image_model import MobileNetV2ImageClassifier, resolve_image_device
from src.models.text_model import DistilBertTextClassifier, resolve_device
from src.utils.io import write_json, write_metrics_csv
from src.watermark.signature import cosine_scores


def _load_required_run_artifacts(run_dir: Path) -> tuple[Dict[str, Any], List[float], float]:
    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config snapshot: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    sig_path = run_dir / "signature.json"
    if not sig_path.exists():
        raise FileNotFoundError(f"missing signature: {sig_path}")
    signature = json.loads(sig_path.read_text(encoding="utf-8"))

    th_path = run_dir / "threshold.json"
    if not th_path.exists():
        raise FileNotFoundError(f"missing threshold: {th_path}")
    threshold = float(json.loads(th_path.read_text(encoding="utf-8"))["threshold"])

    manifest_path = str(cfg.get("manifest", {}).get("path", ""))
    if not manifest_path:
        raise ValueError("run config missing manifest.path")

    return cfg, signature, threshold


def _infer_target(cfg: Dict[str, Any]) -> str:
    dataset = str(cfg.get("dataset", {}).get("name", "")).lower()
    model = str(cfg.get("model", {}).get("name", "")).lower()
    if dataset == "ag_news" and "distilbert" in model:
        return "text"
    if dataset == "cifar10" and "mobilenetv2" in model:
        return "image"
    if dataset == "flickr30k" and "clip" in model:
        return "clip"
    if dataset == "flickr30k" and "vilt" in model:
        return "vilt"
    return "unsupported"


def _stats(trigger_scores: List[float], benign_scores: List[float], threshold: float) -> Dict[str, float]:
    wsr = sum(1 for s in trigger_scores if s >= threshold) / max(1, len(trigger_scores))
    fpr = sum(1 for s in benign_scores if s >= threshold) / max(1, len(benign_scores))
    fnr = sum(1 for s in trigger_scores if s < threshold) / max(1, len(trigger_scores))
    return {
        "watermark_success_rate": float(wsr),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "mean_trigger_similarity": float(sum(trigger_scores) / max(1, len(trigger_scores))),
        "mean_benign_similarity": float(sum(benign_scores) / max(1, len(benign_scores))),
    }


def _text_scores(cfg: Dict[str, Any], model: DistilBertTextClassifier, signature: List[float], threshold: float) -> Dict[str, float]:
    manifest_path = str(cfg["manifest"]["path"])
    splits = load_real_ag_news_splits_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=cfg.get("training", {}).get("max_train_samples"),
        max_eval_samples=cfg.get("training", {}).get("max_eval_samples"),
    )
    triggers = load_text_triggers_from_manifest(manifest_path)
    strategy = cfg.get("triggers", {}).get("injection_strategy", "append")
    triggered_test = build_triggered_text_rows(splits["test"], triggers, strategy=strategy)

    batch_size = int(cfg.get("training", {}).get("batch_size", 8))
    max_length = int(cfg.get("training", {}).get("max_length", 128))

    trigger_scores = cosine_scores(model.extract_embeddings(triggered_test, batch_size=batch_size, max_length=max_length), signature)
    benign_scores = cosine_scores(model.extract_embeddings(splits["test"], batch_size=batch_size, max_length=max_length), signature)
    base = _stats(trigger_scores, benign_scores, threshold)
    base["classification_accuracy"] = float(model.classification_accuracy(splits["test"], batch_size=batch_size, max_length=max_length))
    return base


def _image_scores(cfg: Dict[str, Any], model: MobileNetV2ImageClassifier, signature: List[float], threshold: float) -> Dict[str, float]:
    manifest_path = str(cfg["manifest"]["path"])
    splits = load_real_cifar10_splits_from_manifest(
        manifest_path=manifest_path,
        max_train_samples=cfg.get("training", {}).get("max_train_samples"),
        max_eval_samples=cfg.get("training", {}).get("max_eval_samples"),
    )
    triggers = load_image_triggers_from_manifest(manifest_path)
    triggered_test = build_triggered_image_rows(splits["test"], triggers)

    batch_size = int(cfg.get("training", {}).get("batch_size", 8))

    trigger_scores = cosine_scores(model.extract_embeddings(triggered_test, batch_size=batch_size), signature)
    benign_scores = cosine_scores(model.extract_embeddings(splits["test"], batch_size=batch_size), signature)
    base = _stats(trigger_scores, benign_scores, threshold)
    base["classification_accuracy"] = float(model.classification_accuracy(splits["test"], batch_size=batch_size))
    return base


def _apply_magnitude_pruning(torch_mod, nn_model, strength: float) -> None:
    strength = min(max(strength, 0.0), 0.95)
    params = [p for p in nn_model.parameters() if getattr(p, "requires_grad", False) and p.data.numel() > 0]
    if not params:
        return
    all_abs = torch_mod.cat([p.data.detach().abs().reshape(-1) for p in params])
    k = int(strength * all_abs.numel())
    if k <= 0:
        return
    threshold = float(torch_mod.kthvalue(all_abs, k).values.item())
    for p in params:
        mask = (p.data.abs() > threshold).to(p.data.dtype)
        p.data.mul_(mask)


def _text_predict_labels(model: DistilBertTextClassifier, rows: List[Dict[str, Any]], batch_size: int, max_length: int) -> List[int]:
    loader = model._make_loader(rows, batch_size=batch_size, max_length=max_length, shuffle=False)
    preds: List[int] = []
    model.model.eval()
    with model.torch.no_grad():
        for input_ids, attention_mask, _labels in loader:
            logits = model.model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device)).logits
            preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return [int(p) for p in preds]


def _image_predict_labels(model: MobileNetV2ImageClassifier, rows: List[Dict[str, Any]], batch_size: int) -> List[int]:
    loader = model._make_loader(rows, batch_size=batch_size, shuffle=False)
    preds: List[int] = []
    model.model.eval()
    with model.torch.no_grad():
        for images, _labels in loader:
            logits = model.model(images.to(model.device))
            preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return [int(p) for p in preds]


def run_real_full_attack(
    run_dir: Path,
    attack: str,
    strength: float,
    finetune_epochs: int,
    distill_epochs: int,
    student_model: str,
    output_root: Path,
) -> Dict[str, Any]:
    cfg, signature, threshold = _load_required_run_artifacts(run_dir)
    target = _infer_target(cfg)
    if target == "unsupported":
        raise ValueError("Unsupported run_dir config for attacks")

    model_name = str(cfg.get("model", {}).get("name", ""))
    dataset_name = str(cfg.get("dataset", {}).get("name", ""))
    seed = int(cfg.get("experiment", {}).get("seed", 0))

    attack_dir = output_root / attack / run_dir.name
    attack_dir.mkdir(parents=True, exist_ok=True)

    # Verification-only multimodal models are unsupported for active model mutation attacks in this stage.
    if target in {"clip", "vilt"}:
        raise ValueError(
            f"Unsupported attack/model combination: attack={attack} is not implemented for verification-only {target.upper()} pipeline"
        )

    if target == "text":
        device = resolve_device(str(cfg.get("training", {}).get("device", "auto")))
        clean_model = DistilBertTextClassifier.load(str(run_dir), device=device)
        clean_stats = _text_scores(cfg, clean_model, signature, threshold)

        attacked_model = DistilBertTextClassifier.load(str(run_dir), device=device)
        if attack == "pruning":
            _apply_magnitude_pruning(attacked_model.torch, attacked_model.model, strength)
            attack_cfg: Dict[str, Any] = {"strength": strength, "strategy": "global_magnitude"}
        elif attack == "finetuning":
            splits = load_real_ag_news_splits_from_manifest(
                str(cfg["manifest"]["path"]),
                max_train_samples=cfg.get("training", {}).get("max_train_samples"),
                max_eval_samples=cfg.get("training", {}).get("max_eval_samples"),
            )
            attacked_model.train_epochs(
                splits["train"],
                splits["validation"],
                epochs=max(1, finetune_epochs),
                batch_size=int(cfg.get("training", {}).get("batch_size", 8)),
                learning_rate=float(cfg.get("training", {}).get("learning_rate", 2e-5)),
                max_length=int(cfg.get("training", {}).get("max_length", 128)),
            )
            attack_cfg = {"epochs": max(1, finetune_epochs), "mode": "same_task_local"}
        elif attack == "distillation":
            splits = load_real_ag_news_splits_from_manifest(
                str(cfg["manifest"]["path"]),
                max_train_samples=cfg.get("training", {}).get("max_train_samples"),
                max_eval_samples=cfg.get("training", {}).get("max_eval_samples"),
            )
            bsz = int(cfg.get("training", {}).get("batch_size", 8))
            mlen = int(cfg.get("training", {}).get("max_length", 128))
            teacher_preds = _text_predict_labels(clean_model, splits["train"], batch_size=bsz, max_length=mlen)
            pseudo_train = [dict(r, label=int(y)) for r, y in zip(splits["train"], teacher_preds)]

            student_name = student_model or model_name
            attacked_model = DistilBertTextClassifier(model_name=student_name, num_labels=4, device=device)
            attacked_model.train_epochs(
                pseudo_train,
                splits["validation"],
                epochs=max(1, distill_epochs),
                batch_size=bsz,
                learning_rate=float(cfg.get("training", {}).get("learning_rate", 2e-5)),
                max_length=mlen,
            )
            attack_cfg = {"epochs": max(1, distill_epochs), "student_model": student_name, "teacher_run_dir": str(run_dir)}
        else:
            raise ValueError(f"unsupported attack: {attack}")

        attacked_stats = _text_scores(cfg, attacked_model, signature, threshold)

    elif target == "image":
        device = resolve_image_device(str(cfg.get("training", {}).get("device", "auto")))
        clean_model = MobileNetV2ImageClassifier.load(str(run_dir), device=device)
        clean_stats = _image_scores(cfg, clean_model, signature, threshold)

        attacked_model = MobileNetV2ImageClassifier.load(str(run_dir), device=device)
        if attack == "pruning":
            _apply_magnitude_pruning(attacked_model.torch, attacked_model.model, strength)
            attack_cfg = {"strength": strength, "strategy": "global_magnitude"}
        elif attack == "finetuning":
            splits = load_real_cifar10_splits_from_manifest(
                str(cfg["manifest"]["path"]),
                max_train_samples=cfg.get("training", {}).get("max_train_samples"),
                max_eval_samples=cfg.get("training", {}).get("max_eval_samples"),
            )
            attacked_model.train_epochs(
                splits["train"],
                splits["validation"],
                epochs=max(1, finetune_epochs),
                batch_size=int(cfg.get("training", {}).get("batch_size", 8)),
                learning_rate=float(cfg.get("training", {}).get("learning_rate", 1e-4)),
            )
            attack_cfg = {"epochs": max(1, finetune_epochs), "mode": "same_task_local"}
        elif attack == "distillation":
            splits = load_real_cifar10_splits_from_manifest(
                str(cfg["manifest"]["path"]),
                max_train_samples=cfg.get("training", {}).get("max_train_samples"),
                max_eval_samples=cfg.get("training", {}).get("max_eval_samples"),
            )
            bsz = int(cfg.get("training", {}).get("batch_size", 8))
            teacher_preds = _image_predict_labels(clean_model, splits["train"], batch_size=bsz)
            pseudo_train = [dict(r, label=int(y)) for r, y in zip(splits["train"], teacher_preds)]
            attacked_model = MobileNetV2ImageClassifier(num_labels=int(cfg.get("model", {}).get("num_labels", 10)), device=device, image_size=int(cfg.get("training", {}).get("image_size", 224)))
            attacked_model.train_epochs(
                pseudo_train,
                splits["validation"],
                epochs=max(1, distill_epochs),
                batch_size=bsz,
                learning_rate=float(cfg.get("training", {}).get("learning_rate", 1e-4)),
            )
            attack_cfg = {"epochs": max(1, distill_epochs), "student_model": student_model or "mobilenet_v2_student", "teacher_run_dir": str(run_dir)}
        else:
            raise ValueError(f"unsupported attack: {attack}")

        attacked_stats = _image_scores(cfg, attacked_model, signature, threshold)
    else:
        raise ValueError("unsupported target")

    clean_wsr = float(clean_stats["watermark_success_rate"])
    attacked_wsr = float(attacked_stats["watermark_success_rate"])
    robustness_drop = clean_wsr - attacked_wsr
    retained = attacked_wsr / clean_wsr if clean_wsr > 0 else 0.0

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "attack_type": attack,
        "attack_config": attack_cfg,
        "attack_strength": float(strength),
        "seed": seed,
        "clean_watermark_success_rate": clean_wsr,
        "attacked_watermark_success_rate": attacked_wsr,
        "clean_false_positive_rate": float(clean_stats["false_positive_rate"]),
        "attacked_false_positive_rate": float(attacked_stats["false_positive_rate"]),
        "clean_false_negative_rate": float(clean_stats["false_negative_rate"]),
        "attacked_false_negative_rate": float(attacked_stats["false_negative_rate"]),
        "clean_mean_trigger_similarity": float(clean_stats["mean_trigger_similarity"]),
        "attacked_mean_trigger_similarity": float(attacked_stats["mean_trigger_similarity"]),
        "clean_mean_benign_similarity": float(clean_stats["mean_benign_similarity"]),
        "attacked_mean_benign_similarity": float(attacked_stats["mean_benign_similarity"]),
        "robustness_drop": float(robustness_drop),
        "retained_robustness": float(retained),
        "notes": "real_full local attack evaluation; no frozen paper results; no synthetic fallback",
    }
    if "classification_accuracy" in clean_stats:
        metrics["clean_classification_accuracy"] = float(clean_stats["classification_accuracy"])
    if "classification_accuracy" in attacked_stats:
        metrics["attacked_classification_accuracy"] = float(attacked_stats["classification_accuracy"])

    write_json(metrics, attack_dir / "attack_metrics.json")
    write_metrics_csv(metrics, attack_dir / "attack_metrics.csv")
    write_json(cfg, attack_dir / "config_snapshot.yaml")
    with (attack_dir / "attack_log.txt").open("w", encoding="utf-8") as f:
        f.write("real_full attack evaluation complete\n")
        f.write(f"attack={attack} target={target} run_dir={run_dir}\n")
    return metrics
