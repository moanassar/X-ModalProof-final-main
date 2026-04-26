#!/usr/bin/env python
"""Evaluate watermark decision in scaffold or real_full mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_full_inputs import validate_manifest
from src.data.text_dataset import create_text_dataloaders
from src.evaluation.real_full_clip import evaluate_real_full_clip_flickr30k
from src.evaluation.real_full_image import evaluate_real_full_mobilenetv2_cifar10
from src.evaluation.real_full_vilt import evaluate_real_full_vilt_flickr30k
from src.evaluation.real_full_text import evaluate_real_full_distilbert_agnews, evaluate_real_full_distilbert_squad_v2
from src.evaluation.verification import verify_watermark
from src.models.text_model import SimpleTextWatermarkModel
from src.utils.config import load_config
from src.utils.io import write_json, write_metrics_csv
from src.watermark.signature import cosine_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--run-dir", required=False, type=str)
    return parser.parse_args()


def _run_real_full_eval(config: dict, run_dir: Path) -> None:
    dataset_name = str(config.get("dataset", {}).get("name", "")).lower()
    model_name = str(config.get("model", {}).get("name", "")).lower()

    manifest_path = config.get("manifest", {}).get("path")
    if not manifest_path:
        raise SystemExit("real_full mode requires manifest.path in config")
    errors = validate_manifest(
        manifest_path=manifest_path,
        check_datasets=False,
        check_processed=False,
        check_runs=False,
        check_explainability=False,
        check_deployment=False,
    )
    if errors:
        raise SystemExit("real_full validation failed:\n" + "\n".join(errors))

    try:
        if dataset_name == "ag_news" and "distilbert" in model_name:
            metrics = evaluate_real_full_distilbert_agnews(config=config, run_dir=run_dir)
        elif dataset_name == "squad_v2" and "distilbert" in model_name:
            metrics = evaluate_real_full_distilbert_squad_v2(config=config, run_dir=run_dir)
        elif dataset_name == "cifar10" and "mobilenetv2" in model_name:
            metrics = evaluate_real_full_mobilenetv2_cifar10(config=config, run_dir=run_dir)
        elif dataset_name == "flickr30k" and "clip" in model_name:
            metrics = evaluate_real_full_clip_flickr30k(config=config, run_dir=run_dir)
        elif dataset_name == "flickr30k" and "vilt" in model_name:
            metrics = evaluate_real_full_vilt_flickr30k(config=config, run_dir=run_dir)
        else:
            raise SystemExit(
                "Unsupported real_full configuration. Supported combinations: "
                "(dataset=ag_news, model=distilbert*), (dataset=squad_v2, model=distilbert* verification-focused), "
                "(dataset=cifar10, model=mobilenetv2*), "
                "(dataset=flickr30k, model=clip*), (dataset=flickr30k, model=vilt*)."
            )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"real_full evaluation failed: {exc}") from exc

    metrics_path = run_dir / "metrics.json"
    merged = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            merged.update(json.load(f))
    merged.update(metrics)

    write_json(merged, metrics_path)
    write_metrics_csv(merged, run_dir / "metrics.csv")
    write_json(merged, run_dir / "metrics" / "eval_metrics.json")
    with (run_dir / "eval_log.txt").open("w", encoding="utf-8") as f:
        f.write("real_full evaluation completed\n")
        f.write(f"dataset={dataset_name} model={model_name}\n")
        f.write(f"metrics_path={metrics_path}\n")
    print(f"Evaluation complete. Outputs: {metrics_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = config.get("experiment", {}).get("execution_mode", "scaffold_smoke")

    if mode == "real_full":
        run_dir = Path(args.run_dir or config["experiment"]["output_dir"])
        _run_real_full_eval(config, run_dir)
        return

    run_dir = Path(args.run_dir or (Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]))

    _, _, test_rows = create_text_dataloaders(config)
    model = SimpleTextWatermarkModel(
        vocab_size=config["preprocessing"]["vocab_size"],
        embedding_dim=config["model"]["embedding_dim"],
        num_labels=config["model"]["num_labels"],
    )

    with (run_dir / "checkpoints" / "model.pt").open("r", encoding="utf-8") as f:
        model.load_state_dict(json.load(f))
    with (run_dir / "signatures" / "signature.pt").open("r", encoding="utf-8") as f:
        signature = json.load(f)

    train_metrics = load_config(run_dir / "metrics" / "train_metrics.json")
    threshold = float(train_metrics["threshold"])

    trigger_scores, benign_scores = [], []
    for row in test_rows:
        emb = model.extract_embedding(row)
        score = cosine_scores([emb], signature)[0]
        if int(row["is_trigger"]) == 1:
            trigger_scores.append(score)
        else:
            benign_scores.append(score)

    verification = verify_watermark(trigger_scores, threshold)
    verification["mean_benign_score"] = sum(benign_scores) / max(1, len(benign_scores))
    verification["num_test_samples"] = len(test_rows)

    write_json(verification, run_dir / "metrics" / "eval_metrics.json")
    print(f"Evaluation complete. Outputs: {run_dir / 'metrics' / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
