#!/usr/bin/env python
"""Train watermarked model in scaffold or real_full mode."""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_full_inputs import validate_manifest
from src.data.text_dataset import create_text_dataloaders
from src.models.text_model import SimpleTextWatermarkModel
from src.training.trainer import (
    train_real_full_distilbert_agnews,
    train_real_full_distilbert_squad_v2,
    train_real_full_mobilenetv2_cifar10,
    train_text_watermark,
)
from src.utils.config import load_config, save_config_snapshot
from src.utils.io import append_jsonl, prepare_real_full_run_dir, prepare_run_dir, write_json, write_metrics_csv
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


def _run_real_full(config: dict) -> None:
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

    set_global_seed(config["experiment"]["seed"], config.get("reproducibility", {}).get("deterministic", True))

    output_dir = str(config["experiment"]["output_dir"])
    run_dir = prepare_real_full_run_dir(output_dir)

    save_config_snapshot(config, run_dir / "config_snapshot.yaml")
    write_json(
        {
            "run_started_utc": datetime.datetime.utcnow().isoformat(),
            "execution_mode": "real_full",
            "task": config.get("experiment", {}).get("task", "unknown"),
        },
        run_dir / "run_metadata.json",
    )

    try:
        if dataset_name == "ag_news" and "distilbert" in model_name:
            metrics = train_real_full_distilbert_agnews(config=config, run_dir=run_dir)
        elif dataset_name == "squad_v2" and "distilbert" in model_name:
            metrics = train_real_full_distilbert_squad_v2(config=config, run_dir=run_dir)
        elif dataset_name == "cifar10" and "mobilenetv2" in model_name:
            metrics = train_real_full_mobilenetv2_cifar10(config=config, run_dir=run_dir)
        elif dataset_name == "flickr30k" and "clip" in model_name:
            raise SystemExit(
                "Flickr30K+CLIP stage is verification-only in this repository stage. "
                "Use scripts/eval.py --config configs/full_real_clip_flickr30k.example.yaml"
            )
        elif dataset_name == "flickr30k" and "vilt" in model_name:
            raise SystemExit(
                "Flickr30K+ViLT stage is verification-only in this repository stage. "
                "Use scripts/eval.py --config configs/full_real_vilt_flickr30k.example.yaml"
            )
        else:
            raise SystemExit(
                "Unsupported real_full configuration. Supported combinations: "
                "(dataset=ag_news, model=distilbert*), (dataset=squad_v2, model=distilbert* verification-focused), "
                "and (dataset=cifar10, model=mobilenetv2*). "
                "For (dataset=flickr30k, model=clip*) or (dataset=flickr30k, model=vilt*) use eval.py verification-only path."
            )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"real_full data/trigger loading failed: {exc}") from exc

    write_json(metrics, run_dir / "metrics.json")
    write_json(metrics, run_dir / "metrics" / "train_metrics.json")
    write_metrics_csv(metrics, run_dir / "metrics.csv")
    write_json({"threshold": metrics["threshold"], "threshold_f1": metrics.get("threshold_f1")}, run_dir / "threshold.json")

    with (run_dir / "training_log.txt").open("w", encoding="utf-8") as f:
        f.write("real_full training completed\n")
        f.write(f"dataset={dataset_name} model={model_name}\n")
        f.write(f"metrics_path={run_dir / 'metrics.json'}\n")

    append_jsonl(metrics, run_dir / "logs" / "events.jsonl")
    print(f"Run complete. Outputs: {run_dir}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    mode = config.get("experiment", {}).get("execution_mode", "scaffold_smoke")
    if mode == "real_full":
        _run_real_full(config)
        return

    set_global_seed(config["experiment"]["seed"], config["reproducibility"]["deterministic"])

    run_dir = prepare_run_dir(config["experiment"]["output_dir"], config["experiment"]["name"])
    save_config_snapshot(config, run_dir / "config_snapshot.json")

    train_loader, val_loader, _ = create_text_dataloaders(config)

    model = SimpleTextWatermarkModel(
        vocab_size=config["preprocessing"]["vocab_size"],
        embedding_dim=config["model"]["embedding_dim"],
        num_labels=config["model"]["num_labels"],
    )

    results = train_text_watermark(
        model=model,
        optimizer=None,
        criterion=None,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cpu",
        run_dir=Path(run_dir),
    )

    write_json(results, Path(run_dir) / "metrics" / "train_metrics.json")
    append_jsonl(results, Path(run_dir) / "logs" / "events.jsonl")
    print(f"Run complete. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
