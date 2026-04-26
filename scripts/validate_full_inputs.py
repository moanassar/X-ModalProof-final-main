#!/usr/bin/env python
"""Validate local full-paper input contracts without fabricating data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def _check_path(path: str, errors: list[str], label: str) -> None:
    p = Path(path)
    if not p.exists():
        errors.append(f"missing {label}: {path}")


def validate_manifest(manifest_path: str, check_datasets: bool, check_processed: bool, check_runs: bool, check_explainability: bool, check_deployment: bool) -> list[str]:
    errors: list[str] = []
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        return [f"missing manifest: {manifest_path}"]
    manifest = load_config(manifest_path)

    if check_datasets:
        for key, p in manifest.get("raw", {}).items():
            _check_path(p, errors, f"raw.{key}")

    if check_processed:
        for key, p in manifest.get("processed", {}).items():
            _check_path(p, errors, f"processed.{key}")

    full_real_root = manifest.get("outputs", {}).get("full_real_root", "outputs/full_real")
    if check_runs:
        _check_path(full_real_root, errors, "outputs.full_real_root")

    if check_explainability:
        for sub in ["explainability/shap", "explainability/gradcam", "explainability/scorecam", "explainability/attention_rollout"]:
            _check_path(str(Path(full_real_root) / sub), errors, f"outputs.{sub}")

    if check_deployment:
        for sub in ["deployment/onnx", "deployment/tensorrt", "deployment/latency"]:
            _check_path(str(Path(full_real_root) / sub), errors, f"outputs.{sub}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/datasets_manifest.yaml")
    parser.add_argument("--check-datasets", action="store_true")
    parser.add_argument("--check-processed", action="store_true")
    parser.add_argument("--check-runs", action="store_true")
    parser.add_argument("--check-explainability", action="store_true")
    parser.add_argument("--check-deployment", action="store_true")
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()

    if args.check_all:
        args.check_datasets = args.check_processed = args.check_runs = args.check_explainability = args.check_deployment = True

    if not any([args.check_datasets, args.check_processed, args.check_runs, args.check_explainability, args.check_deployment]):
        print("No checks selected. Use --check-all or specific flags.")
        raise SystemExit(2)

    errors = validate_manifest(
        manifest_path=args.manifest,
        check_datasets=args.check_datasets,
        check_processed=args.check_processed,
        check_runs=args.check_runs,
        check_explainability=args.check_explainability,
        check_deployment=args.check_deployment,
    )

    if errors:
        print("Validation failed:")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(1)

    print("Validation passed: required full inputs exist.")


if __name__ == "__main__":
    main()
