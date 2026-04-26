#!/usr/bin/env python
"""Explainability runner for frozen/full reference mode and real_full per-run execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability.real_full import (
    _ensure_full_explain_dirs,
    run_image_cam,
    run_multimodal_attention_summary,
    run_text_token_attribution,
)
from src.results.reference import ensure_output_dirs, load_paper_results, write_csv, write_log, write_text_figure

TEXT_METHODS = {"shap", "captum_ig"}
IMAGE_METHODS = {"gradcam", "scorecam"}
MULTIMODAL_METHODS = {"attention_rollout"}


def _validate_real_explainability(source_dir: Path) -> list[dict]:
    required = {
        "shap": source_dir / "shap",
        "captum_ig": source_dir / "captum_ig",
        "gradcam": source_dir / "gradcam",
        "scorecam": source_dir / "scorecam",
        "attention_rollout": source_dir / "attention_rollout",
    }
    rows = []
    missing = []
    for k, p in required.items():
        exists = p.exists()
        rows.append({"artifact_group": k, "path": str(p), "exists": exists})
        if not exists:
            missing.append(str(p))
    if missing:
        raise SystemExit("Missing explainability artifacts for real full mode:\n" + "\n".join(missing))
    return rows


def _infer_run_target(run_dir: Path) -> str:
    """Infer explainability target from config snapshot first, then run-dir naming fallback."""
    cfg_path = run_dir / "config_snapshot.yaml"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            dataset_name = str(cfg.get("dataset", {}).get("name", "")).lower()
            model_name = str(cfg.get("model", {}).get("name", "")).lower()

            if dataset_name == "ag_news" and "distilbert" in model_name:
                return "text_distilbert_agnews"
            if dataset_name == "cifar10" and "mobilenet" in model_name:
                return "image_mobilenetv2_cifar10"
            if dataset_name == "flickr30k" and "clip" in model_name:
                return "multimodal_clip_flickr30k"
            if dataset_name == "flickr30k" and "vilt" in model_name:
                return "multimodal_vilt_flickr30k"
        except json.JSONDecodeError:
            # Keep fallback behavior for malformed/legacy snapshots.
            pass

    run_name = run_dir.name.lower()
    if "text_distilbert_agnews" in run_name:
        return "text_distilbert_agnews"
    if "image_mobilenetv2_cifar10" in run_name:
        return "image_mobilenetv2_cifar10"
    if "multimodal_clip_flickr30k" in run_name:
        return "multimodal_clip_flickr30k"
    if "multimodal_vilt_flickr30k" in run_name:
        return "multimodal_vilt_flickr30k"

    return "unsupported"


def _run_real_full_method(method: str, run_dir: Path) -> dict:
    if not run_dir.exists():
        raise SystemExit(f"run-dir does not exist: {run_dir}")
    if not (run_dir / "config_snapshot.yaml").exists():
        raise SystemExit(f"missing required config snapshot in run-dir: {run_dir / 'config_snapshot.yaml'}")

    run_target = _infer_run_target(run_dir)
    if run_target == "text_distilbert_agnews":
        if method not in TEXT_METHODS:
            raise SystemExit(
                "Unsupported method/model combination: text_distilbert_agnews supports shap or captum_ig"
            )
        return run_text_token_attribution(run_dir, method=method)

    if run_target == "image_mobilenetv2_cifar10":
        if method not in IMAGE_METHODS:
            raise SystemExit(
                "Unsupported method/model combination: image_mobilenetv2_cifar10 supports gradcam or scorecam"
            )
        return run_image_cam(run_dir, method=method)

    if run_target in {"multimodal_clip_flickr30k", "multimodal_vilt_flickr30k"}:
        if method not in MULTIMODAL_METHODS:
            raise SystemExit("Unsupported method/model combination: multimodal CLIP/ViLT supports attention_rollout")
        return run_multimodal_attention_summary(run_dir, method=method)

    raise SystemExit(
        "Unsupported run-dir for real_full explainability. Expected one of: "
        "text_distilbert_agnews_seed*, image_mobilenetv2_cifar10_seed*, "
        "multimodal_clip_flickr30k_seed*, multimodal_vilt_flickr30k_seed*."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--mode", choices=["frozen", "full", "real_full"], default="frozen")
    parser.add_argument("--source-dir", default="outputs/full_real/explainability")
    parser.add_argument("--method", default="attention_rollout")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")
    if args.mode == "real_full":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --mode real_full")
        explain_root = Path("outputs/full_real/explainability")
        _ensure_full_explain_dirs(explain_root)
        summary = _run_real_full_method(method=args.method, run_dir=Path(args.run_dir))

        log_path = explain_root / "explainability_log.txt"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"mode": "real_full", "method": args.method, "run_dir": args.run_dir}) + "\n")

        summary_index = explain_root / "latest_summary.json"
        summary_index.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Explainability complete. Summary generated for {args.run_dir} ({args.method}).")
        return

    if args.mode == "full":
        rows = _validate_real_explainability(Path(args.source_dir))
        write_csv(out["metrics"] / "table4_attribution_alignment.csv", rows)
        write_text_figure(out["figures"] / "figure5_shap_tokens.txt", "figure5_shap_tokens", args.source_dir)
        write_text_figure(out["figures"] / "figure6_gradcam_overlay.txt", "figure6_gradcam_overlay", args.source_dir)
        write_log(
            out["logs"] / "run_explainability.json",
            {"mode": "full", "executed": True, "validated": True, "rows": len(rows)},
        )
        print("Explainability artifacts validated from local real outputs.")
        return

    ref = load_paper_results(args.paper_results)
    rows = ref.get("tables", {}).get("table4_attribution_alignment", [])
    write_csv(out["metrics"] / "table4_attribution_alignment.csv", rows)
    write_text_figure(out["figures"] / "figure5_shap_tokens.txt", "figure5_shap_tokens", "table4_attribution_alignment")
    write_text_figure(out["figures"] / "figure6_gradcam_overlay.txt", "figure6_gradcam_overlay", "table4_attribution_alignment")
    write_log(
        out["logs"] / "run_explainability.json",
        {"mode": "frozen", "message": "Explainability artifacts regenerated from frozen references."},
    )
    print("Explainability artifacts regenerated from frozen references.")


if __name__ == "__main__":
    main()
