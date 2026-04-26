#!/usr/bin/env python
"""Generate latency artifacts from frozen references, scaffold full mode, or real_full local benchmarking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.latency import benchmark_real_full_latency
from src.results.full_pipeline import benchmark_latency
from src.results.reference import ensure_output_dirs, load_paper_results, write_csv, write_log, write_text_figure
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--mode", choices=["frozen", "full", "real_full"], default="frozen")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--backend", choices=["pytorch", "onnxruntime", "tensorrt"], default="pytorch")
    parser.add_argument("--hardware", default="local_cpu")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measured-runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/full_real/deployment/latency")
    parser.add_argument("--onnx", default="")
    parser.add_argument("--engine", default="")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")

    if args.mode == "real_full":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --mode real_full")
        try:
            metrics = benchmark_real_full_latency(
                run_dir=Path(args.run_dir),
                backend=args.backend,
                hardware_label=args.hardware,
                output_root=Path(args.output_dir),
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                batch_size=int(args.batch_size),
                onnx_path=str(args.onnx),
                engine_path=str(args.engine),
            )
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            write_log(
                out["logs"] / "benchmark_latency.json",
                {
                    "mode": "real_full",
                    "run_dir": args.run_dir,
                    "backend": args.backend,
                    "hardware": args.hardware,
                    "success": False,
                    "error": str(exc),
                },
            )
            raise SystemExit(f"real_full latency benchmark failed: {exc}") from exc

        write_log(
            out["logs"] / "benchmark_latency.json",
            {
                "mode": "real_full",
                "run_dir": args.run_dir,
                "backend": args.backend,
                "hardware": args.hardware,
                "success": True,
                "mean_latency_ms": metrics.get("mean_latency_ms"),
            },
        )
        print(f"real_full latency benchmark completed: {args.run_dir} ({args.backend})")
        return

    if args.mode == "full":
        cfg = load_config(args.config)
        lat = benchmark_latency(cfg, repeats=100)
        write_csv(out["metrics"] / "latency_local_full.csv", [lat])
        write_text_figure(out["figures"] / "figure7_latency.txt", "figure7_latency", "latency_local_full.csv")
        write_log(out["logs"] / "benchmark_latency.json", {"mode": "full", "executed": True, **lat})
        print("Latency artifacts generated from local full-mode benchmark.")
        return

    ref = load_paper_results(args.paper_results)
    rows = ref.get("tables", {}).get("table5_internal_baselines", [])
    write_csv(out["metrics"] / "table5_internal_baselines.csv", rows)
    write_text_figure(out["figures"] / "figure7_latency.txt", "figure7_latency", "table5_internal_baselines")
    write_log(out["logs"] / "benchmark_latency.json", {"mode": "frozen", "message": "Latency artifacts regenerated from frozen references."})
    print("Latency artifacts regenerated from frozen references.")


if __name__ == "__main__":
    main()
