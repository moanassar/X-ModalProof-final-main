#!/usr/bin/env python
"""Export ONNX artifacts from frozen references, scaffold full mode, or real_full run directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.onnx_export import export_real_full_onnx
from src.results.reference import ensure_output_dirs, write_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--mode", choices=["frozen", "full", "real_full"], default="frozen")
    parser.add_argument("--output-dir", default="outputs/full_real/deployment/onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")

    if args.mode == "real_full":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --mode real_full")
        try:
            summary = export_real_full_onnx(
                run_dir=Path(args.run_dir),
                output_root=Path(args.output_dir),
                opset=int(args.opset),
                validate=bool(args.validate),
            )
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"real_full ONNX export failed: {exc}") from exc

        write_log(
            out["logs"] / "export_onnx.json",
            {
                "mode": "real_full",
                "run_dir": args.run_dir,
                "onnx_path": summary.get("onnx_path"),
                "export_success": True,
            },
        )
        print(f"ONNX export completed: {summary['onnx_path']}")
        return

    if args.mode == "full":
        # Keep scaffold/full mode behavior explicit: no fake ONNX export.
        write_log(
            out["logs"] / "export_onnx.json",
            {
                "mode": "full",
                "executed": False,
                "message": "No real ONNX export in scaffold full mode. Use --mode real_full with a real_full run-dir.",
            },
        )
        print("Scaffold full mode does not export real ONNX. Use --mode real_full.")
        return

    write_log(
        out["logs"] / "export_onnx.json",
        {
            "mode": "frozen_paper_reported",
            "paper_results": args.paper_results,
            "exported": False,
            "message": "No ONNX model exported in frozen-results mode.",
        },
    )
    print("Wrote ONNX export manifest for frozen-results mode.")


if __name__ == "__main__":
    main()
