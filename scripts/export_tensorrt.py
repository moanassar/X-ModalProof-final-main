#!/usr/bin/env python
"""TensorRT export interface for real_full deployment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.tensorrt_export import export_tensorrt
from src.results.reference import ensure_output_dirs, write_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real_full"], default="real_full")
    parser.add_argument("--onnx", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", default="outputs/full_real/deployment/tensorrt")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace", type=int, default=1 << 30)
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")
    try:
        summary = export_tensorrt(
            onnx_path=str(args.onnx),
            run_dir=str(args.run_dir),
            output_root=Path(args.output_dir),
            fp16=bool(args.fp16),
            workspace_size=int(args.workspace),
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        write_log(
            out["logs"] / "export_tensorrt.json",
            {
                "mode": "real_full",
                "run_dir": args.run_dir,
                "onnx": args.onnx,
                "export_success": False,
                "error": str(exc),
            },
        )
        raise SystemExit(f"TensorRT export failed: {exc}") from exc

    write_log(
        out["logs"] / "export_tensorrt.json",
        {
            "mode": "real_full",
            "run_dir": args.run_dir,
            "onnx": args.onnx,
            "export_success": True,
            "engine_path": summary.get("output_engine_path"),
        },
    )
    print(f"TensorRT export completed: {summary['output_engine_path']}")


if __name__ == "__main__":
    main()
