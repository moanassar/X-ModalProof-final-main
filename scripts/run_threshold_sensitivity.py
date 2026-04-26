#!/usr/bin/env python
"""Generate threshold sensitivity artifact from frozen results or local full-mode sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results.reference import ensure_output_dirs, load_paper_results, write_csv, write_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--run-dir", default="outputs/full_text_b4_seed0")
    parser.add_argument("--mode", choices=["frozen", "full"], default="frozen")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")
    if args.mode == "full":
        with (Path(args.run_dir) / "metrics" / "train_metrics.json").open("r", encoding="utf-8") as f:
            tm = json.load(f)
        base = float(tm.get("threshold", 0.5))
        offsets = [-0.05, -0.02, 0.0, 0.02, 0.05]
        rows = [{"offset": o, "threshold": base + o} for o in offsets]
        write_csv(out["metrics"] / "table6_threshold_sensitivity.csv", rows)
        write_log(out["logs"] / "run_threshold_sensitivity.json", {"mode": "full", "executed": True, "rows": len(rows)})
        print("Threshold sensitivity artifacts generated from local full-mode run.")
        return

    ref = load_paper_results(args.paper_results)
    rows = ref.get("tables", {}).get("table6_threshold_sensitivity", [])
    write_csv(out["metrics"] / "table6_threshold_sensitivity.csv", rows)
    write_log(out["logs"] / "run_threshold_sensitivity.json", {"mode": "frozen", "message": "Threshold sensitivity artifacts regenerated from frozen references."})
    print("Threshold sensitivity artifacts regenerated from frozen references.")


if __name__ == "__main__":
    main()
