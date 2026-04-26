#!/usr/bin/env python
"""Generate baseline artifacts from frozen results or local full-mode runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results.full_pipeline import build_table5_from_runs
from src.results.reference import ensure_output_dirs, load_paper_results, write_csv, write_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--mode", choices=["frozen", "full"], default="frozen")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")
    if args.mode == "full":
        rows = build_table5_from_runs("outputs")
        write_csv(out["metrics"] / "table5_internal_baselines.csv", rows)
        write_log(out["logs"] / "run_baselines.json", {"mode": "full", "executed": True, "rows": len(rows)})
        print("Baseline artifacts generated from local full-mode runs.")
        return

    ref = load_paper_results(args.paper_results)
    rows = ref.get("tables", {}).get("table5_internal_baselines", [])
    write_csv(out["metrics"] / "table5_internal_baselines.csv", rows)
    write_log(out["logs"] / "run_baselines.json", {"mode": "frozen", "message": "Frozen baseline artifact regeneration only."})
    print("Baseline artifacts regenerated from frozen references.")


if __name__ == "__main__":
    main()
