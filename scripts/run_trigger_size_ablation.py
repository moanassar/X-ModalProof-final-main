#!/usr/bin/env python
"""Generate trigger-size ablation artifact from frozen results or validated real local CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results.reference import ensure_output_dirs, load_paper_results, write_csv, write_log


def _load_real_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing real trigger-size CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Real trigger-size CSV is empty: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--mode", choices=["frozen", "full"], default="frozen")
    parser.add_argument("--source-csv", default="outputs/full_real/table7_trigger_size.csv")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")
    if args.mode == "full":
        rows = _load_real_csv(Path(args.source_csv))
        write_csv(out["metrics"] / "table7_trigger_size.csv", rows)
        write_log(out["logs"] / "run_trigger_size_ablation.json", {"mode": "full", "executed": True, "source_csv": args.source_csv, "rows": len(rows)})
        print("Trigger-size artifacts loaded from local real outputs.")
        return

    ref = load_paper_results(args.paper_results)
    rows = ref.get("tables", {}).get("table7_trigger_size", [])
    write_csv(out["metrics"] / "table7_trigger_size.csv", rows)
    write_log(out["logs"] / "run_trigger_size_ablation.json", {"mode": "frozen", "message": "Trigger-size artifacts regenerated from frozen references."})
    print("Trigger-size artifacts regenerated from frozen references.")


if __name__ == "__main__":
    main()
