#!/usr/bin/env python
"""Regenerate tables from frozen references, scaffold full-mode artifacts, or local real_full outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results.full_pipeline import build_table5_from_runs
from src.results.real_full_artifacts import build_real_full_tables, write_artifact_tracking
from src.results.reference import ensure_output_dirs, load_paper_results, write_csv, write_log

TABLE_FILES = {
    "table3_detection_accuracy": "table3_detection_accuracy.csv",
    "table4_attribution_alignment": "table4_attribution_alignment.csv",
    "table5_internal_baselines": "table5_internal_baselines.csv",
    "table6_threshold_sensitivity": "table6_threshold_sensitivity.csv",
    "table7_trigger_size": "table7_trigger_size.csv",
    "table8_repeatability": "table8_repeatability.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--mode", choices=["frozen", "full", "real_full"], default="frozen")
    parser.add_argument("--outputs-root", default="outputs/full_real")
    parser.add_argument("--output-dir", default="outputs/full_real/artifacts")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--include-missing", action="store_true", default=False)
    parser.add_argument("--source-mode", default="real_full")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")
    if args.mode == "real_full":
        summary = build_real_full_tables(
            outputs_root=args.outputs_root,
            output_dir=args.output_dir,
            strict=bool(args.strict),
            include_missing=bool(args.include_missing),
            source_mode=str(args.source_mode),
        )
        write_artifact_tracking(
            output_dir=args.output_dir,
            source_entries=summary["sources_map"],
            source_mode=str(args.source_mode),
            log_message="real_full tables generated from local validated outputs",
        )
        write_log(
            out["logs"] / "make_tables.json",
            {
                "mode": "real_full",
                "outputs_root": args.outputs_root,
                "output_dir": args.output_dir,
                "strict": bool(args.strict),
                "include_missing": bool(args.include_missing),
                "source_mode": str(args.source_mode),
                "generated_tables": len(summary["sources_map"]),
            },
        )
        print(f"Tables generated from real_full outputs: {args.outputs_root}")
        return

    if args.mode == "full":
        table5_rows = build_table5_from_runs("outputs")
        write_csv(out["metrics"] / "table5_internal_baselines.csv", table5_rows)

        for key, filename in TABLE_FILES.items():
            path = out["metrics"] / filename
            if not path.exists():
                write_csv(path, [])

        write_log(out["logs"] / "make_tables.json", {"mode": "full", "executed": True, "table5_rows": len(table5_rows)})
        print("Tables generated from local full-mode artifacts.")
        return

    ref = load_paper_results(args.paper_results)
    for key, filename in TABLE_FILES.items():
        rows = ref.get("tables", {}).get(key, [])
        write_csv(out["metrics"] / filename, rows)

    write_log(out["logs"] / "make_tables.json", {"mode": ref.get("metadata", {}).get("mode", "unknown"), "paper_results": args.paper_results, "message": "Tables regenerated from frozen references only."})
    print("Regenerated table artifacts from frozen paper results.")


if __name__ == "__main__":
    main()
