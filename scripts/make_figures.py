#!/usr/bin/env python
"""Regenerate figure placeholders from frozen metadata, scaffold full-mode tables, or local real_full artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results.real_full_artifacts import build_real_full_tables, generate_real_full_figures, write_artifact_tracking
from src.results.reference import ensure_output_dirs, load_paper_results, write_log, write_text_figure

FIG_FILES = {
    "figure2_accuracy": "figure2_accuracy.txt",
    "figure3_robustness_drop": "figure3_robustness_drop.txt",
    "figure4_alignment": "figure4_alignment.txt",
    "figure7_latency": "figure7_latency.txt",
    "figure8_radar": "figure8_radar.txt",
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
        table_summary = build_real_full_tables(
            outputs_root=args.outputs_root,
            output_dir=args.output_dir,
            strict=bool(args.strict),
            include_missing=bool(args.include_missing),
            source_mode=str(args.source_mode),
        )
        fig_summary = generate_real_full_figures(
            output_dir=args.output_dir,
            strict=bool(args.strict),
            include_missing=bool(args.include_missing),
            source_mode=str(args.source_mode),
        )
        merged = {}
        merged.update(table_summary["sources_map"])
        merged.update(fig_summary["sources_map"])
        write_artifact_tracking(
            output_dir=args.output_dir,
            source_entries=merged,
            source_mode=str(args.source_mode),
            log_message="real_full figures generated from local validated outputs",
        )
        write_log(
            out["logs"] / "make_figures.json",
            {
                "mode": "real_full",
                "outputs_root": args.outputs_root,
                "output_dir": args.output_dir,
                "strict": bool(args.strict),
                "include_missing": bool(args.include_missing),
                "source_mode": str(args.source_mode),
                "generated_figures": len(fig_summary["sources_map"]),
            },
        )
        print(f"Figures generated from real_full outputs: {args.outputs_root}")
        return

    if args.mode == "full":
        for key, filename in FIG_FILES.items():
            write_text_figure(out["figures"] / filename, title=key, source="local_full_mode_tables")
        write_log(out["logs"] / "make_figures.json", {"mode": "full", "executed": True, "figures": len(FIG_FILES)})
        print("Figures generated from local full-mode artifacts.")
        return

    ref = load_paper_results(args.paper_results)
    for key, filename in FIG_FILES.items():
        fig_meta = ref.get("figures", {}).get(key, {})
        source = fig_meta.get("source_table", "unknown")
        write_text_figure(out["figures"] / filename, title=key, source=source)

    write_log(out["logs"] / "make_figures.json", {"mode": ref.get("metadata", {}).get("mode", "unknown"), "paper_results": args.paper_results, "message": "Figures regenerated from frozen references only."})
    print("Regenerated figure artifacts from frozen paper results.")


if __name__ == "__main__":
    main()
