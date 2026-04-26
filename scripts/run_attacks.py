#!/usr/bin/env python
"""Generate attack/robustness artifacts from frozen results, scaffold full mode, or real_full attacks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.real_full import run_real_full_attack
from src.results.full_pipeline import run_attack_eval
from src.results.reference import ensure_output_dirs, load_paper_results, write_csv, write_log
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", default="results/paper_results.json")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--mode", choices=["frozen", "full", "real_full"], default="frozen")
    parser.add_argument("--attack", choices=["pruning", "finetuning", "distillation"], default="pruning")
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--distill-epochs", type=int, default=1)
    parser.add_argument("--student-model", default="")
    parser.add_argument("--output-dir", default="outputs/full_real/attacks")
    args = parser.parse_args()

    out = ensure_output_dirs("outputs")

    if args.mode == "real_full":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --mode real_full")
        run_dir = Path(args.run_dir)
        try:
            metrics = run_real_full_attack(
                run_dir=run_dir,
                attack=args.attack,
                strength=float(args.strength),
                finetune_epochs=int(args.finetune_epochs),
                distill_epochs=int(args.distill_epochs),
                student_model=str(args.student_model),
                output_root=Path(args.output_dir),
            )
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"real_full attack failed: {exc}") from exc

        write_log(
            out["logs"] / "run_attacks.json",
            {
                "mode": "real_full",
                "attack": args.attack,
                "run_dir": args.run_dir,
                "output_dir": args.output_dir,
                "robustness_drop": metrics.get("robustness_drop"),
            },
        )
        print(f"real_full attack completed: {args.attack} for {args.run_dir}")
        return

    if args.mode == "full":
        run_dir = Path(args.run_dir or "outputs/full_text_b4_seed0")
        cfg = load_config(args.config)
        rows = run_attack_eval(cfg, run_dir)
        write_csv(out["metrics"] / "table3_detection_accuracy.csv", rows)
        write_log(out["logs"] / "run_attacks.json", {"mode": "full", "executed": True, "rows": len(rows)})
        print("Attack artifacts generated from local full-mode run.")
        return

    ref = load_paper_results(args.paper_results)
    rows = ref.get("tables", {}).get("table3_detection_accuracy", [])
    write_csv(out["metrics"] / "table3_detection_accuracy.csv", rows)
    write_log(out["logs"] / "run_attacks.json", {"mode": "frozen", "message": "Attack results regenerated from frozen references."})
    print("Attack artifacts regenerated from frozen references.")


if __name__ == "__main__":
    main()
