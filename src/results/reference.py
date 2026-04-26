"""Frozen paper-results loading and artifact regeneration helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def load_paper_results(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dirs(base: str | Path = "outputs") -> Dict[str, Path]:
    root = Path(base)
    metrics = root / "metrics"
    figures = root / "figures"
    logs = root / "logs"
    for d in [root, metrics, figures, logs]:
        d.mkdir(parents=True, exist_ok=True)
    return {"root": root, "metrics": metrics, "figures": figures, "logs": logs}


def write_csv(path: Path, rows: List[Dict], preferred_headers: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_headers is not None:
        headers = preferred_headers
    elif rows:
        headers = sorted({k for row in rows for k in row.keys()})
    else:
        headers = ["note"]
        rows = [{"note": "no frozen rows available"}]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_text_figure(path: Path, title: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n")
        f.write("Regenerated from frozen paper results.\n")
        f.write(f"Source: {source}\n")


def write_log(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
