"""Configuration loading utilities with zero third-party dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load config from JSON-formatted .yaml/.json file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config_snapshot(config: Dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
