"""Verification engine."""

from __future__ import annotations

from typing import Dict, List


def verify_watermark(trigger_scores: List[float], threshold: float) -> Dict[str, float | bool]:
    mean_score = sum(trigger_scores) / max(1, len(trigger_scores))
    return {
        "mean_trigger_score": float(mean_score),
        "threshold": float(threshold),
        "decision": bool(mean_score >= threshold),
    }
