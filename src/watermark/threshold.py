"""Threshold selection logic (stdlib only)."""

from __future__ import annotations

from typing import Dict, List


def _f1(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def select_threshold(trigger_scores: List[float], benign_scores: List[float], step: float = 0.001) -> Dict[str, float]:
    scores = trigger_scores + benign_scores
    labels = [1] * len(trigger_scores) + [0] * len(benign_scores)
    lo, hi = min(scores), max(scores)

    best_t = lo
    best_f1 = -1.0
    t = lo
    while t <= hi + 1e-12:
        preds = [1 if s >= t else 0 for s in scores]
        f1 = _f1(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        t += step

    return {"threshold": float(best_t), "f1": float(best_f1)}
