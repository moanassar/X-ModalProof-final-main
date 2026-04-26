"""Watermark loss helpers without third-party dependencies."""

from __future__ import annotations

from typing import Dict, List

from src.watermark.signature import cosine_scores


def alignment_loss(trigger_embeddings: List[List[float]], signature: List[float]) -> float:
    if not trigger_embeddings:
        return 0.0
    scores = cosine_scores(trigger_embeddings, signature)
    return sum(1.0 - s for s in scores) / len(scores)


def separation_loss(benign_embeddings: List[List[float]], signature: List[float], margin: float) -> float:
    if not benign_embeddings:
        return 0.0
    scores = cosine_scores(benign_embeddings, signature)
    return sum(max(0.0, s - margin) for s in scores) / len(scores)


def watermark_loss(
    task_loss_value: float,
    embeddings: List[List[float]],
    is_trigger: List[int],
    signature: List[float],
    lambda_align: float,
    lambda_sep: float,
    margin_gamma: float,
) -> Dict[str, float]:
    trigger_embeddings = [e for e, f in zip(embeddings, is_trigger) if f == 1]
    benign_embeddings = [e for e, f in zip(embeddings, is_trigger) if f == 0]
    align = alignment_loss(trigger_embeddings, signature)
    sep = separation_loss(benign_embeddings, signature, margin=margin_gamma)
    total = task_loss_value + lambda_align * align + lambda_sep * sep
    return {"total": total, "task": task_loss_value, "align": align, "sep": sep}
