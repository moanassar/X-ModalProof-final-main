"""Signature computation and cosine scoring (stdlib only)."""

from __future__ import annotations

import math
from typing import List


def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _normalize(vec: List[float]) -> List[float]:
    n = _l2_norm(vec)
    if n == 0.0:
        return [0.0 for _ in vec]
    return [v / n for v in vec]


def compute_signature(trigger_embeddings: List[List[float]]) -> List[float]:
    if not trigger_embeddings:
        raise ValueError("trigger_embeddings must be non-empty")
    dim = len(trigger_embeddings[0])
    acc = [0.0] * dim
    for row in trigger_embeddings:
        nrow = _normalize(row)
        for i, v in enumerate(nrow):
            acc[i] += v
    centroid = [v / len(trigger_embeddings) for v in acc]
    return _normalize(centroid)


def cosine_scores(embeddings: List[List[float]], signature: List[float]) -> List[float]:
    nsig = _normalize(signature)
    out: List[float] = []
    for row in embeddings:
        nrow = _normalize(row)
        out.append(sum(a * b for a, b in zip(nrow, nsig)))
    return out
