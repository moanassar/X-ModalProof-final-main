"""Deterministic seed utilities."""

import random


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python RNG seed."""
    random.seed(seed)
    _ = deterministic
