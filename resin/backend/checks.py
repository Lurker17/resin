"""Framework-neutral validation shared by model backends."""

from __future__ import annotations

from collections.abc import Iterable


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def token_ids(values: Iterable[int], vocab_size: int) -> None:
    if any(token < 0 or token >= vocab_size for token in values):
        raise ValueError("token id is out of range")


def positions(values: Iterable[int], maximum: int) -> None:
    if any(position < 0 or position >= maximum for position in values):
        raise ValueError("position exceeds max_position_embeddings")


def nonempty_plan(plan) -> None:
    if not plan.sequence_ids:
        raise ValueError("BatchPlan is empty")
