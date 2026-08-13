"""Framework-injected sampling operations shared by model runners."""

from __future__ import annotations


class GreedySampler:
    def __init__(self, torch) -> None:
        self.torch = torch

    def __call__(self, logits, *, out=None):
        if out is None:
            return logits.argmax(dim=-1)
        self.torch.argmax(logits, dim=-1, out=out)
        return out
