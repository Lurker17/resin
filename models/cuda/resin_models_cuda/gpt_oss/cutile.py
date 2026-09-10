"""GPT-OSS cuTile top-four routing kernel."""

from __future__ import annotations

import cuda.tile as ct
import torch
from torch import Tensor

_NUM_EXPERTS = 128
_TOP_K = 4
_NEGATIVE = -3.402823466e38


@ct.kernel
def _top4_softmax_kernel(logits, expert_indices, expert_weights):
    row = ct.bid(0)
    scores = ct.load(logits, (row, 0), shape=(1, _NUM_EXPERTS))
    expert_ids = ct.arange(_NUM_EXPERTS, dtype=ct.int32).reshape((1, _NUM_EXPERTS))

    index0 = ct.argmax(scores, 1, keepdims=True)
    score0 = ct.max(scores, 1, keepdims=True)
    remaining = ct.where(expert_ids == index0, _NEGATIVE, scores)

    index1 = ct.argmax(remaining, 1, keepdims=True)
    score1 = ct.max(remaining, 1, keepdims=True)
    remaining = ct.where(expert_ids == index1, _NEGATIVE, remaining)

    index2 = ct.argmax(remaining, 1, keepdims=True)
    score2 = ct.max(remaining, 1, keepdims=True)
    remaining = ct.where(expert_ids == index2, _NEGATIVE, remaining)

    index3 = ct.argmax(remaining, 1, keepdims=True)
    score3 = ct.max(remaining, 1, keepdims=True)

    indices01 = ct.cat((index0, index1), 1)
    indices23 = ct.cat((index2, index3), 1)
    indices = ct.cat((indices01, indices23), 1)

    scores01 = ct.cat((score0, score1), 1)
    scores23 = ct.cat((score2, score3), 1)
    selected_scores = ct.cat((scores01, scores23), 1).astype(ct.float32)
    probabilities = ct.exp(selected_scores - ct.max(selected_scores, 1, keepdims=True))
    probabilities /= ct.sum(probabilities, 1, keepdims=True)
    # Match torch.softmax(BF16): round the probabilities to BF16 before
    # emitting the FP32 scales consumed by FlashInfer's fused MoE kernel.
    probabilities = probabilities.astype(ct.bfloat16).astype(ct.float32)

    ct.store(expert_indices, (row, 0), indices)
    ct.store(expert_weights, (row, 0), probabilities)


def top4_softmax(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Select four experts per row and emit FlashInfer-ready routing tensors."""

    rows = logits.shape[0]
    expert_indices = torch.empty(
        (rows, _TOP_K),
        dtype=torch.int32,
        device=logits.device,
    )
    expert_weights = torch.empty(
        (rows, _TOP_K),
        dtype=torch.float32,
        device=logits.device,
    )
    ct.launch(
        torch.cuda.current_stream(logits.device),
        (rows,),
        _top4_softmax_kernel,
        (logits, expert_indices, expert_weights),
    )
    return expert_indices, expert_weights


__all__ = ["top4_softmax"]
