"""Capture-safe categorical sampling with a separate random stream per request."""

import torch
import triton
import triton.language as tl
from triton.language.random import philox


@triton.jit
def _partials(
    P,
    Seeds,
    Offsets,
    Values,
    Indices,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    token = part * BLOCK + tl.arange(0, BLOCK)
    seed = tl.load(Seeds + row).to(tl.uint64)
    offset = tl.load(Offsets + row).to(tl.uint64)
    # Philox4x32-10 uses the full 64-bit request seed as its key. Token IDs
    # and engine offsets occupy separate counter words: offset+token would
    # overlap random streams when consecutive engine offsets differ by 32.
    random, _, _, _ = philox(
        seed,
        token.to(tl.uint32),
        tl.full((BLOCK,), 0, tl.uint32),
        (offset & 0xFFFFFFFF).to(tl.uint32),
        (offset >> 32).to(tl.uint32),
    )
    # Exactly representable open-uniform bin centers keep both logs finite.
    uniform = ((random >> 9).to(tl.float32) + 0.5) * 1.1920928955078125e-7
    p = tl.load(P + row * V + token, token < V, other=0.0)
    finite = (p >= 0) & (p < float("inf"))
    # argmax(log(p) + Gumbel(0, 1)) is a categorical draw from p.
    score = tl.where(
        (p > 0) & finite, tl.log(p) - tl.log(-tl.log(uniform)), -float("inf")
    )
    best = tl.max(score, 0)
    winner = tl.min(tl.where((score == best) & (token < V), token, 2147483647), 0)
    invalid = tl.sum(((token < V) & ~finite).to(tl.int32), 0) > 0
    # +inf marks an invalid input fragment; valid Gumbel scores are finite.
    tl.store(Values + row * PARTS + part, tl.where(invalid, float("inf"), best))
    tl.store(Indices + row * PARTS + part, winner)


@triton.jit
def _finish(
    Values,
    Indices,
    Output,
    Valid,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    values = tl.load(Values + row * PARTS + part, part < PARTS, other=-float("inf"))
    ids = tl.load(Indices + row * PARTS + part, part < PARTS, other=2147483647)
    best = tl.max(values, 0)
    chosen = tl.min(tl.where(values == best, ids, 2147483647), 0)
    valid = (
        (best > -float("inf")) & (best < float("inf")) & (chosen >= 0) & (chosen < V)
    )
    tl.store(Output + row, tl.where(valid, chosen, -1))
    tl.store(Valid + row, valid)


def sample_categorical(probabilities, seeds, offsets):
    """Return row-aligned IDs and validity without consuming mutable RNG state."""
    rows, vocab = probabilities.shape
    block = 1024
    parts = max(1, triton.cdiv(vocab, block))
    device = probabilities.device
    values = torch.empty((rows, parts), device=device, dtype=torch.float32)
    indices = torch.empty((rows, parts), device=device, dtype=torch.int32)
    result = torch.empty(rows, device=device, dtype=torch.int64)
    valid = torch.empty(rows, device=device, dtype=torch.bool)
    if rows == 0:
        return result, valid
    with torch.cuda.device(device):
        _partials.run(
            probabilities,
            seeds,
            offsets,
            values,
            indices,
            vocab,
            parts,
            block,
            grid=(rows, parts),
            warmup=False,
            num_warps=4,
        )
        _finish.run(
            values,
            indices,
            result,
            valid,
            vocab,
            parts,
            triton.next_power_of_2(parts),
            grid=(rows,),
            warmup=False,
            num_warps=4,
        )
    return result, valid
