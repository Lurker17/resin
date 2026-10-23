"""DeepInfer's C1/C2 compressor on Engine-owned physical state pages."""

from types import SimpleNamespace

import torch
from torch import nn

from resin_models_cuda.deepseek_v41.linear import PreparedLinear


def compress_projected(scores, values, positions, cache, layer, ratio):
    """Store projections then pool complete windows with the upstream kernel.

    C2 windows divide the Engine's 128-token pages. Each physical state page
    therefore receives unique positions within this call, including prefill
    chunks spanning many pages. Its previous partial window stays on the same
    physical page; verification writes never overwrite accepted earlier rows.
    """
    from resin_models_cuda.deepseek_v41.upstream.cuda.compress import (
        compress_and_update_state_decode_tl,
    )

    if ratio != 2:
        raise ValueError("the paged compressor kernel requires C2 compression")
    shape = positions.shape
    scores = scores.reshape(-1, scores.shape[-1])
    values = values.reshape(-1, values.shape[-1])
    output = values.new_empty(values.shape, dtype=torch.bfloat16)
    compress_and_update_state_decode_tl(DIM=values.shape[-1], COMPRESS_RATIO=ratio)(
        scores,
        values,
        cache.compressor_state(layer),
        cache.compressor_pages(positions).flatten().int(),
        (positions.flatten() + 1).int(),
        output,
    )
    return output.reshape(*shape, -1)


class PreparedCompressor(nn.Module):
    """One BF16 GEMM, FP32 gated pooling, then the checkpoint RMS norm."""

    def __init__(self, source):
        super().__init__()
        self.layer, self.ratio, self.norm = source.layer, source.ratio, source.norm
        if self.ratio not in (1, 2):
            raise ValueError("DeepSeek-V4.1 supports only C1/C2 compression")
        if self.ratio == 1:
            self.projection = PreparedLinear(source.wkv)
        else:
            # DeepInfer's Gate/A projections use BF16 weights and retain FP32
            # GEMM accumulation through the softmax compression boundary.
            self.projection = PreparedLinear(
                SimpleNamespace(
                    weight=torch.cat(
                        (source.wgate.weight, source.wkv.weight)
                    ).bfloat16(),
                    scale=None,
                ),
                use_fp32_output=True,
            )

    def forward(self, hidden, positions, cache):
        projected = self.projection(hidden)
        if self.ratio == 2:
            scores, values = projected.chunk(2, -1)
            projected = compress_projected(
                scores, values, positions, cache, self.layer, self.ratio
            )
        return self.norm(projected)
