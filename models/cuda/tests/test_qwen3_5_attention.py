from types import SimpleNamespace
from typing import cast

import pytest
import torch

from resin.backend.cuda.paged_kv import SlabPagedKvCache
from resin_models_cuda.qwen3_5.attention import PreparedPagedAttention


@pytest.mark.parametrize("tokens", [1, 4])
def test_layer_pages_use_live_base_pages_and_stable_buffer(tokens):
    base = torch.tensor([9, 3, 15, 0], dtype=torch.int32)
    physical = torch.empty_like(base)
    storage = physical.data_ptr()
    seen = []

    class Wrapper:
        def run(self, query, cache):
            assert physical.data_ptr() == storage
            seen.append(physical.clone())
            keys, values = cache
            return query + keys[physical].sum() + values[physical].sum()

    cache = cast(
        SlabPagedKvCache,
        SimpleNamespace(keys=torch.arange(18), values=torch.arange(18) * 2),
    )
    attention = PreparedPagedAttention(Wrapper(), cache, base, physical, 3, tokens)
    query = torch.randn(2, tokens, 4, 8)
    for mutation in range(2):
        if mutation:
            base.copy_(torch.tensor([6, 12, 0, 3]))
        for layer in (0, 2, 1):
            expected = base + layer
            output = attention.run(query, layer)
            torch.testing.assert_close(seen[-1], expected, rtol=0, atol=0)
            torch.testing.assert_close(output, query + 3 * expected.sum())
    with pytest.raises(ValueError, match="outside"):
        attention.run(query, 3)
