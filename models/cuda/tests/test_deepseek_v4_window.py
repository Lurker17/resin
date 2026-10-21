from __future__ import annotations

import pytest
import torch

from resin_models_cuda.deepseek_v4.cache import DeviceMapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("start, block_length", [(0, 257), (300, 256), (300, 6)])
def test_attention_metadata_keeps_each_query_inside_its_sliding_window(
    start: int, block_length: int
) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("DeepSeek-V4 cuTile kernels require Blackwell")
    pytest.importorskip("cuda.tile")
    from resin_models_cuda.deepseek_v4.cutile import verify_attention_metadata

    window = 128
    block_count = (start + block_length + 127) // 128
    parent_ids = torch.arange(
        1, block_count + 1, dtype=torch.int32, device="cuda"
    ).view(1, -1)
    mapping = DeviceMapping(
        parent_ids=parent_ids,
        child_slots=torch.zeros_like(parent_ids),
        first_blocks=torch.zeros(1, dtype=torch.int32, device="cuda"),
        block_tokens=128,
        divisor=1,
        parent_rows=128,
        packing=1,
        row_scale=1,
        layer_offset=0,
    )
    indices, lengths, sequence_lengths = verify_attention_metadata(
        torch.tensor([start], dtype=torch.int32, device="cuda"),
        block_length,
        window,
        window + block_length - 1,
        None,
        0,
        0,
        window_mapping=mapping,
    )
    indices = indices.cpu()
    lengths = lengths.cpu()
    sequence_lengths = sequence_lengths.cpu()
    assert indices.shape[:2] == (1, block_length)
    # Dynamic cuTile extents use a stable divisibility class, so permit its
    # bounded padding while rejecting a table that grows with query count.
    assert window <= indices.shape[2] < window + 128
    for query in range(block_length):
        position = start + query
        expected = list(range(max(0, position - window + 1), position + 1))
        # The native ABI keeps 128 SWA columns, including invalid early rows.
        # seq_lens and -1 entries provide causality inside that fixed segment.
        assert lengths[0, query].item() == window
        assert sequence_lengths[0, query].item() == position + 1
        # Parent zero is reserved; this identity mapping offsets rows by 128.
        actual = indices[0, query]
        assert actual[: len(expected)].tolist() == [value + 128 for value in expected]
        assert (actual[len(expected) :] == -1).all()
