"""Prepared metadata is shared by layers, preserving Engine page ownership."""

import pytest
import torch

from resin.backend.cuda.slab import DevicePageGroup
from resin_models_cuda.deepseek_v41.cache import CacheGeometry, DeviceCache


def cache_and_positions(*, draft=False, device="cpu"):
    positions = torch.tensor(
        [[0, 1, 2, 3, 4], [126, 127, 128, 129, 130], [382, 383, 384, 385, 386]],
        device=device,
    )
    packings = (4, 8, 2, 16)
    groups = []
    for index, packing in enumerate(packings):
        width = 5
        ordinal = torch.arange(15, device=device).reshape(3, width) * 13 + 7
        first = torch.zeros(3, dtype=torch.int32, device=device)
        if index != 1:
            first = ((positions[:, 0] // 128) - 1).clamp_min(0).int()
        groups.append(
            DevicePageGroup(
                (ordinal // packing + 1).int(), (ordinal % packing).int(), first
            )
        )
    geometry = CacheGeometry(
        attention_layers=6,
        kv_sources=(1, 3),
        ratios=(0, 2, 2, 1, 1, 0),
        compressed_offsets=(None, 0, 0, 0, 0, None),
        index_offsets=(None, 0, 0, 0, 0, None),
        compressor_sources=(1,),
        packings=packings,
        parent_bytes=4194304,
        max_write_tokens=5,
        window_size=128,
        draft_tokens=8,
        max_batch_size=3,
    )
    cache = DeviceCache(
        geometry, tuple(groups), torch.empty(0, dtype=torch.uint8, device=device)
    )
    return cache, positions


def physical(cache, group, row, position, ratio=1):
    table = cache.groups[group]
    column = position // 128 - int(table.first_blocks[row])
    if position < 0 or not 0 <= column < table.parent_ids.shape[1]:
        return -1
    parent, slot = (
        int(table.parent_ids[row, column]),
        int(table.child_slots[row, column]),
    )
    if parent <= 0:
        return -1
    page = parent * cache.geometry.packings[group] + slot
    return page * (128 // ratio) + position % 128 // ratio


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_prepared_metadata_matches_scalar_paged_addresses(draft, masked):
    cache, positions = cache_and_positions(draft=draft)
    valid = torch.tensor([True] * 10 + [False] * 5) if masked else None
    frame = cache.prepare_metadata(positions, draft=draft, valid_tokens=valid)
    assert frame.positions.dtype == frame.context_lens.dtype == torch.int32
    assert torch.equal(frame.positions, positions.int())
    assert torch.equal(frame.context_lens, (positions + 1).int())
    assert frame.query_indices.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    for row in range(3):
        for query in range(5):
            pos = int(positions[row, query])
            active = not masked or row != 2
            assert frame.slot_mapping_store_swa[row, query] == (
                physical(cache, 0, row, pos) if active else -1
            )
            assert (
                frame.state_block_idx[row, query] == physical(cache, 2, row, pos) // 128
            )
            end = int(positions[row, 0]) if draft else pos + 1
            count = min(end, 128)
            expected = [
                physical(cache, 0, row, end - count + column)
                if active and column < count
                else -1
                for column in range(128)
            ]
            assert frame.kv_swa_index[row, query].tolist() == expected
            for ratio in (1, 2):
                complete = active and (pos + 1) % ratio == 0
                assert frame.positions_c[ratio][row, query] == (
                    pos // ratio * ratio if complete else 0
                )
                assert frame.slot_mapping_store_ca[ratio][row, query] == (
                    physical(cache, 1, row, pos, ratio) if complete else -1
                )
                assert frame.klen[ratio][row, query] == (
                    (pos + 1) // ratio if active else 0
                )
            assert frame.draft_slots[row, query] == (
                row * 8 + query if active and draft else -1
            )
            assert frame.draft_indices[row, query].tolist() == [
                row * 8 + column if active and draft and column < 5 else -1
                for column in range(8 if draft else 1)
            ]
    for query in range(5):
        for row in range(3):
            table = cache.groups[1]
            expected = (
                table.parent_ids[row] * cache.geometry.packings[1]
                + table.child_slots[row]
            )
            assert torch.equal(frame.block_tables[query * 3 + row], expected)


def test_compressed_mapping_reuses_identity_and_refreshes_new_source_selection():
    cache, positions = cache_and_positions()
    frame = cache.prepare_metadata(positions)
    selected = (
        torch.tensor([0, -1, 2, 511, 3, 128, -1, 1], dtype=torch.int32)
        .expand(3, 5, 8)
        .clone()
    )
    before = selected.clone()
    first = frame.compressed_indices(2, 2, selected)
    assert frame.compressed_indices(2, 2, selected) is first
    assert frame.compressed_indices(8, 2, selected) is not first
    for row in range(3):
        for query in range(5):
            expected = [
                physical(cache, 1, row, int(value) * 2, 2)
                if 0 <= int(value) < int(frame.klen[2][row, query])
                else -1
                for value in selected[row, query]
            ]
            assert first[row, query].tolist() == expected
    assert torch.equal(selected, before)
    next_selected = selected.clone().fill_(0)
    fresh = frame.compressed_indices(2, 2, next_selected)
    assert fresh is not first
    assert frame.compressed_indices(2, 2, next_selected) is fresh
    assert (fresh[0, 0] == -1).all()  # C2 has no completed history at position 0.


def test_preparation_refreshes_engine_tables_and_cross_page_positions():
    cache, positions = cache_and_positions()
    first = cache.prepare_metadata(positions)
    # Move the last query across the next Engine page and change its physical owner.
    positions[1, -1] = 256
    cache.groups[0].parent_ids[1, 2] += 11
    next_frame = cache.prepare_metadata(positions)
    assert next_frame.slot_mapping_store_swa[1, -1] == physical(cache, 0, 1, 256)
    assert (
        next_frame.slot_mapping_store_swa[1, -1] != first.slot_mapping_store_swa[1, -1]
    )
    assert next_frame.state_block_idx[1, -1] == physical(cache, 2, 1, 256) // 128


def test_private_draft_keys_mask_invalid_columns_and_target_storage_stays_linear():
    cache, positions = cache_and_positions()
    mask = torch.ones(15, dtype=torch.bool)
    mask[2] = False
    frame = cache.prepare_metadata(positions, draft=True, valid_tokens=mask)
    assert (frame.draft_indices[0, :, 2] == -1).all()
    assert (frame.draft_indices[0, 2] == -1).all()
    assert frame.draft_indices[0, 0, 1] == 1
    long_positions = torch.arange(256).expand(3, -1).clone()
    target = cache.prepare_metadata(long_positions)
    assert target.draft_indices.shape == (3, 256, 1)
    assert (target.draft_indices == -1).all()


@pytest.mark.parametrize("width,context", [(1, 0), (3, 126), (1025, 131070)])
def test_growing_compressed_tables_preserve_physical_address_equations(width, context):
    cache, _ = cache_and_positions()
    positions = context + torch.arange(3)[:, None] + torch.arange(4)
    ordinal = torch.arange(3 * width).reshape(3, width) * 7 + 19
    compressed = DevicePageGroup(
        (ordinal // 8 + 1).int(),
        (ordinal % 8).int(),
        torch.zeros(3, dtype=torch.int32),
    )
    cache = DeviceCache(
        cache.geometry,
        cache.groups[:1] + (compressed,) + cache.groups[2:],
        cache.storage,
    )
    for group in (cache.groups[0], cache.groups[2]):
        group.first_blocks.copy_((positions[:, 0] // 128 - 1).clamp_min(0))
    frame = cache.prepare_metadata(positions)
    assert frame.block_tables.shape == (12, width)
    for ratio in (1, 2):
        selected = torch.stack(
            (
                torch.zeros_like(positions),
                (positions + 1) // ratio - 1,
                (positions + 1) // ratio,
                torch.full_like(positions, -1),
            ),
            -1,
        ).int()
        mapped = frame.compressed_indices(3, ratio, selected)
        for row in range(3):
            for query in range(4):
                expected = [
                    physical(cache, 1, row, int(value) * ratio, ratio)
                    if 0 <= int(value) < int(frame.klen[ratio][row, query])
                    else -1
                    for value in selected[row, query]
                ]
                assert mapped[row, query].tolist() == expected


@pytest.mark.parametrize(
    "positions,mask",
    [
        (torch.tensor([1, 2]), None),
        (torch.tensor([[1.0]]), None),
        (torch.tensor([[1, 2]]), torch.ones(2, dtype=torch.int32)),
        (torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.bool)),
    ],
)
def test_preparation_rejects_incompatible_positions_or_masks(positions, mask):
    cache, _ = cache_and_positions()
    with pytest.raises(ValueError):
        cache.prepare_metadata(positions, valid_tokens=mask)
