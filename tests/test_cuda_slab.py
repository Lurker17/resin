from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resin import CacheGroupLayout, CacheLayout
from resin.backend.cuda.slab import GraphPageTableArena


@pytest.mark.parametrize("private_scratch", [False, True])
def test_graph_replay_pads_unused_rows_with_the_capture_scratch_parents(
    private_scratch,
):
    class Arena(GraphPageTableArena):
        device = torch.device("cpu")
        layout = CacheLayout(10, [CacheGroupLayout(128, blocks_per_slab=2)] * 2)

        def _make_graph_cache(self, groups):
            return SimpleNamespace(groups=groups)

        def _graph_padding_address(self, index: int, row: int) -> tuple[int, int]:
            return (
                (self.layout.parent_capacity + index + 1, 0)
                if private_scratch
                else (0, 0)
            )

    arena = Arena()
    cache = arena.graph_cache(2, 256)
    plan = SimpleNamespace(
        rows=1,
        groups=tuple(
            SimpleNamespace(
                block_slabs=lambda row: np.array([7], dtype=np.uint32),
                block_slots=lambda row: np.array([1], dtype=np.uint32),
                block_table_start_indices=np.array([3], dtype=np.uint32),
            )
            for _ in range(2)
        ),
    )

    capture = arena.graph_shape_capture_inputs(2, 256)
    replay = arena.graph_plan_inputs(cache, plan)

    for group in range(2):
        prefix = f"cache.{group}"
        parent, slot = arena._graph_padding_address(group, 1)
        np.testing.assert_array_equal(
            replay[f"{prefix}.slabs"], [[7, 0], [parent, parent]]
        )
        np.testing.assert_array_equal(replay[f"{prefix}.slots"], [[1, 0], [0, 0]])
        np.testing.assert_array_equal(replay[f"{prefix}.first"], [3, 0])
        # Only whole padded requests use scratch storage. A real request's
        # absent history columns must remain zero rather than becoming valid KV.
        assert np.asarray(replay[f"{prefix}.slabs"])[1].tolist() == list(
            np.asarray(capture[f"{prefix}.slabs"])[-2:]
        )
