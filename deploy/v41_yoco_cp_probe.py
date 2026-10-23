"""Two-GPU CP compaction and collective graph probe, without model weights.

Run ``torchrun --standalone --nproc-per-node=2 -m deploy.v41_yoco_cp_probe``.
The CPU entry point exercises identical metadata and query-exchange checks;
only a CUDA run captures and replays the producer/consumer graph regions.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import NoReturn

import torch
import torch.distributed as dist

from deploy.deepseek_v41_validation import _validation_stage
from deploy.v41_batch_probe import _tensors
from resin import BatchPhase, Parallelism, RunnerConfig
from resin.backend.control_plane import ControlPlane
from resin.backend.cuda.context_parallel import ContextParallelTokens
from resin.backend.cuda.graphs import CudaGraphMode
from resin.backend.cuda.runner import CudaRunner
from resin.backend.cuda.slab import DevicePageGroup
from resin.backend.distributed import StepContext
from resin.backend.topology import ParallelGroups
from resin_models_cuda.deepseek_v41.cache import CacheGeometry, DeviceCache
from resin_models_cuda.deepseek_v41.yoco import compact_runtime

_PRODUCER_ROWS = 512
# Retained rows, prompt endpoint, optional future SWA replay boundary.
_CASES = (
    (1, 512, None),
    (1, 513, 131072 - 140),
    (1, 131072, None),
    (11, 517, None),
    (11, 131077, None),
    (140, 516, None),
    (140, 131080, None),
)


def _case(endpoint, device):
    positions = endpoint - _PRODUCER_ROWS + torch.arange(_PRODUCER_ROWS, device=device)
    groups, packings = [], (4, 8, 2, 16)
    for index, packing in enumerate(packings):
        first = 0 if index == 1 else max(0, (endpoint - _PRODUCER_ROWS) // 128 - 1)
        width = (endpoint + 127) // 128 - first + 1
        ordinal = torch.arange(width, device=device)[None] * 13 + 7
        groups.append(
            DevicePageGroup(
                (ordinal // packing + 1).int(),
                (ordinal % packing).int(),
                torch.tensor([first], dtype=torch.int32, device=device),
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
        max_write_tokens=_PRODUCER_ROWS,
        window_size=128,
        draft_tokens=8,
    )
    return DeviceCache(
        geometry, tuple(groups), torch.empty(0, dtype=torch.uint8, device=device)
    ), positions[None]


def _runtime(cache, positions, chosen, lower):
    candidates = torch.arange(
        _PRODUCER_ROWS * 8, dtype=torch.int32, device=positions.device
    ).reshape(1, _PRODUCER_ROWS, 8)
    runtime = dict(
        metadata=cache.prepare_metadata(positions),
        indices=candidates,
        candidate_blocks=candidates.flatten(0, 1),
        index_metadata=object(),
        full_index_kv_stored={20},
    )
    result = compact_runtime(runtime, chosen, swa_start=lower)
    assert "index_metadata" not in result and "index_metadata" in runtime
    assert result["full_index_kv_stored"] == {20}
    assert torch.equal(result["indices"], candidates.index_select(1, chosen))
    assert torch.equal(result["candidate_blocks"], result["indices"].flatten(0, 1))
    return result["metadata"]


def _equal(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


def _unused(_shape) -> NoReturn:
    raise AssertionError("decode is disabled in the CP probe")


class _RegionRunner(CudaRunner):
    """Use production capture/bucketing with a small NCCL tensor operation."""

    def __init__(self, groups, device):
        super().__init__(
            device=device,
            label="CP compaction probe",
            kv_cache_dtype="bfloat16",
            cuda_graph_decode="disabled",
        )
        self.runner_config = RunnerConfig()
        self.prefill_chunk_size = _PRODUCER_ROWS
        self.replays = []
        self._configure_expert_parallel_graphs(torch.nn.Module(), groups)
        self._configure_cuda_graphs(
            {"decode": CudaGraphMode((), _unused, lambda shape: {})}
        )

        def operation(hidden, _ids):
            value = hidden.clone()
            groups.distributed.all_reduce(value, group=groups.device_domain.handle)
            return value + 1

        self._configure_prefill_graphs(
            (operation,) * 4, hidden_size=2, max_tokens=_PRODUCER_ROWS
        )
        self.capture_cuda_graphs()

    def forward(self, plan):
        raise AssertionError("the CP probe calls region callbacks directly")

    def _run_cuda_graph(self, mode, shape, inputs):
        self.replays.append(shape)
        return super()._run_cuda_graph(mode, shape, inputs)


def _graph_round(runner, groups, context, retained, device):
    source = torch.arange(_PRODUCER_ROWS, device=device)
    hidden = torch.stack((source % 32, source // 32), -1).bfloat16()
    expected = hidden.cpu()
    cp = ContextParallelTokens(
        (_PRODUCER_ROWS,), groups.cp, device=device, distributed=groups.distributed
    )
    before = len(runner.replays)
    with runner._distributed_prefill_graphs(
        context, region_tokens={range(2, 4): retained}
    ):
        for region in range(4):
            if region == 2:
                hidden, expected = hidden[-retained:], expected[-retained:]
                cp = ContextParallelTokens(
                    (retained,),
                    groups.cp,
                    device=device,
                    distributed=groups.distributed,
                )
            # Attention gathers query-sharded output before each FFN. In the
            # one-row case one CP peer owns zero queries but still gathers.
            hidden = cp.gather(cp.select(hidden))
            hidden = runner._run_prefill_region(region, hidden)
            expected = expected * groups.domain.size + 1
    _equal(hidden, expected)
    compact = 1 << (retained - 1).bit_length()
    shapes = runner.replays[before:]
    assert shapes == [
        (_PRODUCER_ROWS, 1),
        (_PRODUCER_ROWS, 2),
        (compact, 3),
        (compact, 4),
    ]
    return shapes


def run_rank(groups, *, device, capture_graphs=True):
    """Check CPU/CUDA metadata, uneven CP gathers, then real CUDA graph replay."""
    if groups.cp.size != 2 or groups.domain.size != 2:
        raise ValueError("this focused probe requires a two-rank CP2 domain")
    device = torch.device(device)
    if capture_graphs and device.type != "cuda":
        raise ValueError("real graph capture requires CUDA")
    control = ControlPlane(groups.distributed)

    def coordinate(stage, error):
        _, errors = control.gather(
            groups.domain,
            f"yoco_cp:{stage}",
            error=None if error is None else str(error),
        )
        if any(error is not None for error in errors):
            raise AssertionError(f"CP qualification failed at {stage}: {errors}")

    runner = _RegionRunner(groups, device) if capture_graphs else None
    context = StepContext(
        groups, 0, BatchPhase.PREFILL, ((1, _PRODUCER_ROWS),), 1, _PRODUCER_ROWS
    )
    cases = []
    try:
        for retained, endpoint, boundary in _CASES:
            lower = endpoint - retained if boundary is None else boundary
            chosen = torch.arange(
                _PRODUCER_ROWS - retained, _PRODUCER_ROWS, device=device
            )
            cpu, positions = _case(endpoint, "cpu")
            cache, device_positions = _case(endpoint, device)
            with _validation_stage(f"metadata:{retained}:{endpoint}", coordinate):
                expected = _runtime(cpu, positions, chosen.cpu(), lower)
                frame = _runtime(cache, device_positions, chosen, lower)
                actual_fields, expected_fields = _tensors(frame), _tensors(expected)
                assert actual_fields.keys() == expected_fields.keys()
                for name in actual_fields:
                    _equal(actual_fields[name], expected_fields[name])
            cp = ContextParallelTokens(
                (retained,), groups.cp, device=device, distributed=groups.distributed
            )
            restored = cp.gather(cp.select(frame.positions.flatten()))
            with _validation_stage(f"positions:{retained}:{endpoint}", coordinate):
                _equal(restored, expected.positions.flatten())
                if lower > endpoint:
                    assert (frame.slot_mapping_store_swa == -1).all()
                    assert (frame.klen[1] > 0).all()
            for ratio in (1, 2):
                logical = torch.zeros(
                    (1, retained, 512), dtype=torch.int32, device=device
                )
                logical[..., 0], logical[..., 1] = -1, 0
                logical[..., 2], logical[..., 3] = (
                    frame.klen[ratio] - 1,
                    frame.klen[ratio],
                )
                logical[..., 4:8] = (
                    torch.tensor([127, 128, 256, 2048], device=device) // ratio
                )
                physical = frame.compressed_indices(20, ratio, logical)
                restored = cp.gather(cp.select(physical.squeeze(0)))
                with _validation_stage(
                    f"indices:{retained}:{endpoint}:{ratio}", coordinate
                ):
                    reference = expected.compressed_indices(20, ratio, logical.cpu())
                    _equal(restored, reference.squeeze(0))
            shapes = None
            if runner is not None:
                with _validation_stage(f"graphs:{retained}:{endpoint}", coordinate):
                    shapes = _graph_round(runner, groups, context, retained, device)
            cases.append(
                dict(
                    retained_rows=retained,
                    endpoint=endpoint,
                    swa_start=lower,
                    local_queries=cp.plan.local_tokens,
                    metadata_fields=len(actual_fields),
                    region_shapes=shapes,
                )
            )
        counts = None if runner is None else runner.cuda_graph_counts["prefill"]
        if counts is not None:
            assert counts["replay_count"] == 4 * len(_CASES)
            assert counts["eager_fallback_count"] == 0
        return dict(
            ok=True,
            rank=groups.rank,
            cp=2,
            device=str(device),
            captured_graphs=runner is not None,
            graph_counts=counts,
            cases=cases,
        )
    finally:
        if runner is not None:
            runner.close_cuda_graphs()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    groups = None
    try:
        groups = ParallelGroups.create(Parallelism(world_size=2, cp=2, ep=2))
        report = run_rank(groups, device=device)
        reports = [None] * groups.domain.size
        dist.all_gather_object(reports, report, group=groups.domain.handle)
        if groups.rank == 0:
            output = json.dumps(
                dict(
                    torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(device),
                    ranks=reports,
                ),
                indent=2,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output + "\n")
            print(output, flush=True)
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
