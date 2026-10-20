from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    Parallelism,
)
from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.distributed import StepContext
from resin.backend.topology import RankGroup
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner


def _groups(parallelism: Parallelism, rank: int = 0):
    broadcasts = []

    def broadcast(tensor, *, src, group):
        broadcasts.append((src, group, tensor.tolist()))

    groups = SimpleNamespace(
        topology=parallelism,
        rank=rank,
        distributed=SimpleNamespace(broadcast=broadcast),
        broadcasts=broadcasts,
    )
    for name, partitions in parallelism.partitions().items():
        for ranks in partitions:
            if rank in ranks:
                setattr(
                    groups, name, RankGroup(ranks, (name, ranks), ranks.index(rank))
                )
    return groups


class _Model:
    def __init__(self):
        self.calls = []

    def forward_prefill(self, token_ids, position, *, cache):
        self.calls.append(("prefill", token_ids.tolist(), position, cache))
        return torch.nn.functional.one_hot(
            token_ids[:, -1], num_classes=32
        ).float(), None

    def forward_decode(self, token_ids, positions, sequence_length, *, cache):
        self.calls.append(("decode", token_ids.tolist(), positions.tolist(), cache))
        return torch.nn.functional.one_hot(token_ids[:, -1], num_classes=32).float()

    def forward_experts_empty(self):
        self.calls.append(("empty",))


class _Cache:
    def __init__(self):
        self.plans = []
        self.rows = []

    def tensorize(self, plan, staging):
        self.plans.append(plan)
        return self

    def row(self, slot):
        self.rows.append(slot)
        return f"row-{slot}"


class _Stager:
    def __init__(self, device):
        self.target_device = device

    def device(self, name, values, dtype):
        return torch.tensor(values.tolist(), dtype=dtype)


class _Guard(nullcontext):
    def __init__(self):
        super().__init__()
        self.synchronizations = 0

    def __call__(self):
        self.synchronizations += 1


def _runner(monkeypatch, groups):
    import resin_models_cuda.deepseek_v4.cuda as implementation

    monkeypatch.setattr(implementation, "TensorStager", _Stager)
    runner = DeepSeekV4CudaRunner.__new__(DeepSeekV4CudaRunner)
    runner.parallel_groups = groups
    runner.tensor_parallel = TensorParallelContext()
    runner.device = torch.device("cpu")
    runner.max_batch_size = 4
    runner.max_seq_len = 256
    runner.prefill_chunk_size = 16
    runner.speculative_tokens = 0
    runner._model = _Model()
    runner._cache = _Cache()
    runner._cache_plan = lambda plan: plan
    runner._cuda_failure = _Guard()
    runner._prefill_compute_context = nullcontext
    runner._sampler = lambda logits, plan, *, offset_delta: logits.argmax(dim=-1)
    return runner


def _context(groups, phase, sizes, plan):
    return StepContext(
        groups=groups,
        step_id=0,
        phase=phase,
        replica_sizes=sizes,
        num_sequences=0 if plan is None else plan.num_sequences,
        num_tokens=0 if plan is None else plan.num_tokens,
    )


def test_ragged_prefill_replicas_enter_the_same_expert_rounds(monkeypatch):
    parallelism = Parallelism(world_size=4, ep=4, attn_dp=2)
    plans = (
        BatchPlan([7, 8], [0, 9], [2, 3, 4, 5, 6], [0, 3, 5], phase=BatchPhase.PREFILL),
        BatchPlan([7], [0], [9], [0, 1], phase=BatchPhase.PREFILL),
    )
    runners = []
    for rank, plan in zip((0, 2), plans, strict=True):
        groups = _groups(parallelism, rank)
        runner = _runner(monkeypatch, groups)
        output = runner.forward_distributed(
            plan, _context(groups, BatchPhase.PREFILL, ((2, 5), (1, 1)), plan)
        )
        assert output == ([4, 6] if rank == 0 else [9])
        assert runner._cache.plans == [plan]
        assert groups.broadcasts == [(rank, groups.attn_replication.handle, output)]
        runners.append(runner)

    assert [call[0] for call in runners[0]._model.calls] == ["prefill", "prefill"]
    assert [call[0] for call in runners[1]._model.calls] == ["prefill", "empty"]
    assert runners[0]._cache.rows == [0, 1]
    assert runners[1]._cache.rows == [0]
    assert runners[0]._model.calls[1][2] == 9


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_idle_replica_joins_experts_without_staging_or_touching_cache(
    monkeypatch, phase
):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    sizes = ((0, 0), (2, 5 if phase == BatchPhase.PREFILL else 2), (0, 0), (0, 0))

    assert (
        runner.forward_distributed(None, _context(groups, phase, sizes, None)) is None
    )

    assert runner._model.calls == [("empty",)] * (
        2 if phase == BatchPhase.PREFILL else 1
    )
    assert runner._cache.plans == []
    assert runner._cache.rows == []
    assert groups.broadcasts == []
    assert runner._cuda_failure.synchronizations == 1


def test_decode_keeps_the_replica_batch_and_cache_namespace(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4), rank=2)
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan([7, 8], [14, 25], [2, 3], [0, 1, 2], phase=BatchPhase.DECODE)

    assert runner.forward_distributed(
        plan,
        _context(groups, BatchPhase.DECODE, ((0, 0), (0, 0), (2, 2), (1, 1)), plan),
    ) == [2, 3]

    assert runner._model.calls == [("decode", [[2], [3]], [14, 25], runner._cache)]
    assert runner._cache.plans == [plan]
    assert groups.broadcasts == []


def test_distributed_metadata_is_validated_before_model_or_cache_work(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan([7], [0], [2], [0, 1], phase=BatchPhase.PREFILL)
    context = _context(groups, BatchPhase.DECODE, ((1, 1),) * 4, plan)

    with pytest.raises(ValueError, match="metadata"):
        runner.forward_distributed(plan, context)

    assert runner._model.calls == []
    assert runner._cache.plans == []


@pytest.mark.parametrize(
    ("slab", "error"),
    [(2, None), (5, "outside its layout")],
)
def test_managed_cache_addresses_are_validated_before_expert_rounds(
    monkeypatch, slab, error
):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    runner.cache_layout = CacheLayout(4, [CacheGroupLayout(128)])
    del runner._cache_plan  # Exercise the production cache-plan validator.
    group = CacheGroupPlan(
        0,
        block_indptr=[0, 1],
        block_slab_ids=[slab],
        block_slotnums=[0],
        block_table_start_indices=[0],
        append_slab_ids=[slab],
        append_slotnums=[0],
        append_block_offsets=[0],
    )
    plan = BatchPlan(
        [7], [0], [2], [0, 1], phase=BatchPhase.PREFILL, cache_groups=[group]
    )
    context = _context(
        groups, BatchPhase.PREFILL, ((1, 1), (0, 0), (0, 0), (0, 0)), plan
    )

    if error is None:
        assert runner.validate_distributed(plan, context) is None
        assert runner.forward_distributed(plan, context) == [2]
        assert runner._model.calls == [("prefill", [[2]], 0, "row-0")]
    else:
        with pytest.raises(ValueError, match=error):
            runner.validate_distributed(plan, context)
        assert runner._model.calls == []
        assert runner._cache.plans == []
        assert runner._cuda_failure.synchronizations == 0


@pytest.mark.parametrize(
    ("parallelism", "options", "message"),
    [
        (Parallelism(world_size=4, ep=4, cp=2), {}, "cp=2"),
        (Parallelism(world_size=4, ep=2), {}, "expert_tp=2"),
        (Parallelism(world_size=4, ep=4, attn_tp=2), {}, "attn_tp=1"),
        (
            Parallelism(world_size=4, ep=4),
            {"speculative_tokens": 1},
            "speculative_tokens=0",
        ),
        (
            Parallelism(world_size=4, ep=4),
            {"cuda_graph_decode": "full"},
            "disabled CUDA graphs",
        ),
        (
            Parallelism(world_size=4, ep=4),
            {"cuda_graph_prefill": "breakable"},
            "disabled CUDA graphs",
        ),
        (
            Parallelism(world_size=4, ep=4),
            {"tensor_parallel": TensorParallelContext()},
            "either",
        ),
    ],
)
def test_unsupported_distributed_preparation_fails_before_cuda(
    parallelism, options, message
):
    kwargs = dict(cuda_graph_prefill="disabled", cuda_graph_decode="disabled")
    kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        DeepSeekV4CudaRunner(
            ".",
            max_seq_len=128,
            parallel_groups=_groups(parallelism),
            device="cuda",
            **kwargs,
        )
