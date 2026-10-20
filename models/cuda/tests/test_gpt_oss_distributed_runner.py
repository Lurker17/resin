from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resin import BatchPhase, BatchPlan, Parallelism
from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.distributed import StepContext
from resin.backend.topology import RankGroup
from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner


def _groups(parallelism, rank=0):
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


class _Guard(nullcontext):
    def __init__(self):
        super().__init__()
        self.synchronizations = 0

    def __call__(self):
        self.synchronizations += 1


def _runner(monkeypatch, groups):
    import resin_models_cuda.gpt_oss.cuda as implementation

    runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
    runner.parallel_groups = groups
    runner.tensor_parallel = TensorParallelContext()
    runner.device = torch.device("cpu")
    runner.hidden_size = 32
    runner.cache_layout = object()
    runner.cuda_graph_prefill = "disabled"
    runner._prefill_graph_active = False
    runner._cuda_failure = _Guard()
    runner._check_cuda_graphs = lambda _phase: None
    runner._graph_forward = lambda *_args: None
    runner._sampler = lambda logits, plan: logits.argmax(dim=-1)
    runner.calls = []
    runner.tensorized = []
    runner._model = SimpleNamespace(
        lm_head=lambda hidden: hidden,
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        experts=SimpleNamespace(
                            forward_empty=lambda index=index: runner.calls.append(
                                ("empty", index)
                            )
                        )
                    )
                )
                for index in range(3)
            ]
        ),
    )

    def batches(plan, cache_layout):
        assert cache_layout is runner.cache_layout
        batch = SimpleNamespace(
            sequence_indices=tuple(range(plan.num_sequences)),
            last_token_indices=torch.tensor(plan.token_indptr[1:].astype(np.int64) - 1),
            phase=plan.phase,
        )
        return ((batch,), None) if plan.phase == BatchPhase.PREFILL else ((), batch)

    def tensorize(host, staging, prefix):
        runner.tensorized.append(prefix)
        return SimpleNamespace(
            sequence_indices=torch.tensor(host.sequence_indices),
            last_token_indices=host.last_token_indices,
        )

    def forward(token_ids, positions, batch):
        runner.calls.append(("batch", len(token_ids), positions.tolist()))
        return torch.nn.functional.one_hot(
            token_ids[batch.last_token_indices], num_classes=runner.hidden_size
        ).bfloat16()

    monkeypatch.setattr(implementation, "build_attention_batches", batches)
    monkeypatch.setattr(
        implementation,
        "TensorStager",
        lambda _device: SimpleNamespace(
            device=lambda _name, values, dtype: torch.tensor(
                values.tolist(), dtype=dtype
            )
        ),
    )
    runner._tensorize = tensorize
    runner._forward_batch = forward
    return runner


def _context(groups, phase, sizes, plan):
    return StepContext(
        groups=groups,
        step_id=0,
        phase=phase,
        replica_sizes=sizes,
        num_sequences=0 if plan is None else plan.num_sequences,
        num_tokens=0 if plan is None else len(plan.token_ids),
    )


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_distributed_forward_packs_unequal_replica_rows_once(monkeypatch, phase):
    topology = Parallelism(world_size=2, ep=2, attn_dp=2)
    if phase == BatchPhase.PREFILL:
        plans = (
            BatchPlan([7, 8], [0, 9], [2, 3, 4, 5, 6], [0, 3, 5], phase=phase),
            BatchPlan([7], [0], [9], [0, 1], phase=phase),
        )
        sizes = ((2, 5), (1, 1))
    else:
        plans = (
            BatchPlan([7, 8], [11, 15], [4, 6], [0, 1, 2], phase=phase),
            BatchPlan([7], [3], [9], [0, 1], phase=phase),
        )
        sizes = ((2, 2), (1, 1))
    for rank, plan in enumerate(plans):
        groups = _groups(topology, rank)
        runner = _runner(monkeypatch, groups)
        assert runner.forward_distributed(
            plan, _context(groups, phase, sizes, plan)
        ) == ([4, 6] if rank == 0 else [9])
        assert runner.calls == [("batch", len(plan.token_ids), plan.positions.tolist())]
        assert runner.tensorized == [
            "prefill.0" if phase == BatchPhase.PREFILL else "decode"
        ]
        assert groups.broadcasts == []


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_idle_replica_joins_each_expert_layer_without_attention_or_cache(
    monkeypatch, phase
):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)

    assert (
        runner.forward_distributed(
            None, _context(groups, phase, ((0, 0), (1, 1)), None)
        )
        is None
    )

    assert runner.calls == [("empty", 0), ("empty", 1), ("empty", 2)]
    assert runner.tensorized == []
    assert groups.broadcasts == []
    assert runner._cuda_failure.synchronizations == 1


def test_sample_broadcast_uses_attention_replication_group(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2), rank=1)
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan([7], [3], [9], [0, 1], phase=BatchPhase.DECODE)

    assert runner.forward_distributed(
        plan, _context(groups, BatchPhase.DECODE, ((1, 1),), plan)
    ) == [9]

    assert groups.broadcasts == [(0, groups.attn_replication.handle, [9])]


def test_distributed_metadata_mismatch_is_rejected_before_execution(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan([7], [0], [2], [0, 1], phase=BatchPhase.PREFILL)

    with pytest.raises(ValueError, match="metadata"):
        runner.forward_distributed(
            plan, _context(groups, BatchPhase.DECODE, ((1, 1), (0, 0)), plan)
        )
    assert runner.calls == []
    assert runner.tensorized == []


@pytest.mark.parametrize(
    ("parallelism", "options", "message"),
    [
        (Parallelism(world_size=2, ep=2, cp=2), {}, "cp=2"),
        (Parallelism(world_size=2), {}, "expert_tp=1"),
        (Parallelism(world_size=2, ep=2, attn_tp=2), {}, "attn_tp=1"),
        (
            Parallelism(world_size=2, ep=2),
            {"cuda_graph_decode": "full"},
            "disabled CUDA graphs",
        ),
        (
            Parallelism(world_size=2, ep=2),
            {"cuda_graph_prefill": "breakable"},
            "disabled CUDA graphs",
        ),
        (
            Parallelism(world_size=2, ep=2),
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
        GptOssCudaRunner(device="cuda", parallel_groups=_groups(parallelism), **kwargs)
