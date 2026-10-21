from __future__ import annotations

from contextlib import contextmanager, nullcontext
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
    SamplingParams,
)
from resin.backend.cuda.graphs import CudaGraphMode
from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.cuda.runner import CudaRunner
from resin.backend.cuda.sampling import FlashInferSampler
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
    for name, source in (
        ("device_replica", groups.replica),
        ("device_domain", groups.domain),
    ):
        setattr(
            groups,
            name,
            RankGroup(source.ranks, (name, source.ranks), source.local_rank),
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
    runner.cuda_graph_prefill = "disabled"
    runner.cuda_graph_decode = "disabled"
    runner._prefill_graph_active = False
    runner._model = _Model()
    runner._cache = _Cache()
    runner._cache_plan = lambda plan: plan
    runner._cuda_failure = _Guard()
    runner._prefill_compute_context = nullcontext
    runner._distributed_prefill_graphs = lambda context, **kwargs: nullcontext()
    runner._select_distributed_cuda_graph = lambda *args, **kwargs: None
    runner._check_cuda_graphs = lambda phase: None
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


@pytest.mark.parametrize("policy", ["disabled", "breakable"])
def test_ragged_prefill_replicas_enter_the_same_expert_rounds(monkeypatch, policy):
    parallelism = Parallelism(world_size=4, ep=4, attn_dp=2)
    plans = (
        BatchPlan([7, 8], [0, 9], [2, 3, 4, 5, 6], [0, 3, 5], phase=BatchPhase.PREFILL),
        BatchPlan([7], [0], [9], [0, 1], phase=BatchPhase.PREFILL),
    )
    runners = []
    for rank, plan in zip((0, 2), plans, strict=True):
        groups = _groups(parallelism, rank)
        runner = _runner(monkeypatch, groups)
        runner.cuda_graph_prefill = policy
        output = runner.forward_distributed(
            plan, _context(groups, BatchPhase.PREFILL, ((2, 5), (1, 1)), plan)
        )
        assert output == ([4, 6] if rank == 0 else [9])
        assert runner._cache.plans == [plan]
        assert groups.broadcasts == [(rank, groups.device_replica.handle, output)]
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


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_eager_decode_preserves_fixed_row_gemm_selection(monkeypatch, phase):
    groups = _groups(Parallelism(world_size=2, ep=2))
    runner = _runner(monkeypatch, groups)
    events = []

    @contextmanager
    def dynamic_prefill_gemms():
        events.append("enter_prefill")
        yield
        events.append("leave_prefill")

    runner._prefill_compute_context = dynamic_prefill_gemms
    method_name = "forward_prefill" if phase == BatchPhase.PREFILL else "forward_decode"
    original = getattr(runner._model, method_name)

    def execute(*args, **kwargs):
        events.append(method_name)
        assert torch.get_default_dtype() == torch.bfloat16
        return original(*args, **kwargs)

    setattr(runner._model, method_name, execute)
    plan = BatchPlan([7], [0], [2], [0, 1], phase=phase)
    previous_dtype = torch.get_default_dtype()
    assert runner.forward_distributed(
        plan, _context(groups, phase, ((1, 1),), plan)
    ) == [2]
    assert events == (
        ["enter_prefill", "forward_prefill", "leave_prefill"]
        if phase == BatchPhase.PREFILL
        else ["forward_decode"]
    )
    assert torch.get_default_dtype() == previous_dtype


def test_distributed_metadata_is_validated_before_model_or_cache_work(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan([7], [0], [2], [0, 1], phase=BatchPhase.PREFILL)
    context = _context(groups, BatchPhase.DECODE, ((1, 1),) * 4, plan)

    with pytest.raises(ValueError, match="metadata"):
        runner.forward_distributed(plan, context)

    assert runner._model.calls == []
    assert runner._cache.plans == []


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
@pytest.mark.parametrize("idle", [False, True])
def test_missing_graph_capture_is_rejected_in_worker_readiness_validation(
    monkeypatch, phase, idle
):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    runner.cuda_graph_decode = "full"
    runner.cuda_graph_prefill = "breakable"
    runner._cuda_graph_executor = None
    pending = "prefill" if phase == BatchPhase.PREFILL else "decode"
    runner._configure_cuda_graphs(
        {
            name: CudaGraphMode(
                ((1, 1),) if name == pending else (),
                lambda _: None,
                lambda _: {},
            )
            for name in ("prefill", "decode")
        }
    )
    runner._check_cuda_graphs = CudaRunner._check_cuda_graphs.__get__(runner)
    plan = None if idle else BatchPlan([7], [0], [2], [0, 1], phase=phase)
    context = _context(
        groups, phase, ((0, 0) if idle else (1, 1), (1, 1), (0, 0), (0, 0)), plan
    )

    with pytest.raises(RuntimeError, match="not been captured"):
        runner.validate_distributed(plan, context)

    assert runner._model.calls == runner._cache.plans == []
    assert runner._cuda_failure.synchronizations == 0


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
        (
            Parallelism(world_size=4, ep=4),
            {"speculative_tokens": 1},
            "speculative_tokens=0",
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


@pytest.mark.parametrize(
    "parallelism",
    [
        Parallelism(world_size=4, ep=4, attn_dp=4),
        Parallelism(world_size=4, ep=2),
        Parallelism(world_size=4, ep=4, attn_tp=2),
        Parallelism(world_size=4, ep=2, cp=2),
        Parallelism(world_size=8, attn_dp=2, attn_tp=2, cp=2, ep=2),
    ],
)
def test_distributed_graph_policies_and_composed_meshes_pass_preparation(
    monkeypatch, parallelism
):
    import resin_models_cuda.deepseek_v4.cuda as implementation

    class CudaEntry(Exception):
        pass

    def enter_cuda(self, **kwargs):
        assert kwargs["cuda_graph_prefill"] == "breakable"
        assert kwargs["cuda_graph_decode"] == "full"
        raise CudaEntry

    monkeypatch.setattr(implementation.CudaRunner, "__init__", enter_cuda)
    with pytest.raises(CudaEntry):
        DeepSeekV4CudaRunner(
            ".",
            max_seq_len=128,
            parallel_groups=_groups(parallelism),
            device="cuda",
        )


def test_distributed_decode_graph_uses_the_domain_selected_bucket(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4), rank=2)
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan([7, 8], [14, 25], [2, 3], [0, 1, 2], phase=BatchPhase.DECODE)
    context = _context(
        groups, BatchPhase.DECODE, ((0, 0), (0, 0), (2, 2), (1, 1)), plan
    )
    selected, replayed = [], []

    def select(mode, context_arg, **kwargs):
        selected.append((mode, context_arg, kwargs))
        return (4, 128)

    def replay(plan_arg, cache_plan, *, shape):
        replayed.append((plan_arg, cache_plan, shape))
        return [12, 13]

    runner._select_distributed_cuda_graph = select
    runner._graph_forward = replay

    assert runner.forward_distributed(plan, context) == [12, 13]
    assert selected == [("decode", context, {"required_shape": (2, 26)})]
    assert replayed == [(plan, plan, (4, 128))]
    assert runner._cache.plans == []
    assert runner._model.calls == []


def test_idle_decode_replica_replays_zero_route_graph_on_scratch_cache(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    context = _context(
        groups, BatchPhase.DECODE, ((0, 0), (2, 2), (0, 0), (0, 0)), None
    )
    selected, replayed = [], []

    def select(mode, context_arg, **kwargs):
        selected.append((mode, context_arg, kwargs))
        return (2, 128)

    runner._select_distributed_cuda_graph = select
    runner._decode_graph_capture_inputs = lambda shape: {
        "scratch_shape": shape,
        "ep_valid_rows": shape[0],
    }
    runner._run_cuda_graph = lambda mode, shape, inputs: replayed.append(
        (mode, shape, inputs)
    )

    assert runner.forward_distributed(None, context) is None
    assert selected == [("decode", context, {"required_shape": (0, 0)})]
    assert replayed == [
        ("decode", (2, 128), {"scratch_shape": (2, 128), "ep_valid_rows": 0})
    ]
    assert runner._cache.plans == []
    assert runner._model.calls == []


def test_missing_prefill_rounds_join_the_same_captured_ffn_regions(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    runner.cuda_graph_prefill = "breakable"
    rounds = []
    runner._empty_prefill_regions = lambda: rounds.append("captured-ffns")

    @contextmanager
    def captured_regions(context, **kwargs):
        runner._prefill_graph_active = True
        try:
            yield
        finally:
            runner._prefill_graph_active = False

    runner._distributed_prefill_graphs = captured_regions
    plan = BatchPlan([7], [0], [2, 3], [0, 2], phase=BatchPhase.PREFILL)
    context = _context(
        groups, BatchPhase.PREFILL, ((1, 2), (3, 5), (0, 0), (0, 0)), plan
    )

    assert runner.forward_distributed(plan, context) == [3]
    assert rounds == ["captured-ffns", "captured-ffns"]
    assert [call[0] for call in runner._model.calls] == ["prefill"]


def test_decode_graph_formats_logprobs_from_retained_graph_logits(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=4))
    runner = _runner(monkeypatch, groups)
    shape = (4, 128)
    runner.vocab_size = 3
    runner._graph_device_caches = {shape: "graph-cache"}
    runner._graph_logits = {
        shape: torch.tensor([[1.0, 3.0, 0.0], [2.0, 1.0, 4.0], [0.0] * 3, [0.0] * 3])
    }
    runner._dspark_phase_events = {}
    runner._sampler = FlashInferSampler
    runner._cache.graph_plan_inputs = lambda *args: {}
    replayed = []

    def replay(mode, shape_arg, inputs):
        replayed.append((mode, shape_arg, inputs))
        return [1, 2, 0, 0]

    runner._run_cuda_graph = replay
    plan = BatchPlan(
        [7, 8],
        [14, 25],
        [0, 0],
        [0, 1, 2],
        phase=BatchPhase.DECODE,
        sampling_params=[SamplingParams(logprobs=True, seed=1)] * 2,
    )

    output = runner._graph_forward(plan, "cache-plan", shape=shape)

    assert output.token_ids.tolist() == [1, 2]
    expected = runner._graph_logits[shape][:2].log_softmax(-1)
    assert output.logprobs.tolist() == pytest.approx([expected[0, 1], expected[1, 2]])
    assert len(replayed) == 1
    assert replayed[0][2]["ep_valid_rows"] == 2


def test_decode_graph_launch_fixes_expert_capacity_and_replica_broadcast(monkeypatch):
    groups = _groups(Parallelism(world_size=4, ep=4, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    runner.vocab_size = 3
    runner._graph_device_caches = {}
    runner._graph_logits = {}
    runner._cache.graph_cache = lambda *args: "graph-cache"
    runner._cache.graph_device_inputs = lambda cache: {}
    runner._sampler = SimpleNamespace(
        empty=lambda *args: object(),
        graph_inputs=lambda sampling: {},
        sample_tensors=lambda logits, sampling, *, out: out.copy_(logits.argmax(-1)),
    )
    logits = torch.tensor([[1.0, 3.0, 0.0], [2.0, 1.0, 4.0]])
    scopes = []

    @contextmanager
    def capacity(rows, *, valid_rows):
        scopes.append((rows, int(valid_rows)))
        yield

    runner._expert_graph_capacity = capacity
    runner._model.forward_decode = lambda *args, **kwargs: logits
    step = runner._new_decode_graph_step((2, 128))
    step.inputs["ep_valid_rows"].fill_(1)

    step.launch()

    assert scopes == [(2, 1)]
    assert step.output.tolist() == [1, 2]
    assert groups.broadcasts == [(0, groups.device_replica.handle, [1, 2])]
    torch.testing.assert_close(runner._graph_logits[(2, 128)], logits)
