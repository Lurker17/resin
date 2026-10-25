"""Execute runner phase boundaries on CPU; inspect real target head operator M."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from resin import BatchPhase, BatchPlan
from resin.backend.topology import ParallelGroups
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash import cuda as runner_module
from resin_models_cuda.glm5_3_flash import model as model_module
from resin_models_cuda.glm5_3_flash.cache import LayerCache
from resin_models_cuda.glm5_3_flash.cuda import Glm53FlashCudaRunner


@pytest.mark.parametrize("drafts", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("capacity", [None, 2, 8, 64])
def test_constructor_target_profile_is_shared_across_capacity_and_draft_width(
    monkeypatch, drafts, capacity
):
    # Keep actual capacity/profile/catalog calculation; replace only weights,
    # allocation and graph registration at the CPU boundary.
    monkeypatch.setattr(
        runner_module.CudaRunner,
        "__init__",
        lambda self, **kwargs: setattr(self, "device", torch.device("cpu")),
    )
    monkeypatch.setattr(runner_module, "resolve_model_path", lambda value: value)
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(layers=(), mtp=None), Glm53Config(), {}

    monkeypatch.setattr(runner_module, "load_model", load)
    for name in ("_configure_parallel_graphs", "_configure_cache_binding"):
        monkeypatch.setattr(runner_module.CudaRunner, name, lambda *args: None)
    monkeypatch.setattr(
        runner_module.CudaRunner,
        "_configure_prefill_graphs",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        runner_module.CudaRunner,
        "_configure_cuda_graphs",
        lambda self, modes: setattr(self, "observed_modes", modes),
    )
    monkeypatch.setattr(
        runner_module,
        "CacheArena",
        lambda *args, **kwargs: SimpleNamespace(binding=None),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (128 * 1024**3,) * 2)
    options = {} if capacity is None else {"max_batch_size": capacity}
    runner = Glm53FlashCudaRunner(
        "unused",
        device="cpu",
        max_seq_len=4096,
        prefill_chunk_size=512,
        speculative_tokens=drafts,
        **options,
    )
    batch = 8 if capacity is None else capacity
    assert runner.max_batch_size == batch
    assert runner.target_projection_rows == 64
    assert runner.max_seq_len == 4096
    assert runner.runner_config.cache_write_lookahead_tokens == (
        drafts + 1 if drafts else 0
    )
    assert loads[0]["max_tokens"] == max(512, batch * (drafts + 1))
    assert getattr(runner, "observed_modes")["decode"].shapes == tuple(
        (size, 4096) for size in (1, 2, 4, 8, 12, 16, 24, 32, 64) if size <= batch
    )


class _DeviceCache:
    def __init__(self, hidden):
        self.hidden = hidden
        self.batches = []

    def batch(self, positions, sequences, offsets, contexts, cpu_offsets=None):
        layer = LayerCache(
            positions,
            sequences,
            offsets,
            contexts,
            contexts + offsets[1:] - offsets[:-1],
        )
        batch = SimpleNamespace(
            positions=positions,
            sequence_indices=sequences,
            cu_seqlens=offsets,
            cu_seqlens_cpu=cpu_offsets,
            context_lengths=contexts,
            lengths=layer.lengths,
            previous_hidden=torch.ones(contexts.numel(), self.hidden),
            is_decode=False,
            projection_rows=0,
            layers={0: layer},
            layer=lambda _: layer,
            store=lambda value: None,
        )
        self.batches.append(batch)
        return batch


class _Mtp(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.head_norm = torch.ones(hidden)
        self.calls = []

    def forward(self, embedded, previous, positions, cache):
        self.calls.append((embedded.shape[0], getattr(cache, "projection_rows", 0)))
        return embedded * 0.25 + previous


class _Sampler:
    def empty(self, rows, device):
        return SimpleNamespace()

    def graph_inputs(self, values):
        return {}

    def sample_tensors(self, logits, values, *, out):
        out.copy_(logits.argmax(-1))

    def __call__(self, logits, plan, **kwargs):
        return logits.argmax(-1)


def _cpu_runner(monkeypatch, drafts, capacity):
    runner = object.__new__(Glm53FlashCudaRunner)
    runner.device = torch.device("cpu")
    runner.max_batch_size = capacity
    runner.target_projection_rows = 64
    runner._projection_batch_buckets = tuple(
        size for size in (1, 2, 4, 8, 12, 16, 24, 32, 64) if size <= capacity
    )
    runner.speculative_tokens = drafts
    runner.parallel_groups = None
    runner.vocab_size = 32
    runner._graph_device_caches, runner._graph_logits = {}, {}
    monkeypatch.setattr(runner, "_new_phase_timing", lambda *args: None)
    monkeypatch.setattr(runner, "_eager_phase_timing", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_speculation",
        SimpleNamespace(benchmark_accepted_drafts=None),
        raising=False,
    )
    monkeypatch.setattr(
        runner, "_format_decode", lambda output, logits, plan: (output, logits)
    )
    capacities = []

    def capacity_scope(rows, **kwargs):
        capacities.append(rows)
        return nullcontext()

    monkeypatch.setattr(runner, "_parallel_graph_capacity", capacity_scope)
    device_cache = _DeviceCache(4)
    monkeypatch.setattr(
        runner,
        "_cache",
        SimpleNamespace(
            graph_cache=lambda *args: device_cache,
            tensorize=lambda *args: device_cache,
            graph_device_inputs=lambda cache: {},
            graph_shape_capture_inputs=lambda *args: {},
            reset_null=lambda: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(runner, "_sampler", _Sampler(), raising=False)
    monkeypatch.setattr(
        runner_module,
        "TensorStager",
        lambda device: SimpleNamespace(
            device=lambda name, values, dtype: torch.tensor(
                np.asarray(values), dtype=dtype, device="cpu"
            )
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "graph_sample",
        lambda sampler, logits, values, offset: logits.argmax(-1),
    )
    config = SimpleNamespace(hc_mult=4, rms_norm_eps=1e-6, num_hidden_layers=0)
    generator = torch.Generator().manual_seed(921)
    mtp = _Mtp(4) if drafts else None
    model = model_module.Glm53Model(
        config,
        torch.randn(32, 4, generator=generator),
        torch.ones(4),
        torch.randn(32, 4, generator=generator),
        [],
        mtp=mtp,
    )
    target_calls, head_rows = [], []
    original_hidden, original_linear = model.forward_hidden, model_module.F.linear

    def observe_hidden(ids, positions, cache, **kwargs):
        target_calls.append((ids.numel(), cache.is_decode, cache.projection_rows))
        return original_hidden(ids, positions, cache, **kwargs)

    def observe_linear(value, weight, *args, **kwargs):
        if weight is model.head:
            head_rows.append(value.shape[0])
        return original_linear(value, weight, *args, **kwargs)

    monkeypatch.setattr(model, "forward_hidden", observe_hidden)
    monkeypatch.setattr(model_module.F, "linear", observe_linear)
    runner._model = model
    return runner, device_cache, target_calls, head_rows, capacities


def _plan(rows, phase):
    tokens = 3 if phase == BatchPhase.PREFILL else 1
    return BatchPlan(
        sequence_ids=list(range(1, rows + 1)),
        context_lengths=[0 if phase == BatchPhase.PREFILL else 17] * rows,
        token_ids=list(range(1, tokens + 1)) * rows,
        token_indptr=[index * tokens for index in range(rows + 1)],
        phase=phase,
    )


@pytest.mark.parametrize(
    "capacity,bucket",
    [
        (2, 2),
        (8, 8),
        (64, 1),
        (64, 8),
        (64, 12),
        (64, 16),
        (64, 24),
        (64, 32),
        (64, 64),
    ],
)
@pytest.mark.parametrize("drafts", [0, 1, 3, 5, 7])
def test_full_graph_launch_target_head_uses_selected_bucket_not_max_capacity(
    monkeypatch, capacity, bucket, drafts
):
    runner, cache, calls, heads, capacities = _cpu_runner(monkeypatch, drafts, capacity)
    # Execute the actual capture closure on CPU, retaining the real model head
    # and speculative orchestration. GPU capture itself is tested separately.
    step = runner._new_decode_graph_step((bucket, 4096))
    projection_rows = max(64, bucket * 8)
    step.inputs["token_ids"].fill_(1)
    step.inputs["positions"].fill_(17)
    if drafts:
        step.inputs["fixed_acceptance"].fill_(-1)
    step.launch()
    target_rows = bucket * (drafts + 1)
    assert calls == [(target_rows, True, projection_rows)]
    assert heads == [bucket] * drafts + [projection_rows]
    assert capacities == ([bucket] * drafts + [target_rows] * 2 if drafts else [bucket])
    assert len(cache.batches) == 1
    assert cache.batches[0].positions.numel() == target_rows
    if drafts:
        assert runner._model.mtp.calls == [(bucket, 0)] * drafts + [(target_rows, 0)]
    # Reusing a prepared launch with changed values cannot inherit a draft phase.
    calls.clear()
    heads.clear()
    step.inputs["token_ids"].fill_(7)
    step.inputs["positions"].fill_(63)
    step.launch()
    assert calls == [(target_rows, True, projection_rows)] and heads == [
        bucket
    ] * drafts + [projection_rows]


@pytest.mark.parametrize("capacity", [2, 8])
@pytest.mark.parametrize("drafts", [0, 1, 3, 5, 7])
def test_eager_prefill_decode_prefill_preserves_head_and_mtp_phase_isolation(
    monkeypatch, capacity, drafts
):
    runner, cache, calls, heads, _ = _cpu_runner(monkeypatch, drafts, capacity)
    for phase in (BatchPhase.PREFILL, BatchPhase.DECODE, BatchPhase.PREFILL):
        calls.clear()
        heads.clear()
        if drafts:
            runner._model.mtp.calls.clear()
        runner._eager(_plan(capacity, phase), object())
        if phase == BatchPhase.PREFILL:
            assert calls == [(capacity * 3, False, 0)]
            assert heads == [capacity]
            if drafts:
                assert runner._model.mtp.calls == [(capacity * 2, 0)]
        else:
            target_rows = capacity * (drafts + 1)
            assert calls == [(target_rows, True, 64)]
            assert heads == [capacity] * drafts + [64]
            if drafts:
                assert runner._model.mtp.calls == [(capacity, 0)] * drafts + [
                    (target_rows, 0)
                ]
    assert cache.batches[-1].projection_rows == 0


@pytest.mark.parametrize("drafts", [0, 3, 5])
@pytest.mark.parametrize(
    "rows,expected",
    [
        (1, 64),
        (8, 64),
        (9, 96),
        (12, 96),
        (16, 128),
        (17, 192),
        (24, 192),
        (32, 256),
        (64, 512),
    ],
)
def test_eager_decode_uses_batch_catalog_without_max_capacity_padding(
    monkeypatch, drafts, rows, expected
):
    runner, _, calls, heads, _ = _cpu_runner(monkeypatch, drafts, 64)
    runner._eager(_plan(rows, BatchPhase.DECODE), object())
    assert calls == [(rows * (drafts + 1), True, expected)]
    assert heads == [rows] * drafts + [expected]


@pytest.mark.parametrize("drafts", [0, 3, 5])
@pytest.mark.parametrize(
    "local_rows,global_rows,expected",
    [(1, 8, 64), (1, 9, 96), (8, 16, 128), (1, 17, 192), (1, 32, 256), (8, 64, 512)],
)
def test_distributed_eager_fallback_uses_existing_global_request_count(
    monkeypatch, drafts, local_rows, global_rows, expected
):
    from resin.backend.distributed import StepContext

    runner, _, calls, heads, _ = _cpu_runner(monkeypatch, drafts, 64)
    runner._cuda_failure = nullcontext()
    monkeypatch.setattr(
        runner, "_distributed_graph", lambda *args, **kwargs: (False, None)
    )
    plan = _plan(local_rows, BatchPhase.DECODE)
    context = StepContext(
        groups=cast(ParallelGroups, SimpleNamespace()),
        step_id=1,
        phase=BatchPhase.DECODE,
        replica_sizes=(
            (local_rows, local_rows),
            (global_rows, global_rows),
            (0, 0),
            (0, 0),
        ),
        num_sequences=local_rows,
        num_tokens=local_rows,
    )
    runner.forward_distributed(
        plan, context, prepared=SimpleNamespace(cache_plan=object())
    )
    assert calls == [(local_rows * (drafts + 1), True, expected)]
    assert heads == [local_rows] * drafts + [expected]
    # Selection is pure CPU metadata and never changes the runner's floor.
    assert runner.target_projection_rows == 64


@pytest.mark.parametrize("drafts", [0, 3, 5])
@pytest.mark.parametrize("idle", [False, True])
def test_distributed_graph_replays_agreed_bucket_for_active_and_idle_ranks(
    monkeypatch, drafts, idle
):
    runner, _, calls, heads, _ = _cpu_runner(monkeypatch, drafts, 64)
    runner.parallel_groups = SimpleNamespace()
    bucket = 16
    step = runner._new_decode_graph_step((bucket, 4096))
    selections = []

    def select(mode, context, *, required_shape):
        selections.append(required_shape)
        return (bucket, 4096)

    def replay(mode, shape, inputs):
        assert shape == (bucket, 4096)
        for name, destination in step.inputs.items():
            if name in inputs:
                destination.copy_(
                    torch.as_tensor(inputs[name]).reshape(destination.shape)
                )
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_select_distributed_cuda_graph", select)
    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    monkeypatch.setattr(runner_module, "pad_sampling_inputs", lambda *args: {})
    monkeypatch.setattr(
        runner._cache, "graph_plan_inputs", lambda *args: {}, raising=False
    )
    plan = None if idle else _plan(1, BatchPhase.DECODE)
    used, output = runner._distributed_graph(
        plan, object(), SimpleNamespace(cache_plan=object())
    )
    assert used and (output is None) is idle
    assert selections == [(0, 0) if idle else (1, 18)]
    assert calls == [(bucket * (drafts + 1), True, 128)]
    assert heads == [bucket] * drafts + [128]
    assert step.inputs["ep_valid_rows"].tolist() == (0 if idle else 1)


@pytest.mark.parametrize("drafts", [0, 3, 5])
def test_distributed_idle_eager_only_joins_experts_without_target_projection(
    monkeypatch, drafts
):
    from resin.backend.distributed import StepContext

    runner, _, calls, heads, _ = _cpu_runner(monkeypatch, drafts, 64)
    runner._cuda_failure = nullcontext()
    runner._prefill_graph_active = False
    monkeypatch.setattr(
        runner, "_distributed_graph", lambda *args, **kwargs: (False, None)
    )
    expert_calls = []
    monkeypatch.setattr(
        runner._model, "forward_experts_empty", lambda: expert_calls.append("target")
    )
    monkeypatch.setattr(
        runner._model, "forward_mtp_experts_empty", lambda: expert_calls.append("mtp")
    )
    context = StepContext(
        groups=cast(ParallelGroups, SimpleNamespace()),
        step_id=1,
        phase=BatchPhase.DECODE,
        replica_sizes=((0, 0), (16, 16), (0, 0), (0, 0)),
        num_sequences=0,
        num_tokens=0,
    )
    assert (
        runner.forward_distributed(
            None, context, prepared=SimpleNamespace(cache_plan=None)
        )
        is None
    )
    assert not calls and not heads
    assert expert_calls == ["mtp"] * drafts + ["target"] + (["mtp"] if drafts else [])


def test_eager_projection_catalog_honors_custom_buckets_and_empty_catalog(monkeypatch):
    runner, _, calls, heads, _ = _cpu_runner(monkeypatch, 3, 64)
    runner._projection_batch_buckets = (1, 8, 32, 64)
    runner._eager(_plan(9, BatchPhase.DECODE), object())
    assert calls == [(36, True, 256)] and heads[-1] == 256
    calls.clear()
    heads.clear()
    runner._projection_batch_buckets = ()
    for rows, expected in ((9, 128), (17, 256)):
        calls.clear()
        heads.clear()
        runner._eager(_plan(rows, BatchPhase.DECODE), object())
        assert calls == [(rows * 4, True, expected)] and heads[-1] == expected
    runner._projection_batch_buckets = (1, 8)
    calls.clear()
    heads.clear()
    runner._eager(_plan(17, BatchPhase.DECODE), object())
    assert calls == [(68, True, 256)] and heads[-1] == 256
    with pytest.raises(ValueError, match="omit local requests"):
        runner._eager(_plan(9, BatchPhase.DECODE), object(), decode_batch_size=8)
