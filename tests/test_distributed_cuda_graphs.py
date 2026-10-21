"""Graph buckets are collective choices across the expert communication domain."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from datetime import timedelta
from typing import NoReturn

import pytest
import torch
import torch.distributed as dist

from resin import BatchPhase, CacheGroupLayout, CacheLayout, Parallelism, RunnerConfig
from resin.backend.cuda.graphs import CudaGraphMode
from resin.backend.cuda.runner import CudaRunner
from resin.backend.distributed import StepContext
from resin.backend.expert_parallel import ExpertParallelDispatch
from resin.backend.topology import ParallelGroups


def _unreachable() -> NoReturn:
    raise AssertionError("fixture graph builder must not run")


class _CpuRunner(CudaRunner):
    """Exercise coordination and graph ownership without constructing CUDA."""

    def forward(self, plan):
        raise AssertionError("this fixture only tests graph coordination")

    def __init__(self, groups, shapes=((1, 4), (2, 4), (4, 4), (4, 8))):
        self.device = torch.device("cpu")
        self.cuda_graph_decode = "full"
        self.cuda_graph_prefill = "breakable"
        self.prefill_chunk_size = 4
        self._prefill_graph_active = False
        self._prefill_operations = ()
        self._prefill_buffers = {}
        self._cuda_failure = lambda: None
        self._cuda_graph_executor = None
        self._model = torch.nn.Module()
        self.runner_config = RunnerConfig()
        self._configure_expert_parallel_graphs(self._model, groups)
        self._configure_cuda_graphs(
            {"decode": CudaGraphMode(shapes, lambda _: _unreachable(), lambda _: {})}
        )


def _context(groups, *, phase=BatchPhase.DECODE, sizes=((1, 2), (3, 6))):
    rows, tokens = sizes[groups.rank]
    return StepContext(groups, 0, phase, sizes, rows, tokens)


def _prefill_rounds(groups):
    """Use real exchanges with CPU launches of the persistent graph closures."""

    runner = _CpuRunner(groups, shapes=())
    received = []

    def local(hidden, indices, weights):
        received.append((hidden.shape[0], int(weights[:, 0].count_nonzero())))
        coefficient = ((indices == groups.rank) * weights).sum(1) * (groups.rank + 1)
        # Nonzero expert output bias makes padded rows observable unless the
        # persistent valid-row scalar suppresses their routing probabilities.
        return (hidden.float() + 1) * coefficient[:, None]

    dispatch = ExpertParallelDispatch(local, num_experts=2, group=groups.ep)
    setattr(runner._model, "dispatch", dispatch)

    def operation(hidden, _ids):
        indices = torch.tensor([0, 1], dtype=torch.int32).expand(hidden.shape[0], 2)
        weights = torch.tensor([0.25, 0.75]).expand(hidden.shape[0], 2)
        return dispatch(hidden, indices, weights)

    runner._configure_prefill_graphs(
        (operation, operation), hidden_size=2, max_tokens=4
    )
    steps = {
        shape: runner._new_prefill_graph_step(shape)
        for shape in runner._require_cuda_graph_executor().mode("prefill").shapes
    }
    replays = []

    def replay(mode, shape, inputs):
        assert mode == "prefill"
        step = steps[shape]
        assert step.inputs is inputs
        replays.append((shape, int(inputs["ep_valid_rows"])))
        step.launch()
        return step.output

    setattr(runner, "_run_cuda_graph", replay)
    for sizes, capacity in (
        (((1, 1), (1, 3)), 4),
        (((0, 0), (1, 2)), 2),
        (((1, 3), (1, 1)), 4),
    ):
        context = _context(groups, phase=BatchPhase.PREFILL, sizes=sizes)
        tokens = context.num_tokens
        hidden = torch.arange(tokens * 2, dtype=torch.bfloat16).reshape(tokens, 2)
        hidden += groups.rank
        with runner._distributed_prefill_graphs(context):
            assert runner._distributed_prefill_bucket == capacity
            if context.placeholder:
                runner._empty_prefill_regions()
            else:
                expected = hidden
                output = hidden
                for region in range(2):
                    output = runner._run_prefill_region(region, output)
                    expected = ((expected.float() + 1) * 1.75).bfloat16()
                torch.testing.assert_close(output, expected)
        assert not runner._prefill_graph_active
        assert runner._distributed_prefill_bucket is None
        assert replays[-2:] == [((capacity, 1), tokens), ((capacity, 2), tokens)]
        assert received[-2:] == [(2 * capacity, sum(n for _, n in sizes))] * 2

    # Packed prefill can exceed a per-sequence chunk limit. Clamping this
    # request to four would select a graph too small for the five live rows.
    # Nested model scopes must preserve the collective decision to run eager.
    context = _context(groups, phase=BatchPhase.PREFILL, sizes=((1, 5), (0, 0)))
    hidden = torch.ones(context.num_tokens, 2, dtype=torch.bfloat16)
    replay_count = len(replays)
    with runner._distributed_prefill_graphs(context):
        assert runner._distributed_prefill_bucket is None
        assert not runner._prefill_graph_active
        with runner._prefill_graphs():
            assert not runner._prefill_graph_active
            expected = hidden
            output = hidden
            for region in range(2):
                output = runner._run_prefill_region(region, output)
                expected = ((expected.float() + 1) * 1.75).bfloat16()
            torch.testing.assert_close(output, expected)
    assert len(replays) == replay_count
    assert runner.cuda_graph_stats["prefill"]["eager_fallback_count"] == 1
    assert received[-2:] == [(5, 5)] * 2
    assert not runner._prefill_graph_active
    assert not runner._distributed_prefill_scope

    # A model that executes one sequence at a time explicitly requests the
    # bounded chunk bucket instead of the total packed token count.
    with runner._distributed_prefill_graphs(context, per_sequence=True):
        assert runner._distributed_prefill_bucket == 4
        assert runner._prefill_graph_active
    assert runner._distributed_prefill_bucket is None


def _selection_worker(rank, rendezvous):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    dist.barrier()
    groups = None
    try:
        groups = ParallelGroups.create(
            Parallelism(world_size=2, ep=2, attn_dp=2), device_backend="gloo"
        )
        runner = _CpuRunner(groups)
        context = _context(groups)

        # A rank with a skinny local batch must launch the same graph as a
        # peer needing a larger batch and wider metadata buffers.
        required = (1, 2) if rank == 0 else (3, 6)
        shape = runner._select_distributed_cuda_graph(
            "decode", context, required_shape=required
        )
        assert shape == (4, 8)

        # Model-level graph eligibility is also collective. Mixing a replay
        # and eager variable-size exchange would deadlock the NCCL domain.
        shape = runner._select_distributed_cuda_graph(
            "decode",
            context,
            required_shape=required,
            eligible=rank != 0,
            fallback="eligibility",
        )
        assert shape is None
        assert runner.cuda_graph_stats["decode"]["eager_fallback_count"] == 1

        required = (5, 1) if rank == 1 else (1, 1)
        assert (
            runner._select_distributed_cuda_graph(
                "decode", context, required_shape=required
            )
            is None
        )
        assert runner.cuda_graph_stats["decode"]["eager_fallback_count"] == 2

        # Idle attention replicas still vote and select their active peer's
        # physical bucket, then can resume with a smaller logical demand.
        context = _context(groups, sizes=((0, 0), (2, 3)))
        required = (0, 0) if rank == 0 else (2, 3)
        assert runner._select_distributed_cuda_graph(
            "decode", context, required_shape=required
        ) == (
            2,
            4,
        )
        assert runner._select_distributed_cuda_graph(
            "decode", context, required_shape=(1, 1)
        ) == (
            1,
            4,
        )

        mismatched = _CpuRunner(
            groups, shapes=((1, 4), (4, 8)) if rank else ((1, 4), (2, 4), (4, 8))
        )
        with pytest.raises((ValueError, RuntimeError), match="agree|catalog|same"):
            mismatched._select_distributed_cuda_graph(
                "decode", context, required_shape=(1, 1)
            )

        # Startup must reject incompatible capture catalogs before any rank
        # enters the executor's first CUDA/NCCL warmup or capture operation.
        launched = []
        mismatched._cuda_graph_executor.capture_all = lambda **_: launched.append(True)
        with pytest.raises((ValueError, RuntimeError), match="agree|catalog|same"):
            mismatched.capture_cuda_graphs()
        assert not launched

        policy = _CpuRunner(groups)
        if rank:
            policy.cuda_graph_decode = "breakable"
        policy._cuda_graph_executor.capture_all = lambda **_: launched.append(True)
        with pytest.raises((ValueError, RuntimeError), match="agree|policy|same"):
            policy.capture_cuda_graphs()
        assert not launched

        # Equal physical graph shapes do not imply the same expert schedule:
        # ordinary decode and different draft lengths have different rounds.
        for lookaheads in ((0, 2), (2, 5)):
            speculative = _CpuRunner(groups)
            speculative.runner_config = RunnerConfig(
                cache_layout=CacheLayout(64, [CacheGroupLayout(2)]),
                cache_write_lookahead_tokens=lookaheads[rank],
            )
            speculative._cuda_graph_executor.capture_all = lambda **_: launched.append(
                True
            )
            with pytest.raises(ValueError, match="agree"):
                speculative.capture_cuda_graphs()
            assert not launched

        # An agreed catalog reaches capture on both peers, rather than being
        # rejected simply because the attention replicas have unequal work.
        runner._cuda_graph_executor.capture_all = lambda **_: launched.append(True)
        runner.capture_cuda_graphs()
        assert launched == [True]
        _prefill_rounds(groups)
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_graph_selection_and_startup_capture_agree_across_attention_replicas(tmp_path):
    context = multiprocessing.get_context("spawn")
    rendezvous = str(tmp_path / "graph-domain")
    processes = [
        context.Process(target=_selection_worker, args=(rank, rendezvous))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
