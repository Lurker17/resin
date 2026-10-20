"""Synchronous serving over shared ranks with attention and expert meshes.

The Rust Engine retains scheduling and page ownership. This module coordinates
its existing step boundary; model runtimes own partitioning and device collectives.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

from resin import BatchPhase, BatchPlan, Engine, ForwardOutput, Runner, RunnerConfig
from resin.backend.parallel import batch_plan_from_payload, batch_plan_payload
from resin.backend.topology import ParallelGroups, RankGroup


class DistributedExecutionError(RuntimeError):
    """A coordinated failure is fatal: another rank may have changed model state."""

    _resin_post_accept = True


def _message(error: BaseException | None) -> str | None:
    return None if error is None else f"{type(error).__name__}: {error}"


def _first_error(group: RankGroup, errors) -> str | None:
    return next(
        (
            f"rank {rank}: {error}"
            for rank, error in zip(group.ranks, errors, strict=True)
            if error
        ),
        None,
    )


def _runner_signature(config: RunnerConfig):
    layout = config.cache_layout
    geometry = (
        None
        if layout is None
        else (
            layout.parent_capacity,
            tuple(
                (
                    group.block_tokens,
                    group.blocks_per_slab,
                    int(group.family),
                    int(group.retention),
                    group.window_tokens,
                    group.read_window_tokens,
                )
                for group in layout.groups
            ),
        )
    )
    return (
        config.max_batch_tokens,
        config.prefill_chunk_size,
        config.continuation_prefill,
        config.cache_write_lookahead_tokens,
        geometry,
    )


@dataclass(frozen=True)
class StepContext:
    """Logical replica capacities; CP sharding and graph buckets belong to the model."""

    groups: ParallelGroups
    step_id: int
    phase: BatchPhase
    replica_sizes: tuple[tuple[int, int], ...]
    num_sequences: int
    num_tokens: int

    @property
    def placeholder(self) -> bool:
        return self.num_sequences == 0

    @property
    def max_num_sequences(self) -> int:
        return max(rows for rows, _ in self.replica_sizes)

    @property
    def max_num_tokens(self) -> int:
        return max(tokens for _, tokens in self.replica_sizes)


class DistributedRunner(Protocol):
    """Model-owned execution over explicit process groups, including empty ranks."""

    runner_config: RunnerConfig

    def forward_distributed(
        self, plan: BatchPlan | None, context: StepContext
    ) -> ForwardOutput | Sequence[int] | Sequence[Sequence[int]] | None: ...


class DistributedWorker:
    """One model runtime per rank; only the replica source is called by an Engine.

    CP/EP/attention-DP runtimes implement forward_distributed(plan_or_none, context),
    including empty participation. Optional validate_distributed performs host
    checks before a domain-wide readiness vote. The forward callback must finish
    device work before returning. Bind/release may communicate within a replica;
    collectives across replicas belong only in the forward callback.
    """

    def __init__(self, runner: Runner | DistributedRunner, groups: ParallelGroups):
        self.runner, self.groups = runner, groups
        self.closed = False
        self.failure = None
        self._entered = False
        self._step_id, self._phase = 0, None
        self._forward, self._validate = None, None
        error = None
        try:
            self._forward = getattr(runner, "forward_distributed", None)
            self._validate = getattr(runner, "validate_distributed", None)
            self.runner_config = getattr(runner, "runner_config", RunnerConfig())
            capabilities = getattr(runner, "parallelism_capabilities", None)
            if capabilities is not None:
                capabilities.validate(groups.topology, model=type(runner).__name__)
            if (
                groups.topology.cp > 1
                or groups.topology.ep > 1
                or groups.topology.attn_dp > 1
            ) and not callable(self._forward):
                raise ValueError(
                    "CP, EP, and attention DP require forward_distributed(plan_or_none, context)"
                )
            signature = _runner_signature(self.runner_config)
        except BaseException as exc:
            signature, error = None, _message(exc)
        configs = self._gather(groups.replica, (signature, error))
        error = next((item[1] for item in configs if item[1]), error)
        if any(item[0] != configs[0][0] for item in configs):
            error = "runner configurations must agree within each attention replica"
        self._consensus(groups.domain, error)

    def _gather(self, group: RankGroup, value):
        if group.size == 1:
            return [value]
        results = [None] * group.size
        try:
            self.groups.distributed.all_gather_object(
                results, value, group=group.handle
            )
        except BaseException as exc:
            self.failure = _message(exc)
            raise DistributedExecutionError(self.failure) from exc
        return results

    def _consensus(self, group: RankGroup, error: str | None):
        failure = _first_error(group, self._gather(group, error))
        if failure:
            raise DistributedExecutionError(failure)

    def _drivers(self, operation, value):
        group = self.groups.drivers
        if group is None:
            raise RuntimeError("only replica drivers coordinate serving")
        results = self._gather(group, (operation, value))
        if any(kind != operation for kind, _ in results):
            raise DistributedExecutionError(
                "drivers called collective engine operations in different orders"
            )
        return [value for _, value in results]

    def _send(self, command):
        if self.closed:
            raise DistributedExecutionError("workers are closed")
        if self.groups.replica.size > 1:
            try:
                self.groups.distributed.broadcast_object_list(
                    [command],
                    src=self.groups.replica.source,
                    group=self.groups.replica.handle,
                )
            except BaseException as exc:
                self.failure = _message(exc)
                raise DistributedExecutionError(self.failure) from exc

    def _local_command(self, command):
        operation = command["operation"]
        error = None
        try:
            if operation == "bind":
                self.runner.bind_resources(
                    SimpleNamespace(
                        cache_pool_id=command["cache_pool_id"],
                        cache_layout=self.runner_config.cache_layout,
                    )
                )
            elif operation == "release":
                release = getattr(self.runner, "release", None)
                if release is not None:
                    release(command["ids"])
            else:
                raise ValueError(f"unknown worker operation: {operation}")
        except BaseException as exc:
            error = _message(exc)
        self._consensus(self.groups.replica, error)

    def bind_resources(self, binding):
        command = {"operation": "bind", "cache_pool_id": binding.cache_pool_id}
        self._send(command)
        self._local_command(command)

    def release(self, sequence_ids):
        if self.failure:
            raise DistributedExecutionError(self.failure)
        command = {"operation": "release", "ids": list(sequence_ids)}
        self._send(command)
        self._local_command(command)

    def forward(self, plan: BatchPlan):
        try:
            return self._coordinate(plan)
        except DistributedExecutionError:
            raise
        except BaseException as exc:
            # A transport exception can occur after kernels have changed state.
            # All exits from this callback must carry the acceptance marker.
            self.failure = _message(exc)
            raise DistributedExecutionError(self.failure) from exc

    def _coordinate(self, plan: BatchPlan | None, error: str | None = None):
        self._entered = True
        payload = None
        try:
            if plan is not None:
                if plan.phase != self._phase:
                    raise ValueError("local plan disagrees with the coordinated phase")
                payload = pickle.dumps(
                    batch_plan_payload(plan), protocol=pickle.HIGHEST_PROTOCOL
                )
        except BaseException as exc:
            error = _message(exc)
        descriptors = self._drivers(
            "prepare",
            (
                0 if plan is None else plan.num_sequences,
                0 if plan is None else len(plan.token_ids),
                error,
            ),
        )
        failure = _first_error(self.groups.drivers, [item[2] for item in descriptors])
        if failure:
            raise DistributedExecutionError(failure)
        sizes = tuple((rows, tokens) for rows, tokens, _ in descriptors)
        if not any(rows for rows, _ in sizes):
            return None
        command = {
            "operation": "forward",
            "payload": payload,
            "step_id": self._step_id,
            "phase": "prefill" if self._phase == BatchPhase.PREFILL else "decode",
            "sizes": sizes,
        }
        self._send(command)
        return self._execute(command, plan)

    def _execute(self, command, plan=None):
        error, context = None, None
        try:
            if plan is None and command["payload"] is not None:
                plan = batch_plan_from_payload(pickle.loads(command["payload"]))
            context = StepContext(
                self.groups,
                command["step_id"],
                BatchPhase.PREFILL
                if command["phase"] == "prefill"
                else BatchPhase.DECODE,
                command["sizes"],
                0 if plan is None else plan.num_sequences,
                0 if plan is None else len(plan.token_ids),
            )
            if self._validate is not None:
                self._validate(plan, context)
        except BaseException as exc:
            error = _message(exc)
        self._consensus(self.groups.domain, error)
        result, error = None, None
        try:
            if self._forward is not None:
                result = self._forward(plan, context)
            elif plan is not None:
                result = self.runner.forward(plan)
        except BaseException as exc:
            error = _message(exc)
        self._consensus(self.groups.domain, error)
        return result

    def serve(self):
        """Follower loop; failures remain contained until the driver closes it."""
        if self.groups.is_driver:
            raise ValueError("replica drivers construct DistributedEngine instead")
        while True:
            commands = [None]
            self.groups.distributed.broadcast_object_list(
                commands,
                src=self.groups.replica.source,
                group=self.groups.replica.handle,
            )
            command = commands[0]
            operation = command["operation"]
            if operation == "close":
                self.closed = True
                if command["error"]:
                    raise DistributedExecutionError(command["error"])
                return
            try:
                if operation == "forward":
                    self._execute(command)
                elif operation == "finish":
                    self.failure = self.failure or command["error"]
                else:
                    self._local_command(command)
            except DistributedExecutionError as exc:
                self.failure = self.failure or str(exc)


class DistributedEngine:
    """One native request engine per attention-DP replica, plus coordinated steps.

    Drivers in the same topology call has_work(), step(), and close() in the
    same order. submit/cancel are local to an attention replica. There is no
    distributed request router or pending step API.
    """

    def __init__(self, worker: DistributedWorker, **engine_options):
        if not worker.groups.is_driver:
            raise ValueError("only replica drivers own a request engine")
        self.worker, self.closed = worker, False
        self._step_id, self._last_phase = 0, None
        self.local = None
        error = None
        try:
            self.local = Engine(worker, **engine_options)
        except BaseException as exc:
            error = _message(exc)
        errors = worker._drivers("initialize", error)
        failure = _first_error(worker.groups.drivers, errors)
        if failure:
            self._record_failure(failure)
            worker._send({"operation": "close", "error": failure})
            worker.closed = self.closed = True
            raise DistributedExecutionError(failure)

    @property
    def poisoned(self):
        return self.worker.failure is not None or self.local.poisoned

    def _check(self):
        if self.closed:
            raise RuntimeError("distributed engine is closed")
        if self.poisoned:
            raise DistributedExecutionError(
                self.worker.failure or self.local.poison_error
            )

    def submit(self, *args, **kwargs):
        self._check()
        return self.local.submit(*args, **kwargs)

    def cancel(self, request):
        self._check()
        try:
            return self.local.cancel(request)
        except BaseException as exc:
            if self.local.poisoned:
                self.worker.failure = _message(exc)
            raise

    def _finish(self, failure):
        self._record_failure(failure)
        self.worker._send({"operation": "finish", "error": failure})
        if failure:
            raise DistributedExecutionError(failure)

    def _record_failure(self, failure):
        if failure:
            self.worker.failure = self.worker.failure or failure
            if self.local is not None:
                self.local.poison(failure)

    def has_work(self):
        if self.closed:
            return False
        states = self.worker._drivers(
            "work",
            (self.local.has_work(), self.worker.failure or self.local.poison_error),
        )
        failure = _first_error(
            self.worker.groups.drivers, [error for _, error in states]
        )
        if failure:
            self._finish(failure)
        return any(work for work, _ in states)

    def step(self):
        if self.closed:
            raise RuntimeError("distributed engine is closed")
        proposal, error = None, self.worker.failure
        try:
            phase = self.local.next_phase()
            proposal = (
                None
                if phase is None
                else ("prefill" if phase == BatchPhase.PREFILL else "decode")
            )
        except BaseException as exc:
            error = _message(exc)
        proposals = self.worker._drivers("phase", (proposal, error))
        failure = _first_error(
            self.worker.groups.drivers, [error for _, error in proposals]
        )
        if failure:
            self._finish(failure)
        available = {phase for phase, _ in proposals} - {None}
        if not available:
            return []
        # A mixed domain alternates phases, bounding starvation of either cohort.
        phase = (
            "decode"
            if "decode" in available
            and (self._last_phase == "prefill" or "prefill" not in available)
            else "prefill"
        )
        self._last_phase = phase
        self._step_id += 1
        self.worker._step_id = self._step_id
        self.worker._phase = (
            BatchPhase.PREFILL if phase == "prefill" else BatchPhase.DECODE
        )
        self.worker._entered = False
        events, error = [], None
        try:
            events = self.local.step(phase=self.worker._phase)
        except BaseException as exc:
            error = _message(exc)
        if not self.worker._entered:
            # Empty, mismatched, or rejected preparation still joins the vote.
            try:
                self.worker._coordinate(None, error)
            except BaseException as exc:
                error = error or _message(exc)
        errors = self.worker._drivers("commit", error)
        self._finish(_first_error(self.worker.groups.drivers, errors))
        return events

    def close(self):
        if self.closed:
            return
        self._record_failure(self.worker.failure or self.local.poison_error)
        error = self.worker.failure
        try:
            if not self.local.poisoned:
                self.local.close()
        except BaseException as exc:
            error = error or _message(exc)
        errors = self.worker._drivers("close", error)
        failure = _first_error(self.worker.groups.drivers, errors)
        self._record_failure(failure)
        self.worker._send({"operation": "close", "error": failure})
        self.worker.closed = self.closed = True
        if failure:
            raise DistributedExecutionError(failure)


__all__ = [
    "DistributedEngine",
    "DistributedExecutionError",
    "DistributedRunner",
    "DistributedWorker",
    "StepContext",
]
