"""Framework-neutral runner coordination for tensor-parallel processes."""

from __future__ import annotations

import os
import threading
import traceback
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from resin import BatchPhase, BatchPlan, CacheGroupPlan, SamplingParams


def monitor_process_context(process_context: Any, label: str) -> None:
    """Report a spawned worker failure and terminate its hosting process."""

    def monitor() -> None:
        try:
            process_context.join()
        except BaseException:
            print(f"[{label}] worker process failed:", flush=True)
            traceback.print_exc()
        else:
            print(f"[{label}] worker process exited unexpectedly", flush=True)
        os._exit(1)

    threading.Thread(
        target=monitor,
        name=f"{label}-process-monitor",
        daemon=True,
    ).start()


def batch_plan_payload(plan: BatchPlan) -> dict[str, Any]:
    """Serialize the Python-visible parts of a plan for follower ranks."""

    phase = "prefill" if plan.phase == BatchPhase.PREFILL else "decode"
    return {
        "phase": phase,
        "batch_id": plan.batch_id,
        "batch_key": plan.batch_key,
        "token_ids": plan.token_ids,
        "sequence_ids": plan.sequence_ids,
        "token_indptr": plan.token_indptr,
        "context_lengths": plan.context_lengths,
        "sampling_params": [
            {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "seed": params.seed,
                "extra": params.extra,
            }
            for params in plan.sampling_params
        ],
        "sampling_mask": plan.sampling_mask,
        "media": plan.media,
        "cache_groups": [
            {
                "group_index": group.group_index,
                "blocks_per_slab": group.blocks_per_slab,
                "block_indptr": group.block_indptr,
                "block_slab_ids": group.block_slab_ids,
                "block_slotnums": group.block_slotnums,
                "block_table_start_indices": group.block_table_start_indices,
                "append_slab_ids": group.append_slab_ids,
                "append_slotnums": group.append_slotnums,
                "append_block_offsets": group.append_block_offsets,
            }
            for group in plan.cache_groups
        ],
    }


def batch_plan_from_payload(payload: dict[str, Any]) -> BatchPlan:
    """Rebuild a frozen plan on a tensor-parallel follower rank."""

    phase = BatchPhase.PREFILL if payload["phase"] == "prefill" else BatchPhase.DECODE
    sampling_params = [
        SamplingParams(**values) for values in payload["sampling_params"]
    ]
    cache_groups = [CacheGroupPlan(**values) for values in payload["cache_groups"]]
    return BatchPlan(
        payload["sequence_ids"],
        payload["context_lengths"],
        payload["token_ids"],
        payload["token_indptr"],
        phase=phase,
        batch_id=payload["batch_id"],
        batch_key=payload["batch_key"],
        cache_groups=cache_groups,
        sampling_mask=payload["sampling_mask"],
        sampling_params=sampling_params,
        media=payload["media"],
    )


class TensorParallelRunner:
    """Drive identical runner calls on follower ranks from rank zero."""

    def __init__(self, runner: Any, control_group: Any, distributed: Any) -> None:
        self._runner = runner
        self._control_group = control_group
        self._distributed = distributed
        self._followers_closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)

    def forward(self, plan: BatchPlan):
        self._broadcast({"operation": "forward", "plan": batch_plan_payload(plan)})
        return self._runner.forward(plan)

    def bind_resources(self, binding: Any):
        self._broadcast(
            {
                "operation": "bind_resources",
                "cache_pool_id": binding.cache_pool_id,
            }
        )
        return self._runner.bind_resources(binding)

    def release(self, sequence_ids: Sequence[int]) -> None:
        ids = list(sequence_ids)
        self._broadcast({"operation": "release", "sequence_ids": ids})
        self._runner.release(ids)

    def close_followers(self) -> None:
        if not self._followers_closed:
            self._broadcast({"operation": "close"})
            self._followers_closed = True

    def _broadcast(self, command: dict[str, Any]) -> None:
        self._distributed.broadcast_object_list(
            [command], src=0, group=self._control_group
        )


def follow_tensor_parallel_runner(
    runner: Any, control_group: Any, distributed: Any
) -> None:
    """Execute rank-zero runner commands until the leader shuts down."""

    while True:
        commands: list[Any] = [None]
        distributed.broadcast_object_list(commands, src=0, group=control_group)
        command = commands[0]
        operation = command["operation"]
        if operation == "close":
            return
        if operation == "release":
            runner.release(command["sequence_ids"])
        elif operation == "bind_resources":
            runner.bind_resources(
                SimpleNamespace(
                    cache_pool_id=command["cache_pool_id"],
                    cache_layout=runner.runner_config.cache_layout,
                )
            )
        elif operation == "forward":
            runner.forward(batch_plan_from_payload(command["plan"]))
        else:
            raise RuntimeError(f"unknown tensor-parallel operation: {operation!r}")


__all__ = [
    "TensorParallelRunner",
    "batch_plan_from_payload",
    "batch_plan_payload",
    "follow_tensor_parallel_runner",
    "monitor_process_context",
]
