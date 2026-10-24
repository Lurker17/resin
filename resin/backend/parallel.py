"""Framework-neutral runner coordination for tensor-parallel processes."""

from __future__ import annotations

import os
import threading
import traceback
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from resin import BatchPlan
from resin.backend.command_transport import CommandTransport
from resin.backend.plan_transport import PlanTransport


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


class TensorParallelRunner:
    """Drive identical runner calls on follower ranks from rank zero."""

    def __init__(self, runner: Any, control_group: Any, distributed: Any) -> None:
        self._runner = runner
        self._followers_closed = False
        self._plans = PlanTransport()
        self._wire = CommandTransport(distributed, control_group)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)

    def forward(self, plan: BatchPlan):
        self._broadcast({"operation": "forward", "payload": self._plans.prepare(plan)})
        return self._runner.forward(plan)

    def bind_resources(self, binding: Any):
        self._broadcast(
            {
                "operation": "bind",
                "cache_pool_id": binding.cache_pool_id,
            }
        )
        return self._runner.bind_resources(binding)

    def release(self, sequence_ids: Sequence[int]) -> None:
        ids = list(sequence_ids)
        self._broadcast({"operation": "release", "ids": ids})
        self._runner.release(ids)
        self._plans.release(ids)

    def close_followers(self) -> None:
        if not self._followers_closed:
            self._broadcast({"operation": "close"})
            self._followers_closed = True
            self._plans.clear()

    def _broadcast(self, command: dict[str, Any]) -> None:
        self._wire.send(command)


def follow_tensor_parallel_runner(
    runner: Any, control_group: Any, distributed: Any
) -> None:
    """Execute rank-zero runner commands until the leader shuts down."""

    plans = PlanTransport()
    wire = CommandTransport(distributed, control_group)
    while True:
        command = wire.receive()
        operation = command["operation"]
        if operation == "close":
            plans.clear()
            return
        if operation == "release":
            runner.release(command["ids"])
            plans.release(command["ids"])
        elif operation == "bind":
            runner.bind_resources(
                SimpleNamespace(
                    cache_pool_id=command["cache_pool_id"],
                    cache_layout=runner.runner_config.cache_layout,
                )
            )
        elif operation == "forward":
            runner.forward(plans.restore(command["payload"]))
        else:
            raise RuntimeError(f"unknown tensor-parallel operation: {operation!r}")


__all__ = [
    "TensorParallelRunner",
    "follow_tensor_parallel_runner",
    "monitor_process_context",
]
