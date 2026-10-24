"""Small, framework-independent worker groups with optional NUMA placement."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import resin


class ProcessError(RuntimeError):
    """A worker raised an exception or exited unsuccessfully."""

    def __init__(
        self, rank: int, pid: int | None, exitcode: int, worker_traceback: str = ""
    ) -> None:
        self.rank = rank
        self.pid = pid
        self.exitcode = exitcode
        self.worker_traceback = worker_traceback
        message = f"worker rank {rank} (pid {pid}) exited with code {exitcode}"
        if worker_traceback:
            message += f"\n{worker_traceback}"
        super().__init__(message)


def _run_worker(rank: int, fn: Callable[..., Any], args: tuple, error_path: str):
    try:
        fn(rank, *args)
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        # A file cannot fill a pipe while the parent is busy starting or waiting
        # for other workers. Read it only after this process has exited.
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        if isinstance(error, SystemExit):
            raise
        raise SystemExit(1) from None


class ProcessContext:
    """Own spawned workers and stop their peers if any worker fails.

    ``join`` waits for the whole group, with one timeout for all workers.
    ``close`` terminates and reaps remaining workers. Exiting a context manager
    waits on success and closes the group if its body raises.
    ``placements`` records each rank's effective binding, or ``None`` when NUMA
    placement is disabled.
    """

    def __init__(self, directory: TemporaryDirectory[str]) -> None:
        self.processes: list[BaseProcess] = []
        self.placements: list[resin.numa.Binding | None] = []
        self._directory = directory
        self._remaining: set[int] = set()
        self._closed = False
        self._error: ProcessError | None = None

    def _error_path(self, rank: int) -> Path:
        return Path(self._directory.name) / f"{rank}.txt"

    def _failure(self, rank: int) -> ProcessError:
        process = self.processes[rank]
        try:
            detail = self._error_path(rank).read_text(encoding="utf-8")
        except OSError:
            detail = ""
        assert process.exitcode is not None
        return ProcessError(rank, process.pid, process.exitcode, detail)

    def join(self, timeout: float | None = None) -> bool:
        """Return true when all workers exit successfully; raise on failure.

        A timeout returns false without stopping workers. A worker failure
        terminates and reaps its peers before raising :class:`ProcessError`.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        if self._error is not None:
            raise self._error
        if self._closed:
            for rank, process in enumerate(self.processes):
                if process.exitcode != 0:
                    self._error = self._failure(rank)
                    raise self._error
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._remaining:
            sentinels = {
                self.processes[rank].sentinel: rank for rank in self._remaining
            }
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            ready = wait(sentinels, timeout=remaining)
            if not ready:
                return False
            for sentinel in ready:
                assert isinstance(sentinel, int)
                rank = sentinels[sentinel]
                process = self.processes[rank]
                process.join()
                self._remaining.remove(rank)
                if process.exitcode != 0:
                    self._error = self._failure(rank)
                    self.close()
                    raise self._error
        self.close()
        return True

    def close(self) -> None:
        """Terminate and reap live workers; release temporary error reports."""
        if self._closed:
            return
        started = [process for process in self.processes if process.pid is not None]
        for process in started:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 1.0
        for process in started:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in started:
            if process.is_alive():
                process.kill()
        for process in started:
            process.join()
        self._remaining.clear()
        self._directory.cleanup()
        self._closed = True

    def __enter__(self) -> ProcessContext:
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        try:
            if exc_type is None:
                self.join()
        finally:
            self.close()


def spawn(
    fn: Callable[..., Any],
    args: tuple = (),
    *,
    devices: Sequence[str],
    numa: Literal["auto", "off", "required"] = "auto",
    join: bool = True,
) -> ProcessContext:
    """Spawn ``fn(rank, *args)`` once per device, without importing Torch.

    Device entries control host placement; workers select their own accelerator.
    ``auto`` uses available NUMA placement, ``required`` fails if placement cannot
    be applied, and ``off`` retains the caller's placement. Functions and their
    arguments must be picklable, as with Python's ``spawn`` start method.
    """
    if numa not in ("auto", "off", "required"):
        raise ValueError("numa must be 'auto', 'off', or 'required'")
    if isinstance(devices, (str, bytes)):
        raise TypeError("devices must be a sequence of device strings")
    devices = tuple(devices)
    if not devices or any(not isinstance(device, str) for device in devices):
        raise ValueError("devices must contain at least one device string")
    args = tuple(args)
    context = ProcessContext(TemporaryDirectory(prefix="resin-process-"))
    multiprocessing = mp.get_context("spawn")
    try:
        for rank, device in enumerate(devices):
            process = multiprocessing.Process(
                target=_run_worker,
                args=(rank, fn, args, str(context._error_path(rank))),
                name=f"resin-worker-{rank}",
            )
            context.processes.append(process)
            if numa == "off":
                placement = nullcontext()
            else:
                placement = resin.numa.bind(device=device, strict=numa == "required")
            with placement as binding:
                context.placements.append(binding)
                process.start()
            context._remaining.add(rank)
        if join:
            context.join()
        return context
    except BaseException:
        context.close()
        raise


__all__ = ["ProcessContext", "ProcessError", "spawn"]
