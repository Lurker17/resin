"""Host-side supervision for spawned model processes."""

from __future__ import annotations

import os
import threading
import time
import traceback
from multiprocessing.connection import wait
from typing import Any


def monitor_process_context(process_context: Any, label: str) -> None:
    """Report a spawned worker failure and terminate its hosting process."""

    def monitor() -> None:
        try:
            # A serving worker returning normally is still a group failure.
            # Resin's join waits for every worker; observe the first exit before
            # asking either Resin's or Torch's context to surface its error.
            wait([process.sentinel for process in process_context.processes])
            process_context.join(timeout=0)
        except BaseException:
            print(f"[{label}] worker process failed:", flush=True)
            traceback.print_exc()
        else:
            print(f"[{label}] worker process exited unexpectedly", flush=True)
        try:
            close = getattr(process_context, "close", None)
            if close is not None:
                close()
            else:
                # Torch contexts do not expose close(). Reap their peers before
                # exiting the host rather than relying on parent-death signals.
                processes = process_context.processes
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                deadline = time.monotonic() + 1.0
                for process in processes:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                for process in processes:
                    if process.is_alive():
                        process.kill()
                for process in processes:
                    process.join()
        finally:
            os._exit(1)

    threading.Thread(
        target=monitor,
        name=f"{label}-process-monitor",
        daemon=True,
    ).start()


__all__ = ["monitor_process_context"]
