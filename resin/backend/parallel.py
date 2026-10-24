"""Host-side supervision for spawned model processes."""

from __future__ import annotations

import os
import threading
import traceback
from typing import Any


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


__all__ = ["monitor_process_context"]
