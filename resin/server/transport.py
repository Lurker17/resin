"""Opaque native HTTP packets over driver-group Gloo byte collectives."""

from __future__ import annotations

import numpy as np
import torch

from resin.backend.distributed import DistributedExecutionError, DistributedWorker


class HttpTransport:
    def __init__(self, worker: DistributedWorker):
        self.worker = worker
        group = worker.groups.drivers
        if group is None:
            raise ValueError("HTTP transport requires a replica driver")
        self.group = group
        self.distributed = worker.groups.distributed

    def vote(self, stage, values=(), *, error=None):
        records, errors = self.worker.driver_vote(f"http {stage}", values, error=error)
        failure = next(
            (
                f"HTTP {stage} failed on rank {rank}: {error}"
                for rank, error in zip(self.group.ranks, errors, strict=True)
                if error is not None
            ),
            None,
        )
        if failure:
            raise DistributedExecutionError(failure)
        return records

    def _buffers(self, stage, size, payload, *, gather=False):
        send, receive, error = None, None, None
        try:
            # Gloo writes outside inference_mode, including during decode.
            with torch.inference_mode(False):
                send = torch.zeros(size, dtype=torch.uint8, device="cpu")
                if payload:
                    send.numpy()[: len(payload)] = np.frombuffer(payload, np.uint8)
                if gather and self.group.local_rank == 0:
                    receive = [torch.empty_like(send) for _ in self.group.ranks]
        except BaseException as exc:
            error = str(exc)
        # Allocation/copy failure must be visible before any peer enters payload IO.
        self.vote(f"{stage} buffers", error=error)
        assert send is not None
        return send, receive

    def broadcast(self, payload: bytes) -> bytes:
        lengths = self.vote("ingress length", (len(payload),))
        size = lengths[0][0]
        if self.group.size == 1 or size == 0:
            return payload
        send, _ = self._buffers("ingress", size, payload)
        self.distributed.broadcast(send, src=self.group.source, group=self.group.handle)
        return payload if self.group.local_rank == 0 else send.numpy().tobytes()

    def gather(self, payload: bytes, *, error=None) -> list[bytes]:
        lengths = self.vote("events", (len(payload),), error=error)
        size = max(row[0] for row in lengths)
        if self.group.size == 1 or size == 0:
            return []
        send, receive = self._buffers("events", size, payload, gather=True)
        self.distributed.gather(
            send, gather_list=receive, dst=self.group.source, group=self.group.handle
        )
        if receive is None:
            return []
        # Rank zero keeps its native packet; only remote bytes require decoding.
        return [
            tensor.numpy()[: length[0]].tobytes()
            for tensor, length in zip(receive[1:], lengths[1:], strict=True)
        ]
