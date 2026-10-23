"""Small synchronous rank votes without object serialization on successful steps."""

from __future__ import annotations

from enum import IntEnum
from hashlib import blake2b

import torch

from resin.backend.topology import RankGroup


class ControlOperation(IntEnum):
    WORK = 1
    PHASE = 2
    PREPARE = 3
    VALIDATE = 4
    FORWARD = 5
    COMMIT = 6


class ControlPlaneError(RuntimeError):
    pass


class ControlPlane:
    """Gather fixed-size integer records with one CPU collective per vote.

    Every operation uses the same shape and collective, including the operation
    tag check; ranks calling different step operations therefore fail together.
    Error text takes a separate object gather only when a rank reports failure.
    """

    _fields = 6
    _width = _fields + 2  # Operation tag, error flag, then integer payload.

    def __init__(self, distributed):
        self.distributed = distributed
        self._buffers: dict[
            tuple[int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._tags: dict[str, int] = {}

    def gather(
        self,
        group: RankGroup,
        operation: ControlOperation | str,
        values=(),
        *,
        error: str | None = None,
    ) -> tuple[list[tuple[int, ...]], list[str | None]]:
        values = tuple(map(int, values))
        if len(values) > self._fields:
            raise ValueError(f"control records accept at most {self._fields} integers")
        if isinstance(operation, str):
            if operation not in self._tags:
                self._tags[operation] = (
                    int.from_bytes(blake2b(operation.encode(), digest_size=8).digest())
                    & ((1 << 62) - 1)
                ) | (1 << 62)
            tag = self._tags[operation]
        else:
            tag = int(operation)
        if group.size == 1:
            return [values], [error]

        key = id(group.handle), group.ranks
        buffers = self._buffers.get(key)
        if buffers is None:
            # Votes can first run inside a model's inference scope, then be
            # reused outside it. Gloo also writes receive on its worker thread.
            # Both buffers must therefore remain ordinary mutable tensors.
            with torch.inference_mode(False):
                send = torch.empty(self._width, dtype=torch.int64, device="cpu")
                receive = torch.empty(
                    group.size * self._width, dtype=torch.int64, device="cpu"
                )
            self._buffers[key] = send, receive
        else:
            send, receive = buffers
        send.zero_()
        send[0], send[1] = tag, int(error is not None)
        for column, value in enumerate(values, 2):
            send[column] = value
        self.distributed.all_gather_single(receive, send, group=group.handle)
        records = receive.view(group.size, self._width).tolist()
        if any(record[0] != tag for record in records):
            raise ControlPlaneError(
                "ranks called collective engine operations in different orders"
            )
        errors = [None] * group.size
        if any(record[1] for record in records):
            self.distributed.all_gather_object(errors, error, group=group.handle)
        return [tuple(record[2 : 2 + len(values)]) for record in records], errors
