"""Small synchronous rank votes without object serialization on successful steps."""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.distributed as dist

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

    Each rank owns one record in a shared reduction buffer. All other slots
    contain INT64_MIN, so MAX selects the owning rank's exact signed values.
    Every operation uses the same shape and collective, including the operation
    tag check; ranks calling different step operations therefore fail together.
    Error text takes a separate object gather only when a rank reports failure.
    """

    _fields = 6
    _width = _fields + 1  # Error flag followed by integer payload.

    def __init__(self, distributed):
        self.distributed = distributed
        self._buffers: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    def gather(
        self,
        group: RankGroup,
        operation: ControlOperation,
        values=(),
        *,
        error: str | None = None,
    ) -> tuple[list[tuple[int, ...]], list[str | None]]:
        values = tuple(map(int, values))
        if len(values) > self._fields:
            raise ValueError(f"control records accept at most {self._fields} integers")
        tag = int(operation)
        if group.size == 1:
            return [values], [error]

        key = id(group.handle), group.ranks
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.empty(
                2 + group.size * self._width, dtype=torch.int64, device="cpu"
            )
            self._buffers[key] = buffer
        buffer.fill_(torch.iinfo(torch.int64).min)
        buffer[0], buffer[1] = tag, -tag
        start = 2 + group.local_rank * self._width
        buffer[start : start + self._width].zero_()
        buffer[start] = int(error is not None)
        for column, value in enumerate(values, start + 1):
            buffer[column] = value
        self.distributed.all_reduce(buffer, op=dist.ReduceOp.MAX, group=group.handle)
        maximum, negative_minimum = buffer[:2].tolist()
        if maximum != tag or negative_minimum != -tag:
            raise ControlPlaneError(
                "ranks called collective engine operations in different orders "
                f"(operation tags {-negative_minimum} through {maximum})"
            )
        records = buffer[2:].view(group.size, self._width).tolist()
        errors = [None] * group.size
        if any(record[0] for record in records):
            self.distributed.all_gather_object(errors, error, group=group.handle)
        return [tuple(record[1 : 1 + len(values)]) for record in records], errors
