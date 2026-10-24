"""Packed step metadata with request-scoped sampling and native media ownership."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from resin import BatchPhase, BatchPlan, CacheGroupPlan, SamplingParams
from resin.protocols import MediaInput

_MAGIC = 0x524553494E504C4E  # RESINPLN
_VERSION = 1
_PREFIX_WORDS = 7
_PLAN_FIELDS = (
    ("sequence_ids", np.uint64),
    ("context_lengths", np.uint32),
    ("prefill_lengths", np.uint32),
    ("token_ids", np.uint32),
    ("token_indptr", np.uint32),
    ("sampling_mask", np.bool_),
    ("prompt_score_indices", np.uint32),
    ("prompt_score_token_ids", np.uint32),
    ("prompt_score_indptr", np.uint32),
    ("prompt_score_top_counts", np.uint32),
    ("prompt_target_indptr", np.uint32),
    ("prompt_target_token_ids", np.uint32),
)
_GROUP_FIELDS = (
    "block_indptr",
    "block_slab_ids",
    "block_slotnums",
    "block_table_start_indices",
    "append_slab_ids",
    "append_slotnums",
    "append_block_offsets",
)
_GROUP_WORDS = 2 + len(_GROUP_FIELDS)
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "logprobs",
    "top_logprobs",
    "prompt_logprobs",
    "prompt_top_logprobs",
    "extra",
)


@dataclass(frozen=True, slots=True)
class RequestRegistration:
    """One-time JSON policy metadata and separate native media arrays."""

    metadata: bytes
    buffers: tuple[npt.NDArray[Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PlanPacket:
    """Native-endian header and aligned vector bytes, owned by this step."""

    header: npt.NDArray[np.uint64]
    data: npt.NDArray[np.uint8]
    registrations: tuple[RequestRegistration, ...] = ()


@dataclass(frozen=True, slots=True)
class _Request:
    sampling_params: SamplingParams
    media: MediaInput | None


def _is_json_extra(value: Any) -> bool:
    if type(value) is dict:
        return all(
            type(key) is str and _is_json_extra(item) for key, item in value.items()
        )
    if type(value) is list:
        return all(_is_json_extra(item) for item in value)
    return value is None or type(value) in (bool, int, float, str)


def _register(sequence_id: int, request: _Request) -> RequestRegistration:
    sampling = {
        name: getattr(request.sampling_params, name) for name in _SAMPLING_FIELDS
    }
    if not _is_json_extra(sampling["extra"]):
        raise TypeError(
            "distributed SamplingParams.extra requires JSON-native values: "
            "string-keyed dicts, lists, and None/bool/int/float/str scalars"
        )
    media_metadata = None
    buffers = ()
    if request.media is not None:
        from resin._resin import _media_to_buffers

        encoded, buffers = _media_to_buffers(request.media)
        media_metadata = encoded.decode("utf-8")
    metadata = json.dumps(
        {
            "version": _VERSION,
            "sequence_id": sequence_id,
            "sampling": sampling,
            "media": media_metadata,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return RequestRegistration(metadata, buffers)


def _restore_registration(registration: RequestRegistration) -> tuple[int, _Request]:
    metadata = json.loads(registration.metadata)
    if not isinstance(metadata, dict) or metadata.get("version") != _VERSION:
        raise ValueError("unsupported request registration version")
    sequence_id = metadata["sequence_id"]
    if type(sequence_id) is not int or not 0 <= sequence_id < 2**64:
        raise ValueError("request sequence_id must be a uint64")
    sampling = SamplingParams(**metadata["sampling"])
    media_metadata = metadata["media"]
    if media_metadata is None:
        if registration.buffers:
            raise ValueError("request without media cannot contain media buffers")
        media = None
    else:
        from resin._resin import _media_from_buffers

        media = _media_from_buffers(
            media_metadata.encode("utf-8"), registration.buffers
        )
    return sequence_id, _Request(sampling, media)


def _aligned(offset: int) -> int:
    return (offset + 7) & ~7


def _pack(plan: BatchPlan) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
    groups = plan.cache_groups
    header = np.empty(
        _PREFIX_WORDS + len(_PLAN_FIELDS) + len(groups) * _GROUP_WORDS,
        dtype=np.uint64,
    )
    header[:_PREFIX_WORDS] = (
        _MAGIC,
        _VERSION,
        int(plan.phase == BatchPhase.DECODE),
        plan.batch_id,
        int(plan.batch_key is not None),
        plan.batch_key or 0,
        len(groups),
    )
    arrays = [getattr(plan, name) for name, _ in _PLAN_FIELDS]
    cursor = _PREFIX_WORDS
    for array in arrays:
        header[cursor] = array.size
        cursor += 1
    for group in groups:
        header[cursor : cursor + 2] = (group.group_index, group.blocks_per_slab)
        cursor += 2
        for name in _GROUP_FIELDS:
            array = getattr(group, name)
            arrays.append(array)
            header[cursor] = array.size
            cursor += 1
    offsets = []
    end = 0
    for array in arrays:
        start = _aligned(end)
        offsets.append(start)
        end = start + array.nbytes
    # Zero the alignment padding as well: no uninitialized bytes cross ranks.
    data = np.zeros(end, dtype=np.uint8)
    for array, start in zip(arrays, offsets):
        data[start : start + array.nbytes] = array.view(np.uint8)
    return header, data


def _unpack(packet: PlanPacket) -> tuple[dict[str, Any], list[CacheGroupPlan]]:
    header, data = packet.header, packet.data
    if (
        header.dtype != np.uint64
        or header.ndim != 1
        or not header.flags.c_contiguous
        or len(header) < _PREFIX_WORDS
        or data.dtype != np.uint8
        or data.ndim != 1
        or not data.flags.c_contiguous
    ):
        raise ValueError("plan packet requires contiguous uint64 header and uint8 data")
    if int(header[0]) != _MAGIC or int(header[1]) != _VERSION:
        raise ValueError("unsupported plan packet format")
    if int(header[2]) not in (0, 1) or int(header[4]) not in (0, 1):
        raise ValueError("invalid plan phase or batch_key flag")
    group_count = int(header[6])
    if len(header) != _PREFIX_WORDS + len(_PLAN_FIELDS) + group_count * _GROUP_WORDS:
        raise ValueError("plan packet header length does not match its fields")
    cursor = _PREFIX_WORDS
    offset = 0

    def vector(dtype: Any) -> npt.NDArray[Any]:
        nonlocal cursor, offset
        count = int(header[cursor])
        cursor += 1
        start = _aligned(offset)
        offset = start + count * np.dtype(dtype).itemsize
        if offset > data.nbytes:
            raise ValueError("plan packet vector exceeds its data buffer")
        return np.frombuffer(data, dtype=dtype, count=count, offset=start)

    values: dict[str, Any] = {name: vector(dtype) for name, dtype in _PLAN_FIELDS}
    values.update(
        phase=BatchPhase.DECODE if header[2] else BatchPhase.PREFILL,
        batch_id=int(header[3]),
        batch_key=int(header[5]) if header[4] else None,
    )
    groups = []
    for _ in range(group_count):
        group_index, blocks_per_slab = int(header[cursor]), int(header[cursor + 1])
        cursor += 2
        groups.append(
            CacheGroupPlan(
                group_index=group_index,
                blocks_per_slab=blocks_per_slab,
                **{name: vector(np.uint32) for name in _GROUP_FIELDS},
            )
        )
    if offset != data.nbytes:
        raise ValueError("plan packet has trailing data")
    return values, groups


class PlanTransport:
    """Keep request state once while transporting only typed arrays each step.

    Sender and receiver each own a registry. Releasing a request drops only the
    registry reference; previously restored plans retain their media and arrays.
    """

    def __init__(self) -> None:
        self._requests: dict[int, _Request] = {}

    def prepare(self, plan: BatchPlan) -> PlanPacket:
        pending: dict[int, _Request] = {}
        registrations = []
        sampling = None
        for row, (raw_id, media) in enumerate(zip(plan.sequence_ids, plan.media)):
            sequence_id = int(raw_id)
            existing = self._requests.get(sequence_id)
            if existing is not None:
                if existing.media is not media:
                    raise ValueError(f"request {sequence_id} changed its media object")
                continue
            if sampling is None:
                sampling = plan.sampling_params
            request = _Request(sampling[row], media)
            registrations.append(_register(sequence_id, request))
            pending[sequence_id] = request
        header, data = _pack(plan)
        packet = PlanPacket(header, data, tuple(registrations))
        self._requests.update(pending)
        return packet

    def restore(self, packet: PlanPacket) -> BatchPlan:
        values, groups = _unpack(packet)
        pending: dict[int, _Request] = {}
        for registration in packet.registrations:
            sequence_id, request = _restore_registration(registration)
            if sequence_id in self._requests or sequence_id in pending:
                raise ValueError(f"request {sequence_id} was already registered")
            pending[sequence_id] = request
        requests = []
        for raw_id in values["sequence_ids"]:
            sequence_id = int(raw_id)
            request = pending.get(sequence_id) or self._requests.get(sequence_id)
            if request is None:
                raise ValueError(f"request {sequence_id} has not been registered")
            requests.append(request)
        # Native constructors take owned copies of the raw array views, so a
        # receive buffer can be reused while an earlier GPU step is in flight.
        plan = BatchPlan(
            **values,
            cache_groups=groups,
            sampling_params=[request.sampling_params for request in requests],
            media=[request.media for request in requests],
        )
        self._requests.update(pending)
        return plan

    def release(self, sequence_ids: Iterable[int]) -> None:
        for sequence_id in sequence_ids:
            self._requests.pop(int(sequence_id), None)

    def clear(self) -> None:
        self._requests.clear()
