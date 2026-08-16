"""Native-StatePool-backed fixed-address CUDA state."""

from __future__ import annotations

from dataclasses import dataclass

from resin import mint_state_mapping_arena_id
from resin.backend.checks import positive_int


@dataclass(frozen=True, slots=True)
class MappedStateRow:
    slot: int
    generation: int
    engine_id: int
    request_id: int
    copy_from_slot: int | None = None
    copy_from_generation: int | None = None


@dataclass(frozen=True, slots=True)
class _ArenaRecord:
    arena_id: int
    descriptor: tuple[object, ...]
    slab: object
    generations: list[int | None]
    owners: list[tuple[int, int] | None]


# Arena IDs identify process-owned physical bytes, not compatible replacements.
_MAPPED_STATE_ARENAS: dict[int, _ArenaRecord] = {}


class MappedStateArena:
    def __init__(
        self,
        torch,
        device,
        stream,
        *,
        protocol: str,
        layer_indices,
        row_shape,
        dtype,
        access_mode: str = "write",
        label: str = "mapped state",
    ) -> None:
        self.torch = torch
        self.device = device
        self.stream = stream
        self.protocol = protocol
        self.layer_indices = tuple(layer_indices)
        self.row_shape = tuple(row_shape)
        self.dtype = dtype
        self.access_mode = access_mode
        self.label = label
        self.binding: tuple[int, int, str] | None = None
        self.arena_id: int | None = None
        self.slab = None
        self.generations: list[int | None] | None = None
        self.owners: list[tuple[int, int] | None] | None = None

    def _validate_binding(self, pool_id: int, capacity: int, protocol: str):
        positive_int("state pool id", pool_id)
        positive_int("state pool capacity", capacity)
        if protocol != self.protocol:
            raise ValueError(
                f"unsupported mapped-state protocol: expected {self.protocol!r}, "
                f"got {protocol!r}"
            )
        binding = (pool_id, capacity, protocol)
        if self.binding is not None and binding != self.binding:
            raise ValueError(
                f"this {self.label} allocation is already bound to a different "
                "mapped-state pool"
            )
        return binding

    def _descriptor(self, binding: tuple[int, int, str]):
        pool_id, capacity, protocol = binding
        shape = (len(self.layer_indices), capacity, *self.row_shape)
        return (
            pool_id,
            capacity,
            protocol,
            self.layer_indices,
            shape,
            str(self.device),
            str(self.dtype),
        )

    def _validate_record(self, record: _ArenaRecord, binding) -> None:
        if record.descriptor != self._descriptor(binding):
            raise ValueError(
                f"required mapped-state arena {record.arena_id} has incompatible "
                "pool identity or physical geometry"
            )

    def prepare_binding(
        self,
        pool_id: int,
        capacity: int,
        protocol: str,
        required_arena_id: int | None,
    ) -> tuple[tuple[int, int, str], _ArenaRecord | None, int]:
        binding = self._validate_binding(pool_id, capacity, protocol)
        if required_arena_id is not None:
            positive_int("required mapped-state arena id", required_arena_id)

        if self.binding is not None:
            if required_arena_id is not None and required_arena_id != self.arena_id:
                raise ValueError(
                    f"this {self.label} arena does not match the StatePool's required "
                    "physical arena"
                )
            return binding, None, self.arena_id

        if required_arena_id is not None:
            record = _MAPPED_STATE_ARENAS.get(required_arena_id)
            if record is None:
                raise ValueError(
                    f"required mapped-state arena {required_arena_id} is not "
                    "registered in this process"
                )
            self._validate_record(record, binding)
            return binding, record, record.arena_id

        arena_id = positive_int(
            "native mapped-state arena id", mint_state_mapping_arena_id()
        )
        if arena_id in _MAPPED_STATE_ARENAS:
            raise RuntimeError(
                f"mapped-state arena id {arena_id} is already registered"
            )
        descriptor = self._descriptor(binding)
        shape = descriptor[4]
        with self.torch.cuda.stream(self.stream):
            slab = self.torch.zeros(shape, dtype=self.dtype, device=self.device)
        record = _ArenaRecord(
            arena_id,
            descriptor,
            slab,
            [None] * capacity,
            [None] * capacity,
        )
        return binding, record, arena_id

    def commit_binding(self, binding, record: _ArenaRecord | None) -> int:
        if record is not None:
            self._validate_record(record, binding)
            registered = _MAPPED_STATE_ARENAS.setdefault(record.arena_id, record)
            if registered is not record:
                raise RuntimeError(
                    f"mapped-state arena {record.arena_id} was registered concurrently"
                )
            self.slab, self.generations, self.owners = (
                record.slab,
                record.generations,
                record.owners,
            )
            self.arena_id, self.binding = record.arena_id, binding
        if self.arena_id is None:
            raise RuntimeError("mapped-state binding did not produce an arena")
        return self.arena_id

    def rows(self, plan) -> tuple[MappedStateRow, ...]:
        binding = self.binding
        generations = self.generations
        owners = self.owners
        if binding is None or any(
            value is None for value in (self.slab, generations, owners)
        ):
            raise ValueError(f"{self.label} mapping is not bound")
        pool_id, capacity, _ = binding
        if plan.state_pool_id != pool_id:
            raise ValueError(
                f"BatchPlan state pool id {plan.state_pool_id!r} does not match "
                f"the bound mapped-state pool {pool_id!r}"
            )
        engine_id = positive_int("BatchPlan engine_id", plan.engine_id)

        request_ids = tuple(plan.sequence_ids)
        row_count = len(request_ids)
        slots = tuple(plan.state_slots)
        state_generations = tuple(plan.state_generations)
        access_modes = tuple(plan.state_access_modes)
        indptr = tuple(plan.state_indptr)
        if len(indptr) != row_count + 1 or indptr[0] != 0 or indptr[-1] != len(slots):
            raise ValueError("mapped-state CSR does not cover every state handle")
        if not (len(slots) == len(state_generations) == len(access_modes)):
            raise ValueError(
                "mapped-state slot, generation, and access metadata disagree"
            )

        rows = []
        seen: dict[int, tuple[int, str]] = {}
        for request_id, start, end in zip(
            request_ids, indptr[:-1], indptr[1:], strict=True
        ):
            row_slots = slots[start:end]
            row_generations = state_generations[start:end]
            row_modes = access_modes[start:end]
            if row_modes == (self.access_mode,):
                copy_from = None
                slot, generation = row_slots[0], row_generations[0]
            elif row_modes == ("read", self.access_mode):
                copy_from = (row_slots[0], row_generations[0])
                slot, generation = row_slots[1], row_generations[1]
            else:
                raise ValueError(
                    "mapped-state rows require either one write handle or an ordered "
                    "[read checkpoint, write destination] pair"
                )
            if isinstance(slot, bool) or not isinstance(slot, int):
                raise ValueError("mapped-state slot must be an integer")
            if slot < 0 or slot >= capacity:
                raise ValueError(
                    f"mapped-state slot {slot!r} is outside capacity {capacity}"
                )
            positive_int("mapped-state generation", generation)
            previous = seen.get(slot)
            if previous is not None:
                raise ValueError(
                    "one mapped-state destination cannot serve two batch rows"
                )
            seen[slot] = (generation, self.access_mode)

            current_generation = generations[slot]
            current_owner = owners[slot]
            if (current_generation is None) != (current_owner is None):
                raise RuntimeError("mapped-state ownership sidecars diverged")
            owner = (engine_id, request_id)
            if current_generation is not None and generation < current_generation:
                raise ValueError("mapped-state generation is stale")
            if current_generation == generation and current_owner != owner:
                raise ValueError(
                    "mapped-state slot/generation is already owned by a different "
                    "Engine request"
                )
            copy_slot = copy_generation = None
            if copy_from is not None:
                copy_slot, copy_generation = copy_from
                if isinstance(copy_slot, bool) or not isinstance(copy_slot, int):
                    raise ValueError("mapped-state checkpoint slot must be an integer")
                if copy_slot < 0 or copy_slot >= capacity:
                    raise ValueError(
                        f"mapped-state checkpoint slot {copy_slot!r} is outside "
                        f"capacity {capacity}"
                    )
                positive_int("mapped-state checkpoint generation", copy_generation)
                if generations[copy_slot] != copy_generation:
                    raise ValueError("mapped-state checkpoint generation is stale")
                previous = seen.get(copy_slot)
                if previous is not None and previous != (copy_generation, "read"):
                    raise ValueError(
                        "a mapped-state checkpoint cannot be written while it is copied"
                    )
                seen[copy_slot] = (copy_generation, "read")
            rows.append(
                MappedStateRow(
                    slot,
                    generation,
                    engine_id,
                    request_id,
                    copy_slot,
                    copy_generation,
                )
            )
        return tuple(rows)

    def prepare_rows(self, rows: tuple[MappedStateRow, ...]) -> None:
        if self.slab is None or self.generations is None or self.owners is None:
            raise RuntimeError("mapped-state arena disappeared")
        for row in rows:
            if self.generations[row.slot] == row.generation:
                continue
            if row.copy_from_slot is None:
                self.slab[:, row.slot].zero_()
            else:
                if self.generations[row.copy_from_slot] != row.copy_from_generation:
                    raise ValueError("mapped-state checkpoint generation became stale")
                self.slab[:, row.slot].copy_(self.slab[:, row.copy_from_slot])
            self.generations[row.slot] = row.generation
            self.owners[row.slot] = (row.engine_id, row.request_id)

    def view(self, layer_slot: int, row: MappedStateRow):
        if self.slab is None:
            raise RuntimeError("mapped-state arena disappeared")
        return self.slab[layer_slot, row.slot]
