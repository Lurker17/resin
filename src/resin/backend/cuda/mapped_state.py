"""Native-StatePool-backed fixed-address CUDA state."""

from __future__ import annotations

from dataclasses import dataclass

from ..checks import positive_int


@dataclass(frozen=True, slots=True)
class MappedStateRow:
    slot: int
    generation: int
    engine_id: int
    request_id: int


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

        from resin import mint_state_mapping_arena_id

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
        if tuple(plan.state_indptr) != tuple(range(row_count + 1)):
            raise ValueError("mapped-state CSR must contain exactly one handle per row")
        if not (len(slots) == len(state_generations) == len(access_modes) == row_count):
            raise ValueError(
                "mapped-state metadata must contain exactly one handle per row"
            )
        if any(mode != self.access_mode for mode in access_modes):
            raise ValueError(
                f"mapped-state handles must all use {self.access_mode} access"
            )

        rows = []
        seen: dict[int, int] = {}
        for request_id, slot, generation in zip(
            request_ids, slots, state_generations, strict=True
        ):
            if isinstance(slot, bool) or not isinstance(slot, int):
                raise ValueError("mapped-state slot must be an integer")
            if slot < 0 or slot >= capacity:
                raise ValueError(
                    f"mapped-state slot {slot!r} is outside capacity {capacity}"
                )
            positive_int("mapped-state generation", generation)
            previous = seen.get(slot)
            if previous == generation:
                raise ValueError(
                    "duplicate mapped-state (slot, generation) in one batch"
                )
            if previous is not None:
                raise ValueError("one mapped-state slot cannot serve two batch rows")
            seen[slot] = generation

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
            rows.append(MappedStateRow(slot, generation, engine_id, request_id))
        return tuple(rows)

    def prepare_rows(self, rows: tuple[MappedStateRow, ...]) -> None:
        if self.slab is None or self.generations is None or self.owners is None:
            raise RuntimeError("mapped-state arena disappeared")
        for row in rows:
            if self.generations[row.slot] == row.generation:
                continue
            self.slab[:, row.slot].zero_()
            self.generations[row.slot] = row.generation
            self.owners[row.slot] = (row.engine_id, row.request_id)

    def view(self, layer_slot: int, row: MappedStateRow):
        if self.slab is None:
            raise RuntimeError("mapped-state arena disappeared")
        return self.slab[layer_slot, row.slot]
