"""Streaming weight loading for official GPT-OSS safetensors checkpoints.

This module operates on a model that an execution backend has already
constructed (normally on the meta device). Safetensors are opened only when a
load is requested.

Ordinary parameters and persistent buffers are assigned directly, one tensor
at a time.  Packed expert tensors are handed to an :class:`ExpertLayerBuilder`
one decoder layer at a time.  The builder owns kernel-specific transformation
and installation; this loader never needs to know whether the destination is a
FlashInfer, Triton, or another expert implementation.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Collection, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import torch

from resin.backend.loading import resolve_dtype

_EXPERT_KEY = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\.(?P<name>.+)$"
)


class CheckpointFormatError(ValueError):
    """The checkpoint manifest and physical safetensors files disagree."""


class CheckpointCoverageError(ValueError):
    """The checkpoint does not exactly cover the model and expert builder."""


class ExpertLayerBuilder(Protocol):
    """Kernel-specific consumer for one layer of packed expert tensors.

    ``expected_tensor_names`` returns checkpoint names relative to
    ``model.layers.<layer>.mlp.experts`` in the order they should be consumed.
    The loader does not retain a tensor after ``consume_tensor`` returns.  A
    builder may retain inputs that must be transformed together, but should
    install the prepared weights and release raw inputs in ``finish_layer``.
    """

    def expected_tensor_names(
        self, layer_index: int, config: Mapping[str, Any]
    ) -> Collection[str]: ...

    def begin_layer(self, layer_index: int, config: Mapping[str, Any]) -> None: ...

    def consume_tensor(self, layer_index: int, name: str, tensor: Any) -> None: ...

    def finish_layer(self, layer_index: int) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Validated, allocation-free description of a GPT-OSS checkpoint."""

    root: Path
    config: Mapping[str, Any]
    tensor_files: Mapping[str, Path]
    sharded: bool

    @property
    def tensor_keys(self) -> frozenset[str]:
        return frozenset(self.tensor_files)


@dataclass(frozen=True, slots=True)
class PeakMemoryStrategy:
    """Memory behavior guaranteed by :func:`load_gpt_oss_weights`."""

    shard_dict_materialized: bool
    checkpoint_tensors_in_flight: int
    largest_checkpoint_tensor_bytes: int
    device_transfer: str
    expert_scope: str
    builder_retention_included: bool


@dataclass(frozen=True, slots=True)
class WeightLoadReport:
    """Counts and peak-memory contract for one completed weight load."""

    ordinary_tensor_count: int
    expert_tensor_count: int
    expert_layer_count: int
    total_checkpoint_bytes: int
    meta_parameters_replaced: int
    peak_strategy: PeakMemoryStrategy


def _format_keys(keys: Collection[str], *, limit: int = 8) -> str:
    ordered = sorted(keys)
    shown = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f", ... ({len(ordered)} total)"
    return ", ".join(repr(key) for key in shown) + suffix


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointFormatError(
                    f"duplicate JSON key {key!r} in {path.name}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise CheckpointFormatError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointFormatError(f"expected a JSON object in {path}")
    return value


def _safe_open():
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError(
            "loading GPT-OSS weights requires safetensors; install safetensors"
        ) from error
    return safe_open


class _ShardTensorReader:
    """Read one tensor at a time while retaining at most one shard mapping."""

    def __init__(self) -> None:
        self._safe_open = _safe_open()
        self._stack = ExitStack()
        self._path: Path | None = None
        self._handle: Any | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._stack.close()
        self._stack = ExitStack()
        self._path = None
        self._handle = None

    def get_tensor(self, path: Path, key: str):
        if path != self._path:
            self.close()
            self._handle = self._stack.enter_context(
                self._safe_open(str(path), framework="pt", device="cpu")
            )
            self._path = path
        return self._handle.get_tensor(key)


def _physical_tensor_keys(path: Path) -> tuple[str, ...]:
    safe_open = _safe_open()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return tuple(handle.keys())
    except Exception as error:
        if isinstance(error, (ImportError, CheckpointFormatError)):
            raise
        raise CheckpointFormatError(
            f"cannot read safetensors header {path}: {error}"
        ) from error


def _validate_shard_name(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise CheckpointFormatError("safetensors index contains an invalid shard name")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.suffix != ".safetensors"
    ):
        raise CheckpointFormatError(
            f"safetensors shard must be a local .safetensors filename: {filename!r}"
        )
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint shard: {path}")
    return path


def inspect_gpt_oss_checkpoint(model_path: str | Path) -> CheckpointManifest:
    """Read and fully validate checkpoint metadata without loading tensor data.

    An indexed checkpoint is checked against every referenced safetensors
    header.  The index must name every physical tensor exactly once, point it
    at the shard that actually contains it, and omit no physical tensor.
    Without an index, exactly one ``model.safetensors`` file is accepted.
    """

    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    root = root.resolve()
    config = _load_json_object(root / "config.json")
    if config.get("model_type") != "gpt_oss":
        raise CheckpointFormatError(
            f"expected a GPT-OSS config, got {config.get('model_type')!r}"
        )

    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        model_file = root / "model.safetensors"
        other_shards = set(root.glob("model*.safetensors")) - {model_file}
        if other_shards:
            raise CheckpointFormatError(
                "sharded checkpoint is missing model.safetensors.index.json: "
                f"{_format_keys({path.name for path in other_shards})}"
            )
        if not model_file.is_file():
            raise FileNotFoundError(f"missing checkpoint weights: {model_file}")
        keys = _physical_tensor_keys(model_file)
        return CheckpointManifest(
            root=root,
            config=config,
            tensor_files={key: model_file for key in keys},
            sharded=False,
        )

    index = _load_json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointFormatError(
            f"{index_path.name} must contain a non-empty weight_map object"
        )
    if any(not isinstance(key, str) or not key for key in weight_map):
        raise CheckpointFormatError("safetensors index contains an invalid tensor key")
    if any(
        not isinstance(filename, str) or not filename
        for filename in weight_map.values()
    ):
        raise CheckpointFormatError("safetensors index contains an invalid shard name")

    indexed_files = {
        filename: _validate_shard_name(root, filename)
        for filename in set(weight_map.values())
    }
    unindexed_model_files = set(root.glob("model*.safetensors")) - set(
        indexed_files.values()
    )
    if unindexed_model_files:
        raise CheckpointFormatError(
            "checkpoint contains model shards absent from the index: "
            f"{_format_keys({path.name for path in unindexed_model_files})}"
        )

    physical_locations: dict[str, Path] = {}
    duplicate_locations: dict[str, set[str]] = defaultdict(set)
    for path in indexed_files.values():
        for key in _physical_tensor_keys(path):
            if key in physical_locations:
                duplicate_locations[key].update(
                    (physical_locations[key].name, path.name)
                )
            else:
                physical_locations[key] = path
    if duplicate_locations:
        details = ", ".join(
            f"{key!r} in {sorted(files)!r}"
            for key, files in sorted(duplicate_locations.items())
        )
        raise CheckpointFormatError(f"duplicate tensors across shards: {details}")

    indexed_keys = set(weight_map)
    physical_keys = set(physical_locations)
    missing_physical = indexed_keys - physical_keys
    unindexed_physical = physical_keys - indexed_keys
    if missing_physical or unindexed_physical:
        parts = []
        if missing_physical:
            parts.append(f"missing physical tensors: {_format_keys(missing_physical)}")
        if unindexed_physical:
            parts.append(
                f"unindexed physical tensors: {_format_keys(unindexed_physical)}"
            )
        raise CheckpointFormatError("; ".join(parts))

    misplaced = {
        key
        for key, filename in weight_map.items()
        if physical_locations[key] != indexed_files[filename]
    }
    if misplaced:
        raise CheckpointFormatError(
            "index points tensors at the wrong shard: " + _format_keys(misplaced)
        )

    return CheckpointManifest(
        root=root,
        config=config,
        tensor_files=physical_locations,
        sharded=True,
    )


def _num_hidden_layers(config: Mapping[str, Any]) -> int:
    value = config.get("num_hidden_layers")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CheckpointFormatError(
            "GPT-OSS config num_hidden_layers must be a positive integer"
        )
    return value


def _model_state_keys(model: Any) -> tuple[str, ...]:
    try:
        keys = tuple(model.state_dict().keys())
    except AttributeError as error:
        raise TypeError(
            "model must provide a PyTorch-compatible state_dict()"
        ) from error
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise CheckpointCoverageError(
            "model state_dict contains duplicate keys: " + _format_keys(duplicates)
        )
    return keys


def _expert_expectations(
    builder: ExpertLayerBuilder,
    config: Mapping[str, Any],
) -> dict[int, tuple[str, ...]]:
    expectations = {}
    for layer_index in range(_num_hidden_layers(config)):
        names = tuple(builder.expected_tensor_names(layer_index, config))
        if not names:
            raise CheckpointCoverageError(
                f"expert builder expects no tensors for layer {layer_index}"
            )
        invalid = {
            name
            for name in names
            if not isinstance(name, str) or not name or name.startswith(".")
        }
        if invalid:
            raise CheckpointCoverageError(
                f"invalid expert tensor names for layer {layer_index}: {invalid!r}"
            )
        if len(set(names)) != len(names):
            raise CheckpointCoverageError(
                f"duplicate expert tensor names for layer {layer_index}"
            )
        expectations[layer_index] = names
    return expectations


def _full_expert_key(layer_index: int, name: str) -> str:
    return f"model.layers.{layer_index}.mlp.experts.{name}"


def _validate_expected_coverage(
    manifest: CheckpointManifest,
    ordinary_keys: Collection[str],
    expert_names: Mapping[int, tuple[str, ...]],
) -> None:
    expected_experts = {
        _full_expert_key(layer_index, name)
        for layer_index, names in expert_names.items()
        for name in names
    }
    expected = set(ordinary_keys) | expected_experts
    actual = set(manifest.tensor_files)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing expected tensors: {_format_keys(missing)}")
        if unexpected:
            parts.append(f"unexpected checkpoint tensors: {_format_keys(unexpected)}")
        raise CheckpointCoverageError("; ".join(parts))


def _resolve_target(model: Any, key: str) -> tuple[Any, str, Any, bool]:
    parts = key.split(".")
    module = model
    for part in parts[:-1]:
        try:
            module = getattr(module, part)
        except AttributeError as error:
            raise CheckpointCoverageError(
                f"model state target disappeared while loading {key!r}"
            ) from error
    name = parts[-1]
    parameters = getattr(module, "_parameters", {})
    buffers = getattr(module, "_buffers", {})
    if name in parameters and parameters[name] is not None:
        return module, name, parameters[name], True
    if name in buffers and buffers[name] is not None:
        return module, name, buffers[name], False
    raise CheckpointCoverageError(f"{key!r} is not a model parameter or buffer")


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _move_tensor(tensor: Any, *, device: Any, dtype: Any = None):
    if dtype is not None and tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def _assign_ordinary_tensor(
    torch,
    model: Any,
    key: str,
    tensor: Any,
    *,
    device: Any,
    dtype: Any,
) -> bool:
    module, name, existing, is_parameter = _resolve_target(model, key)
    if tuple(existing.shape) != tuple(tensor.shape):
        raise CheckpointCoverageError(
            f"shape mismatch for {key!r}: checkpoint {tuple(tensor.shape)}, "
            f"model {tuple(existing.shape)}"
        )
    was_meta = bool(getattr(existing, "is_meta", False))
    moved = _move_tensor(tensor, device=device, dtype=dtype)
    if is_parameter:
        module._parameters[name] = torch.nn.Parameter(
            moved, requires_grad=existing.requires_grad
        )
    else:
        module._buffers[name] = moved
    return was_meta


def load_gpt_oss_weights(
    model_path: str | Path,
    model: Any,
    *,
    device: Any,
    expert_builder: ExpertLayerBuilder,
    dtype: Any = None,
    ordinary_expected_keys: Collection[str] | None = None,
) -> tuple[Mapping[str, Any], WeightLoadReport]:
    """Stream an official GPT-OSS checkpoint into ``model`` and expert modules.

    The full key manifest is validated before the first tensor is read.  By
    default, ordinary expected keys come from ``model.state_dict()`` after all
    keys below an official GPT-OSS expert prefix are removed.  Backends with a
    nonstandard state representation may pass ``ordinary_expected_keys``.

    Floating ordinary tensors may be converted with ``dtype``.  Packed expert
    blocks and scales always retain their checkpoint dtype and are moved to the
    requested device before being passed to the builder.
    """

    target = torch.device(device)
    if target.type == "meta":
        raise ValueError("GPT-OSS weights must be loaded onto a material device")
    resolved_dtype = resolve_dtype(torch, dtype)
    if resolved_dtype is not None and not resolved_dtype.is_floating_point:
        raise ValueError("model dtype must be floating point")

    manifest = inspect_gpt_oss_checkpoint(model_path)
    expert_names = _expert_expectations(expert_builder, manifest.config)
    if ordinary_expected_keys is None:
        ordinary_keys = tuple(
            key
            for key in _model_state_keys(model)
            if _EXPERT_KEY.fullmatch(key) is None
        )
    else:
        ordinary_keys = tuple(ordinary_expected_keys)
        unknown = set(ordinary_keys) - set(_model_state_keys(model))
        if unknown:
            raise CheckpointCoverageError(
                "ordinary_expected_keys are absent from model.state_dict(): "
                + _format_keys(unknown)
            )
    if len(set(ordinary_keys)) != len(ordinary_keys):
        raise CheckpointCoverageError("ordinary_expected_keys contains duplicates")
    expert_as_ordinary = {key for key in ordinary_keys if _EXPERT_KEY.fullmatch(key)}
    if expert_as_ordinary:
        raise CheckpointCoverageError(
            "ordinary_expected_keys must not contain expert tensors: "
            + _format_keys(expert_as_ordinary)
        )
    _validate_expected_coverage(manifest, ordinary_keys, expert_names)

    ordinary_count = 0
    expert_count = 0
    total_bytes = 0
    largest_bytes = 0
    meta_replaced = 0

    # Opening a safetensors file parses its header and establishes its backing
    # mapping. Network/FUSE stores can make doing that once per tensor
    # catastrophically slow. Conversely, retaining every 120B shard mapping
    # can keep tens of gigabytes of faulted pages resident. Model keys are
    # layer-ordered, so a one-active-shard reader provides bounded RSS with a
    # small number of shard switches while still transferring one tensor at a
    # time.
    with _ShardTensorReader() as shard_reader:
        for key in ordinary_keys:
            tensor = shard_reader.get_tensor(manifest.tensor_files[key], key)
            size = _tensor_bytes(tensor)
            total_bytes += size
            largest_bytes = max(largest_bytes, size)
            meta_replaced += int(
                _assign_ordinary_tensor(
                    torch,
                    model,
                    key,
                    tensor,
                    device=target,
                    dtype=resolved_dtype,
                )
            )
            ordinary_count += 1
            del tensor

        for layer_index, names in expert_names.items():
            expert_builder.begin_layer(layer_index, manifest.config)
            for name in names:
                key = _full_expert_key(layer_index, name)
                tensor = shard_reader.get_tensor(manifest.tensor_files[key], key)
                size = _tensor_bytes(tensor)
                total_bytes += size
                largest_bytes = max(largest_bytes, size)
                moved = _move_tensor(tensor, device=target)
                del tensor
                expert_builder.consume_tensor(layer_index, name, moved)
                expert_count += 1
                del moved
            expert_builder.finish_layer(layer_index)

    strategy = PeakMemoryStrategy(
        shard_dict_materialized=False,
        checkpoint_tensors_in_flight=1,
        largest_checkpoint_tensor_bytes=largest_bytes,
        device_transfer=(
            "one CPU checkpoint tensor plus one target-device copy may overlap "
            "during a transfer"
        ),
        expert_scope=(
            "one expert layer at a time; the builder controls and must release "
            "any retained raw tensors in finish_layer"
        ),
        builder_retention_included=False,
    )
    return manifest.config, WeightLoadReport(
        ordinary_tensor_count=ordinary_count,
        expert_tensor_count=expert_count,
        expert_layer_count=len(expert_names),
        total_checkpoint_bytes=total_bytes,
        meta_parameters_replaced=meta_replaced,
        peak_strategy=strategy,
    )


__all__ = [
    "CheckpointCoverageError",
    "CheckpointFormatError",
    "CheckpointManifest",
    "ExpertLayerBuilder",
    "PeakMemoryStrategy",
    "WeightLoadReport",
    "inspect_gpt_oss_checkpoint",
    "load_gpt_oss_weights",
]
