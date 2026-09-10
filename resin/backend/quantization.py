"""Backend-neutral checkpoint encodings with no kernel registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class WeightEncoding:
    """Physical checkpoint encoding for one semantic model component."""

    name: str
    group_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("weight encoding name must be nonempty")
        if self.group_size is not None and (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size <= 0
        ):
            raise ValueError("weight encoding group_size must be positive")


BF16 = WeightEncoding("bf16")
FP8_TENSOR = WeightEncoding("fp8_tensor")
NVFP4 = WeightEncoding("nvfp4", group_size=16)
MXFP4 = WeightEncoding("mxfp4", group_size=32)


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Exact semantic component-to-encoding declarations.

    This object describes checkpoint bytes. A model still receives concrete
    kernel implementations as ordinary constructor arguments and validates
    those implementations separately.
    """

    components: Mapping[str, WeightEncoding] = field(default_factory=dict)
    default: WeightEncoding = BF16

    def __post_init__(self) -> None:
        if not isinstance(self.default, WeightEncoding):
            raise TypeError("default quantization must be a WeightEncoding")
        try:
            components = dict(self.components)
        except (TypeError, ValueError) as error:
            raise TypeError("quantization components must be a mapping") from error
        for name, encoding in components.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("quantization component names must be nonempty")
            if not isinstance(encoding, WeightEncoding):
                raise TypeError("quantization encodings must be WeightEncoding values")
        object.__setattr__(self, "components", MappingProxyType(components))

    def resolve(self, component: str) -> WeightEncoding:
        if not isinstance(component, str) or not component.strip():
            raise ValueError("quantization component name must be nonempty")
        return self.components.get(component, self.default)

    def require(self, component: str, checkpoint: WeightEncoding) -> WeightEncoding:
        selected = self.resolve(component)
        if selected != checkpoint:
            raise ValueError(
                f"checkpoint component {component!r} uses {checkpoint.name!r}, "
                f"but the quantization config selected {selected.name!r}"
            )
        return selected


__all__ = [
    "BF16",
    "FP8_TENSOR",
    "MXFP4",
    "NVFP4",
    "QuantizationConfig",
    "WeightEncoding",
]
