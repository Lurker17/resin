"""Pinned host and asynchronous device tensor staging."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


class StagingError(ValueError):
    """Host values do not fit the requested fixed device buffers."""


class TensorStager:
    """Own host tensors until their asynchronous copies are complete."""

    def __init__(self, device) -> None:
        self.target_device = device
        self.host_tensors: dict[str, torch.Tensor] = {}

    def host(self, name: str, values, dtype):
        values = np.asarray(values)
        tensor = self.host_tensors.get(name)
        if tensor is None or tensor.shape != values.shape or tensor.dtype != dtype:
            tensor = torch.empty(
                values.shape,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self.host_tensors[name] = tensor
        np.copyto(tensor.numpy(), values, casting="unsafe")
        return tensor

    def device(self, name: str, values, dtype):
        return self.host(name, values, dtype).to(
            device=self.target_device, non_blocking=True
        )

    def output(self, name: str, source):
        tensor = self.host_tensors.get(name)
        if (
            tensor is None
            or tensor.shape != source.shape
            or tensor.dtype != source.dtype
        ):
            tensor = torch.empty(
                source.shape,
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )
            self.host_tensors[name] = tensor
        tensor.copy_(source, non_blocking=True)
        return tensor

    def tensors(self, fields: Mapping[str, tuple[object, object]], *, device: bool):
        stage = self.device if device else self.host
        return {
            name: stage(name, values, dtype) for name, (values, dtype) in fields.items()
        }

    def copy_into(
        self,
        targets: Mapping[str, torch.Tensor],
        values: Mapping[str, object],
    ) -> None:
        if targets is values:
            return
        if targets.keys() != values.keys():
            missing = targets.keys() - values.keys()
            extra = values.keys() - targets.keys()
            raise StagingError(
                f"staged inputs do not match buffers (missing={sorted(missing)}, "
                f"extra={sorted(extra)})"
            )
        staged = []
        try:
            for name, target in targets.items():
                value = values[name]
                if isinstance(value, getattr(torch, "Tensor", ())):
                    if value.shape != target.shape or value.dtype != target.dtype:
                        raise StagingError(
                            f"device input {name!r} has shape {tuple(value.shape)} and dtype {value.dtype}; "
                            f"expected shape {tuple(target.shape)} and dtype {target.dtype}"
                        )
                    source = value
                else:
                    source = self.host(name, value, target.dtype).view(target.shape)
                staged.append((name, target, source))
        except StagingError:
            raise
        except Exception as error:
            raise StagingError(
                f"input {name!r} does not fit graph buffer with shape "
                f"{tuple(target.shape)} and dtype {target.dtype}: {error}"
            ) from error
        for name, target, source in staged:
            target.copy_(source, non_blocking=True)
