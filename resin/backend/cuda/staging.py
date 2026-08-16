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
        self.host_tensors = {}

    def host(self, name: str, values, dtype):
        values = np.asarray(values)
        tensor = torch.empty(
            values.shape,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        np.copyto(tensor.numpy(), values, casting="unsafe")
        self.host_tensors[name] = tensor
        return tensor

    def device(self, name: str, values, dtype):
        return self.host(name, values, dtype).to(
            device=self.target_device, non_blocking=True
        )

    def output(self, name: str, source):
        tensor = torch.empty(
            source.shape,
            dtype=source.dtype,
            device="cpu",
            pin_memory=True,
        )
        tensor.copy_(source, non_blocking=True)
        self.host_tensors[name] = tensor
        return tensor

    def tensors(self, fields: Mapping[str, tuple[object, object]], *, device: bool):
        stage = self.device if device else self.host
        return {
            name: stage(name, values, dtype) for name, (values, dtype) in fields.items()
        }

    def copy_into(
        self,
        targets: Mapping[str, object],
        values: Mapping[str, object],
    ) -> None:
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
                            f"device input {name!r} does not match its graph buffer"
                        )
                    source = value
                else:
                    source = self.host(name, value, target.dtype).view(target.shape)
                staged.append((name, target, source))
        except Exception as error:
            self.host_tensors.clear()
            raise StagingError("inputs do not fit fixed device buffers") from error
        for name, target, source in staged:
            target.copy_(source, non_blocking=True)
