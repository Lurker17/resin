"""Model-neutral invariants for immutable prepared media inputs."""

import hashlib
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


def owned_cpu_tensor(value: Any, name: str, *, ndim: int) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be CPU-resident")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {tensor.ndim}")
    return tensor.detach().contiguous().clone()


def single_row(value: Any, name: str) -> Tensor:
    tensor = owned_cpu_tensor(value, name, ndim=2)
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one prompt")
    return tensor[0].to(torch.int64)


def salt_token_positions(
    token_ids: tuple[int, ...], positions: tuple[int, ...], digest: str
) -> tuple[int, ...]:
    identity = list(token_ids)
    stream = hashlib.shake_256(bytes.fromhex(digest)).digest(4 * len(positions))
    for index, position in enumerate(positions):
        identity[position] = int.from_bytes(stream[4 * index : 4 * index + 4], "little")
    return tuple(identity)


def digest_tensors(
    namespace: bytes,
    tensors: Iterable[tuple[str, Tensor | None]],
    *,
    metadata: Iterable[bytes] = (),
) -> str:
    """Hash canonical tensor inputs and optional model-specific metadata."""

    digest = hashlib.sha256(namespace)
    for value in metadata:
        digest.update(value)
    for name, tensor in tensors:
        if tensor is not None:
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(bytes(str(tuple(tensor.shape)), "ascii"))
            # NumPy cannot represent BF16 (or every Torch float8 dtype). Hash
            # storage bytes after recording dtype and shape above.
            digest.update(
                tensor.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
    return digest.hexdigest()


__all__ = [
    "digest_tensors",
    "owned_cpu_tensor",
    "salt_token_positions",
    "single_row",
]
