"""Scoped Linux CPU and NUMA placement, inherited by directly spawned children.

Keep the scope around ``Process.start()`` or ``subprocess.Popen()`` so placement
is established before the child imports Python modules. Use multiprocessing's
``spawn`` context, not a preexisting forkserver. Existing pages are not migrated.
"""

from __future__ import annotations

import errno
import os
import re
import sys
import warnings
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import Literal

_NODE_ROOT = Path("/sys/devices/system/node")
_PCI_ROOT = Path("/sys/bus/pci/devices")
_STATUS = Path("/proc/self/status")
_MEMORY_MODES = {"preferred": 1, "bind": 2}


class NumaWarning(RuntimeWarning):
    """Requested placement was only partly applied or is unavailable."""


@dataclass(frozen=True)
class Binding:
    """Effective placement inside a :func:`bind` scope.

    ``memory_policy=None`` means the previous policy was retained. ``reason``
    explains any fallback. CPU affinity is not a reservation of exclusive cores.
    """

    device: str | None = None
    node: int | None = None
    cpus: tuple[int, ...] = ()
    memory_policy: Literal["preferred", "bind"] | None = None
    reason: str | None = None


def _ids(value: str) -> set[int]:
    result = set()
    for part in value.strip().split(","):
        if not part:
            continue
        bounds = part.split("-")
        first, last = int(bounds[0]), int(bounds[-1])
        if len(bounds) > 2 or not 0 <= first <= last < 1_048_576:
            raise ValueError(f"invalid CPU/node list: {value!r}")
        result.update(range(first, last + 1))
    return result


def _pci_node(address: str) -> int:
    node = int((_PCI_ROOT / address / "numa_node").read_text())
    if node < 0:
        raise OSError(f"no NUMA locality is reported for PCI device {address}")
    return node


def _cuda_node(ordinal: int) -> int:
    # The CUDA driver resolves logical ordinals, including CUDA_VISIBLE_DEVICES
    # and CUDA_DEVICE_ORDER. cuInit/querying devices creates no CUDA context.
    import ctypes

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda.cuDeviceGetAttribute.restype = ctypes.c_int
    cuda.cuDeviceGetPCIBusId.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda.cuDeviceGetPCIBusId.restype = ctypes.c_int

    def check(status: int) -> None:
        if status:
            raise OSError(f"CUDA device discovery failed (driver error {status})")

    check(cuda.cuInit(0))
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), ordinal))
    node = ctypes.c_int(-1)
    # CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID (CUDA 12.3+), not NUMA_ID (GPU HBM).
    # Query the driver first: containers may hide the GPU's PCI sysfs entry.
    if cuda.cuDeviceGetAttribute(ctypes.byref(node), 134, device.value) == 0:
        if node.value >= 0:
            return node.value
    address = ctypes.create_string_buffer(32)
    check(cuda.cuDeviceGetPCIBusId(address, len(address), device.value))
    return _pci_node(address.value.decode("ascii").lower())


def _device_node(device: str) -> int:
    if device.startswith("pci:"):
        address = device[4:].lower()
        if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", address) is None:
            raise ValueError("PCI devices use 'pci:0000:ab:00.0' addresses")
    else:
        match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
        if match is None:
            raise ValueError("device must be 'cpu', 'cuda:N', or 'pci:0000:ab:00.0'")
        return _cuda_node(int(match[1] or 0))
    return _pci_node(address)


def _allowed_memory_nodes() -> set[int]:
    for line in _STATUS.read_text().splitlines():
        if line.startswith("Mems_allowed_list:"):
            return _ids(line.split(":", 1)[1])
    raise OSError("cannot read the allowed NUMA memory nodes")


def _fallback(reason: str, strict: bool) -> None:
    if strict:
        raise RuntimeError(reason)
    warnings.warn(reason, NumaWarning, stacklevel=4)


@contextmanager
def bind(
    *,
    device: str | None = None,
    node: int | None = None,
    cpus: Iterable[int] | None = None,
    memory: Literal["preferred", "bind"] = "preferred",
    strict: bool = False,
) -> Generator[Binding, None, None]:
    """Temporarily place the calling thread and its newly created children.

    Select a device's nearby NUMA node, an explicit ``node``, or only ``cpus``.
    The default memory policy prefers the node and permits allocation elsewhere
    under memory pressure; ``memory='bind'`` restricts allocation to that node.
    Only CPUs allowed by the current affinity are used. Both previous settings
    are restored on exit, including when child startup raises.

    ``strict=False`` warns and yields the effective :class:`Binding` if discovery
    or memory-policy permissions are unavailable. ``strict=True`` raises instead.
    ``device='cpu'`` with no explicit CPU set is a no-op on every platform.

    Example::

        import resin

        ctx = multiprocessing.get_context('spawn')
        process = ctx.Process(target=worker)
        with resin.numa.bind(device='cuda:0') as placement:
            process.start()
    """
    if device is not None and not isinstance(device, str):
        raise TypeError("device must be a string")
    if device is not None and node is not None:
        raise ValueError("choose either device or node")
    if node is not None:
        node = index(node)
        if node < 0:
            raise ValueError("node must be nonnegative")
    if memory not in _MEMORY_MODES:
        raise ValueError("memory must be 'preferred' or 'bind'")
    requested_cpus = None if cpus is None else {index(cpu) for cpu in cpus}
    if requested_cpus is not None and (not requested_cpus or min(requested_cpus) < 0):
        raise ValueError("cpus must contain nonnegative CPU IDs")
    if device is None and node is None and requested_cpus is None:
        raise ValueError("supply device, node, or cpus")
    if (
        device is not None
        and device != "cpu"
        and not (
            re.fullmatch(r"cuda(?::[0-9]+)?", device)
            or re.fullmatch(
                r"pci:[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", device
            )
        )
    ):
        raise ValueError("device must be 'cpu', 'cuda:N', or 'pci:0000:ab:00.0'")
    if device == "cpu" and requested_cpus is None:
        yield Binding(device=device)
        return
    if sys.platform != "linux":
        reason = "NUMA binding is only available on Linux"
        _fallback(reason, strict)
        yield Binding(device=device, node=node, reason=reason)
        return

    try:
        if device is not None and device != "cpu":
            node = _device_node(device)
        previous_cpus = os.sched_getaffinity(0)
        target_cpus = previous_cpus
        if node is not None:
            target_cpus = target_cpus & _ids(
                (_NODE_ROOT / f"node{node}" / "cpulist").read_text()
            )
        if requested_cpus is not None:
            target_cpus = target_cpus & requested_cpus
        if not target_cpus:
            raise OSError(errno.EINVAL, "placement has no CPUs in the allowed affinity")
    except OSError as error:
        reason = f"NUMA binding unavailable: {error}"
        _fallback(reason, strict)
        yield Binding(device=device, node=node, reason=reason)
        return

    previous_policy = None
    memory_policy = None
    reason = None
    if node is not None:
        from resin._resin import _numa_get_policy, _numa_set_policy

        try:
            if node not in _allowed_memory_nodes():
                raise OSError(
                    errno.EINVAL, f"NUMA node {node} is not allowed for memory"
                )
            previous_policy = _numa_get_policy()
        except OSError as error:
            reason = f"NUMA memory policy unavailable; binding CPUs only: {error}"
            _fallback(reason, strict)

    # Memory policy and affinity are per calling thread. Set/restore in this same
    # thread, and keep the body synchronous (do not yield to an async event loop).
    affinity_changed = False
    policy_changed = False
    try:
        try:
            os.sched_setaffinity(0, target_cpus)
            affinity_changed = True
        except OSError as error:
            reason = f"NUMA CPU binding unavailable: {error}"
            _fallback(reason, strict)
            yield Binding(device=device, node=node, reason=reason)
            return
        if previous_policy is not None:
            try:
                _numa_set_policy(_MEMORY_MODES[memory], [node])
                policy_changed = True
                memory_policy = memory
            except OSError as error:
                reason = f"NUMA memory policy unavailable; binding CPUs only: {error}"
                _fallback(reason, strict)
        yield Binding(
            device=device,
            node=node,
            cpus=tuple(sorted(target_cpus)),
            memory_policy=memory_policy,
            reason=reason,
        )
    finally:
        # Restore affinity even if restoring the memory policy fails. Restoration
        # failures must surface: continuing with a silently changed parent policy
        # would place subsequent ranks incorrectly.
        try:
            if policy_changed:
                _numa_set_policy(*previous_policy)
        finally:
            if affinity_changed:
                os.sched_setaffinity(0, previous_cpus)


__all__ = ["Binding", "NumaWarning", "bind"]
