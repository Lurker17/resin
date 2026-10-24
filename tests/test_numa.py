from __future__ import annotations

import ctypes
import errno
import json
import os
import subprocess
import sys
from dataclasses import asdict
from typing import Any

import pytest

import resin


@pytest.fixture
def host(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes"
    for node, cpus in enumerate(("0-3", "4-7")):
        path = nodes / f"node{node}"
        path.mkdir(parents=True)
        (path / "cpulist").write_text(cpus)
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nMems_allowed_list:\t0-1\n")
    pci = tmp_path / "pci" / "0000:ab:00.0"
    pci.mkdir(parents=True)
    (pci / "numa_node").write_text("1\n")
    monkeypatch.setattr(resin.numa.sys, "platform", "linux")
    monkeypatch.setattr(resin.numa, "_NODE_ROOT", nodes)
    monkeypatch.setattr(resin.numa, "_PCI_ROOT", pci.parent)
    monkeypatch.setattr(resin.numa, "_STATUS", status)
    state: dict[str, Any] = {"cpus": set(range(8)), "policy": (2 | (1 << 15), [0, 1])}

    def set_affinity(pid, cpus):
        assert pid == 0
        state["cpus"] = set(cpus)

    def set_policy(mode, nodes):
        state["policy"] = (mode, list(nodes))

    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: state["cpus"].copy(), raising=False
    )
    monkeypatch.setattr(os, "sched_setaffinity", set_affinity, raising=False)
    monkeypatch.setattr(
        resin._resin, "_numa_get_policy", lambda: state["policy"], raising=False
    )
    monkeypatch.setattr(resin._resin, "_numa_set_policy", set_policy, raising=False)
    return state


def test_binding_restores_full_policy_after_body_error(host):
    original = dict(host)
    with pytest.raises(LookupError, match="startup failed"):
        with resin.numa.bind(node=1, cpus=[4, 5], strict=True) as binding:
            assert host == {"cpus": {4, 5}, "policy": (1, [1])}
            assert asdict(binding) == {
                "device": None,
                "node": 1,
                "cpus": (4, 5),
                "memory_policy": "preferred",
                "reason": None,
            }
            raise LookupError("startup failed")
    assert host == original


def test_nested_placement_restores_outer_binding(host):
    original = dict(host)
    with resin.numa.bind(node=1, strict=True):
        with resin.numa.bind(cpus=[4], strict=True) as binding:
            assert host["cpus"] == {4}
            assert host["policy"] == (1, [1])
            assert binding.memory_policy is None
        assert host["cpus"] == {4, 5, 6, 7}
        assert host["policy"] == (1, [1])
    assert host == original


def test_cpu_mask_intersects_allowed_affinity(host):
    host["cpus"] = {3, 4, 5}
    with resin.numa.bind(node=1, memory="bind", strict=True) as binding:
        assert binding.cpus == (4, 5)
        assert host["policy"] == (2, [1])
    assert host["cpus"] == {3, 4, 5}


def test_device_discovery_uses_cuda_logical_ordinal(host, monkeypatch):
    ordinals = []

    def device_node(ordinal):
        ordinals.append(ordinal)
        return 1

    monkeypatch.setattr(resin.numa, "_cuda_node", device_node)
    with resin.numa.bind(device="cuda:3", strict=True) as binding:
        assert binding.node == 1 and binding.cpus == (4, 5, 6, 7)
    assert ordinals == [3]
    with resin.numa.bind(device="pci:0000:AB:00.0", strict=True) as binding:
        assert binding.node == 1


@pytest.mark.parametrize("attribute_supported", [True, False])
def test_cuda_queries_host_node_without_a_context_and_can_fall_back_to_pci(
    host, monkeypatch, attribute_supported
):
    from types import SimpleNamespace

    def initialize(flags):
        assert flags == 0
        return 0

    def get_device(pointer, ordinal):
        assert ordinal == 3
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 103
        return 0

    def get_attribute(pointer, attribute, device):
        assert attribute == 134 and device == 103
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0 if attribute_supported else 1

    def get_pci_bus_id(buffer, length, device):
        assert not attribute_supported
        assert device == 103 and length >= 13
        buffer.value = b"0000:ab:00.0"
        return 0

    driver = SimpleNamespace(
        cuInit=initialize,
        cuDeviceGet=get_device,
        cuDeviceGetAttribute=get_attribute,
        cuDeviceGetPCIBusId=get_pci_bus_id,
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda name: driver)
    assert resin.numa._cuda_node(3) == 1


@pytest.mark.parametrize("operation", ["_numa_get_policy", "_numa_set_policy"])
def test_denied_memory_policy_keeps_cpu_binding_and_reports_fallback(
    host, monkeypatch, operation
):
    original = dict(host)

    def denied(*args):
        raise PermissionError(errno.EPERM, "seccomp denied memory policy")

    monkeypatch.setattr(resin._resin, operation, denied)
    with pytest.warns(resin.numa.NumaWarning, match="binding CPUs only"):
        with resin.numa.bind(node=1) as binding:
            assert host["cpus"] == {4, 5, 6, 7}
            assert binding.memory_policy is None
            assert binding.reason is not None and "seccomp" in binding.reason
    assert host == original


def test_required_policy_failure_restores_cpu_affinity(host, monkeypatch):
    original = dict(host)

    def denied(*args):
        raise PermissionError(errno.EPERM, "denied")

    monkeypatch.setattr(resin._resin, "_numa_set_policy", denied)
    with pytest.raises(RuntimeError, match="memory policy unavailable"):
        with resin.numa.bind(node=1, strict=True):
            raise AssertionError("binding body must not run")
    assert host == original


def test_memory_node_restriction_falls_back_to_cpu_only(host, monkeypatch, tmp_path):
    status = tmp_path / "restricted_status"
    status.write_text("Mems_allowed_list:\t0\n")
    monkeypatch.setattr(resin.numa, "_STATUS", status)
    with pytest.warns(resin.numa.NumaWarning, match="not allowed for memory"):
        with resin.numa.bind(node=1) as binding:
            assert binding.cpus == (4, 5, 6, 7)
            assert binding.memory_policy is None


def test_empty_cpu_intersection_does_not_change_parent(host):
    original = dict(host)
    with pytest.raises(RuntimeError, match="no CPUs"):
        with resin.numa.bind(node=1, cpus=[0], strict=True):
            raise AssertionError("binding body must not run")
    assert host == original


def test_affinity_restored_even_if_policy_restoration_fails(host, monkeypatch):
    original_cpus = host["cpus"].copy()
    set_policy = resin._resin._numa_set_policy

    def fail_restore(mode, nodes):
        if mode != 1:
            raise PermissionError(errno.EPERM, "restore denied")
        set_policy(mode, nodes)

    monkeypatch.setattr(resin._resin, "_numa_set_policy", fail_restore)
    with pytest.raises(PermissionError, match="restore denied"):
        with resin.numa.bind(node=1, strict=True):
            pass
    assert host["cpus"] == original_cpus


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"node": -1},
        {"cpus": []},
        {"cpus": [-1]},
        {"device": "cuda:-1"},
        {"device": "gpu:0"},
        {"device": "pci:../../bad"},
        {"device": "cuda:0", "node": 0},
        {"node": 0, "memory": "typo"},
    ],
)
def test_invalid_arguments_fail_before_os_changes(host, options):
    original = dict(host)
    with pytest.raises(ValueError):
        with resin.numa.bind(**options):
            raise AssertionError("invalid arguments must not enter the context")
    assert host == original


def test_unsupported_platform_is_observable(monkeypatch):
    monkeypatch.setattr(resin.numa.sys, "platform", "darwin")
    with resin.numa.bind(device="cpu", strict=True) as binding:
        assert binding == resin.numa.Binding(device="cpu")
    with pytest.warns(resin.numa.NumaWarning, match="only available on Linux"):
        with resin.numa.bind(device="cuda:0") as binding:
            assert binding.cpus == () and binding.reason
    with pytest.raises(RuntimeError, match="only available on Linux"):
        with resin.numa.bind(node=0, strict=True):
            raise AssertionError("required binding must fail")


@pytest.mark.skipif(
    sys.platform != "linux", reason="requires Linux affinity inheritance"
)
def test_fresh_interpreter_inherits_affinity_before_imports():
    get_affinity = getattr(os, "sched_getaffinity")
    previous = get_affinity(0)
    selected = min(previous)
    with resin.numa.bind(cpus=[selected], strict=True):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
    assert get_affinity(0) == previous
    stdout, _ = process.communicate(timeout=15)
    assert process.returncode == 0
    assert json.loads(stdout) == [selected]


@pytest.mark.skipif(
    sys.platform != "linux", reason="requires Linux memory-policy inheritance"
)
def test_fresh_interpreter_inherits_memory_policy():
    try:
        previous = resin._resin._numa_get_policy()
    except OSError:
        pytest.skip()
    allowed_cpus = getattr(os, "sched_getaffinity")(0)
    node = next(
        int(path.name[4:])
        for path in resin.numa._NODE_ROOT.glob("node[0-9]*")
        if resin.numa._ids((path / "cpulist").read_text()) & allowed_cpus
        and int(path.name[4:]) in resin.numa._allowed_memory_nodes()
    )
    try:
        with resin.numa.bind(node=node, strict=True):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import json; from resin._resin import _numa_get_policy; print(json.dumps(_numa_get_policy()))",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
    except RuntimeError:
        pytest.skip()
    assert resin._resin._numa_get_policy() == previous
    stdout, _ = process.communicate(timeout=15)
    assert process.returncode == 0
    assert json.loads(stdout) == [1, [node]]
