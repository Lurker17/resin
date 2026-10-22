"""The FlashMLA build exemption must never hide a fused-kernel spill."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_flashmla_artifact as artifact

FUSED = artifact.FUSED
NAMES = [
    FUSED
    + "9core_attn10fwd_kernelINS2_6KernelIXtlNS2_6ConfigEL17SparseAttnFwdMode0EL9ModelType0ELS7_0ELj64ELb0EEEEEEEvNT_6ParamsE",
    FUSED
    + "9core_attn10fwd_kernelINS2_6KernelIXtlNS2_6ConfigEL17SparseAttnFwdMode1EL9ModelType0ELS7_0ELj64ELb0EEEEEEEvNT_6ParamsE",
    FUSED
    + "9core_attn10fwd_kernelINS2_6KernelIXtlNS2_6ConfigEL17SparseAttnFwdMode1EL9ModelType0ELS7_1ELj64ELb0EEEEEEEvNT_6ParamsE",
    FUSED + "16permute_q_b_proj23permute_q_b_proj_kernelENS2_6ParamsE",
    FUSED + "15permute_wv_proj22permute_wv_proj_kernelENS2_6ParamsE",
]


def _dump(names=NAMES, *, stack=0, local=0):
    return "arch = sm_100a\n" + "\n".join(
        f"    Function {name}:\n        REG:64 STACK:{stack} SHARED:1024 LOCAL:{local} CONSTANT[0]:0\n"
        for name in names
    )


def test_parse_and_check_both_architectures_with_only_narrow_spill_exemptions():
    output = _dump()
    output += _dump().replace("sm_100a", "sm_103a")
    output += _dump(["_ZN7cutlass13device_kernelIexample"], stack=64)
    output += _dump([artifact.LEGACY_SPLIT_KV + "example"], local=8)
    rows = artifact.parse_resources(output)
    assert rows[0] == artifact.Resource("sm_100a", NAMES[0], 64, 0, 0)
    report = artifact.check_resources(rows)
    assert report["kernels"] == 12
    assert report["architectures"] == {"sm_100a": 7, "sm_103a": 5}
    assert set(report["required_kernels"].values()) == {2}
    assert report["allowed_cutlass_spills"] == 1
    assert report["allowed_legacy_split_kv_spills"] == 1


@pytest.mark.parametrize(
    "name",
    [
        NAMES[0],
        "_ZN5sm1006decode6sparse6head6439flash_fwd_splitkv_mla_fp8_sparse_kernel_extra",
        "unrecognized_kernel",
    ],
)
def test_rejects_fused_and_unknown_spills(name):
    rows = artifact.parse_resources(_dump() + _dump([name], local=16))
    with pytest.raises(ValueError, match="unexpected FlashMLA spills"):
        artifact.check_resources(rows)


def test_rejects_missing_selected_kernel_family():
    with pytest.raises(ValueError, match="missing required.*output_permutation"):
        artifact.check_resources(artifact.parse_resources(_dump(NAMES[:-1])))


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Function incomplete:",
        "Function incomplete:\nREG:32 STACK:0 SHARED:8",
        "Function first:\nFunction second:\nREG:32 STACK:0 LOCAL:0",
    ],
)
def test_unparseable_resource_output_fails_closed(output):
    with pytest.raises(ValueError):
        artifact.parse_resources(output)


def test_finds_installed_binary_from_metadata_without_importing_cuda(
    monkeypatch, tmp_path
):
    files = [
        Path("flash_mla/__init__.py"),
        Path("flash_mla/cuda.cpython-312-x86_64-linux-gnu.so"),
    ]
    package = SimpleNamespace(files=files, locate_file=lambda path: tmp_path / path)
    calls = []

    def distribution(name):
        calls.append(name)
        return package

    monkeypatch.setattr(artifact, "distribution", distribution)
    assert artifact.installed_artifact() == tmp_path / files[1]
    assert calls == ["flash-mla"]
