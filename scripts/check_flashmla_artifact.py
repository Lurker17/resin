"""Check the pinned FlashMLA binary without importing its CUDA extension.

CUDA 13.3 spills in the unused legacy split-KV head64 decoder. Keep the
upstream CUTLASS exemption and that precise namespace; all other kernels,
including Resin's fused prefill/decode/projection kernels, must be spill-free.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import distribution
from pathlib import Path

FUSED = "_ZN5sm1007prefill34fused_norm_rope_attn_rope_cast_fwd"
LEGACY_SPLIT_KV = (
    "_ZN5sm1006decode6sparse6head6439flash_fwd_splitkv_mla_fp8_sparse_kernelI"
)
REQUIRED = {
    "h64_prefill": re.compile(
        FUSED + r"9core_attn10fwd_kernel.*SparseAttnFwdMode0.*ELj64ELb0"
    ),
    "h64_fp8_decode": re.compile(
        FUSED
        + r"9core_attn10fwd_kernel.*SparseAttnFwdMode1EL9ModelType0ELS\w+_0ELj64ELb0"
    ),
    "h64_fp4_decode": re.compile(
        FUSED
        + r"9core_attn10fwd_kernel.*SparseAttnFwdMode1EL9ModelType0ELS\w+_1ELj64ELb0"
    ),
    "query_permutation": re.compile(
        FUSED + "16permute_q_b_proj23permute_q_b_proj_kernel"
    ),
    "output_permutation": re.compile(
        FUSED + "15permute_wv_proj22permute_wv_proj_kernel"
    ),
}


@dataclass(frozen=True)
class Resource:
    architecture: str
    name: str
    registers: int
    stack: int
    local: int


def parse_resources(output: str) -> list[Resource]:
    records, architecture, name = [], "unknown", None
    for line in output.splitlines():
        if match := re.match(r"\s*arch\s*=\s*(\S+)", line):
            architecture = match[1]
        elif match := re.match(r"\s*Function\s+(.+?)\s*$", line):
            if name is not None:
                raise ValueError(f"missing resource usage for {name}")
            name = match[1].removeprefix(":").strip().removesuffix(":")
        elif name is not None and "REG:" in line:
            fields = dict(re.findall(r"\b(REG|STACK|LOCAL):\s*(\d+)", line))
            if set(fields) != {"REG", "STACK", "LOCAL"}:
                raise ValueError(f"incomplete resource usage for {name}: {line}")
            records.append(
                Resource(
                    architecture,
                    name,
                    int(fields["REG"]),
                    int(fields["STACK"]),
                    int(fields["LOCAL"]),
                )
            )
            name = None
    if name is not None or not records:
        raise ValueError(
            "cuobjdump returned missing or unparseable kernel resource usage"
        )
    return records


def check_resources(records: list[Resource]) -> dict:
    required = {
        label: sum(bool(pattern.match(row.name)) for row in records)
        for label, pattern in REQUIRED.items()
    }
    if missing := [label for label, count in required.items() if not count]:
        raise ValueError(f"missing required FlashMLA kernels: {', '.join(missing)}")
    spills = [row for row in records if row.stack or row.local]
    unexpected = [
        row
        for row in spills
        if not row.name.startswith(("_ZN7cutlass", LEGACY_SPLIT_KV))
    ]
    if unexpected:
        raise ValueError(
            "unexpected FlashMLA spills: "
            + json.dumps([asdict(row) for row in unexpected])
        )
    return {
        "kernels": len(records),
        "architectures": dict(Counter(row.architecture for row in records)),
        "required_kernels": required,
        "allowed_cutlass_spills": sum(
            row.name.startswith("_ZN7cutlass") for row in spills
        ),
        "allowed_legacy_split_kv_spills": sum(
            row.name.startswith(LEGACY_SPLIT_KV) for row in spills
        ),
        "allowed_spills": [asdict(row) for row in spills],
    }


def installed_artifact() -> Path:
    package = distribution("flash-mla")
    artifacts = [
        Path(str(package.locate_file(file)))
        for file in package.files or ()
        if file.parent.as_posix() == "flash_mla"
        and file.name.startswith("cuda")
        and file.suffix == ".so"
    ]
    if len(artifacts) != 1:
        raise ValueError(
            f"expected one installed FlashMLA CUDA artifact, found {artifacts}"
        )
    return artifacts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path)
    parser.add_argument(
        "--cuobjdump",
        default=shutil.which("cuobjdump")
        or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/cuobjdump"),
    )
    args = parser.parse_args()
    artifact = args.artifact if args.artifact is not None else installed_artifact()
    result = subprocess.run(
        [args.cuobjdump, "-res-usage", str(artifact)],
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                **check_resources(parse_resources(result.stdout)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
