from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


class TestImport:
    def test_import_resin_does_not_import_accelerator_stacks(self) -> None:
        code = (
            "import sys; import resin; import resin.backend; "
            "import resin.backend.cuda; import resin_models_cuda; "
            "import resin_models_torch; "
            "print(*(int(name in sys.modules) for name in "
            "('torch', 'flashinfer', 'flash_attn')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert result.stdout.strip() == "0 0 0"

    def test_accelerator_dependencies_are_optional(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
        assert project["dependencies"] == ["blake3>=1.0,<2"]
        optional_dependencies = project["optional-dependencies"]
        assert "torch" not in optional_dependencies
        assert "demo" not in optional_dependencies

        gpu = optional_dependencies["gpu"]
        assert any(item.startswith("torch") for item in gpu)
        assert any(item.startswith("flashinfer-python") for item in gpu)
        assert any(item.startswith("quack-kernels==0.6.4") for item in gpu)
        assert any(item.startswith("nvidia-cutlass-dsl") for item in gpu)
        assert any(item.startswith("flash-attn-4") for item in gpu)
        assert not any(item.startswith("numpy") for item in gpu)
        assert not any(item.startswith("huggingface-hub") for item in gpu)
        assert not any(item.startswith("safetensors") for item in gpu)

        cuda_models = tomllib.loads(Path("models/cuda/pyproject.toml").read_text())[
            "project"
        ]
        assert "resin[gpu]==0.1.0" in cuda_models["dependencies"]
        assert "huggingface-hub==1.27.0" in cuda_models["dependencies"]
        assert "safetensors==0.8.0" in cuda_models["dependencies"]

        torch_models = tomllib.loads(Path("models/torch/pyproject.toml").read_text())[
            "project"
        ]
        assert "resin==0.1.0" in torch_models["dependencies"]
        assert any(item.startswith("torch>=") for item in torch_models["dependencies"])
        assert any(
            item.startswith("huggingface-hub>=")
            for item in torch_models["dependencies"]
        )
        assert any(
            item.startswith("safetensors>=") for item in torch_models["dependencies"]
        )
