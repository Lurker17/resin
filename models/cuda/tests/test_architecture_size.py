"""Keep the CUDA package model-owned and small enough to audit."""

import ast
import inspect
from pathlib import Path

import pytest

import resin_models_cuda
from resin.backend.cuda.runner import CudaRunner
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner
from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
from resin_models_cuda.inkling.cuda import InklingCudaRunner
from resin_models_cuda.paged import PagedCudaRunner
from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner

MODEL_FAMILIES = {"deepseek_v4", "gemma4", "gpt_oss", "inkling", "qwen3_5"}
PACKAGE_BASELINE_LINES = 22_524
PACKAGE_MAX_LINES = 18_019
FORBIDDEN_CATEGORY_PACKAGES = {"drafts", "kernels", "loading", "media", "runners"}
FORBIDDEN_ROOT_SHIMS = {"paged_moe.py"}
BACKEND_NEUTRAL_SHIMS = {
    "deepseek_v4": {"codec.py", "config.py"},
    "gemma4": {"assistant.py", "codec.py", "config.py", "media.py", "plan.py"},
    "gpt_oss": {"config.py", "model.py", "plan.py"},
    "inkling": {"codec.py", "config.py", "media.py"},
    "qwen3_5": {"codec.py", "config.py", "media.py"},
}
PACKAGE_ROOT = Path(resin_models_cuda.__file__).parent
CUDA_RUNNERS = (
    DeepSeekV4CudaRunner,
    Gemma4CudaRunner,
    GptOssCudaRunner,
    InklingCudaRunner,
    PagedCudaRunner,
    Qwen3_5CudaRunner,
)


def test_cuda_model_package_is_at_least_twenty_percent_smaller() -> None:
    lines = sum(
        len(path.read_text().splitlines()) for path in PACKAGE_ROOT.rglob("*.py")
    )
    assert lines <= PACKAGE_MAX_LINES, (
        f"CUDA model package: {lines} lines exceeds {PACKAGE_MAX_LINES} "
        f"(80% of the {PACKAGE_BASELINE_LINES}-line baseline)"
    )


def test_every_cuda_model_directory_is_an_architecture_family() -> None:
    families = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert families == MODEL_FAMILIES


@pytest.mark.parametrize("runner", CUDA_RUNNERS)
def test_every_cuda_runner_uses_the_common_graph_base(runner: type) -> None:
    assert issubclass(runner, CudaRunner)


@pytest.mark.parametrize("runner", CUDA_RUNNERS)
def test_model_runners_do_not_reach_into_the_graph_executor(runner: type) -> None:
    source = inspect.getsource(runner)
    assert "_cuda_graph_executor" not in source
    assert "CudaGraphExecutor" not in source
    assert ".capture_all(" not in source
    assert "def bind_resources" not in source
    assert "FlashInferSampler()" not in source


@pytest.mark.parametrize("category", sorted(FORBIDDEN_CATEGORY_PACKAGES))
def test_cuda_package_has_no_implementation_category_directories(category: str) -> None:
    assert not (PACKAGE_ROOT / category).exists()


@pytest.mark.parametrize("module", sorted(FORBIDDEN_ROOT_SHIMS))
def test_cuda_package_has_no_renamed_root_module_shims(module: str) -> None:
    assert not (PACKAGE_ROOT / module).exists()


def test_cuda_package_has_no_import_and_alias_only_modules() -> None:
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        ]
        shim_statements = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)
        assert not body or not all(isinstance(node, shim_statements) for node in body), (
            f"{path.relative_to(PACKAGE_ROOT)} only imports or renames symbols"
        )


@pytest.mark.parametrize(
    ("family", "module"),
    [
        (family, module)
        for family, modules in BACKEND_NEUTRAL_SHIMS.items()
        for module in modules
    ],
)
def test_cuda_families_have_no_backend_neutral_or_name_translation_shims(
    family: str, module: str
) -> None:
    assert not (PACKAGE_ROOT / family / module).exists()
