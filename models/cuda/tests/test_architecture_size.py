"""Keep the CUDA package model-owned and small enough to audit."""

from pathlib import Path

import pytest

import resin_models_cuda

MODEL_FAMILIES = {"deepseek_v4", "gemma4", "gpt_oss", "inkling", "qwen3_5"}
PACKAGE_BASELINE_LINES = 22_524
PACKAGE_MAX_LINES = 18_019
FORBIDDEN_CATEGORY_PACKAGES = {"drafts", "kernels", "loading", "media", "runners"}
BACKEND_NEUTRAL_SHIMS = {
    "deepseek_v4": {"codec.py", "config.py"},
    "gemma4": {"assistant.py", "codec.py", "config.py", "media.py", "plan.py"},
    "gpt_oss": {"config.py", "plan.py"},
    "inkling": {"codec.py", "config.py", "media.py"},
    "qwen3_5": {"codec.py", "config.py", "media.py"},
}
PACKAGE_ROOT = Path(resin_models_cuda.__file__).parent


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


@pytest.mark.parametrize("category", sorted(FORBIDDEN_CATEGORY_PACKAGES))
def test_cuda_package_has_no_implementation_category_directories(category: str) -> None:
    assert not (PACKAGE_ROOT / category).exists()


@pytest.mark.parametrize(
    ("family", "module"),
    [
        (family, module)
        for family, modules in BACKEND_NEUTRAL_SHIMS.items()
        for module in modules
    ],
)
def test_cuda_families_have_no_backend_neutral_compatibility_shims(
    family: str, module: str
) -> None:
    assert not (PACKAGE_ROOT / family / module).exists()
