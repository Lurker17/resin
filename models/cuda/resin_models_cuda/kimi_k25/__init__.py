"""Kimi K2.5, K2.6, and K2.7-Code support for CUDA."""

from resin_models_cuda.kimi_k25.config import (
    EagleVersion,
    KimiK25Config,
    KimiK25DraftConfig,
)

__all__ = [
    "EagleVersion",
    "KimiK25Config",
    "KimiK25CudaRunner",
    "KimiK25DraftConfig",
]


def __getattr__(name: str):
    if name == "KimiK25CudaRunner":
        from resin_models_cuda.kimi_k25.cuda import KimiK25CudaRunner

        return KimiK25CudaRunner
    raise AttributeError(name)
