"""LFM2 reference and CUDA runners.

Importing this package does not import PyTorch or initialize CUDA.
"""

from .config import Lfm2Config
from .model import FullAttentionState, Lfm2State, ShortConvState, make_model_class
from .reference import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Lfm2TorchRunner,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "FullAttentionState",
    "Lfm2Config",
    "Lfm2State",
    "Lfm2TorchRunner",
    "ShortConvState",
    "make_model_class",
]
