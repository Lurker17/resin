"""GPT-OSS reference, CUDA, and MXFP4 components.

Importing this package does not import PyTorch, CUDA, or a kernel compiler.
"""

from .config import GptOssConfig
from .model import AttentionState, GptOssState, make_model_class
from .reference import (
    DEFAULT_MODEL,
    MODEL_20B,
    MODEL_120B,
    MODEL_REVISIONS,
    GptOssRunner,
)

__all__ = [
    "AttentionState",
    "DEFAULT_MODEL",
    "GptOssConfig",
    "MODEL_20B",
    "MODEL_120B",
    "MODEL_REVISIONS",
    "GptOssRunner",
    "GptOssState",
    "make_model_class",
]
