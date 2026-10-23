from typing import Any

# Adapted from DeepInfer, Apache-2.0; see ../LICENSE.
# https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/quant.py
import tilelang.language as _language

# TileLang interprets this module as GPU IR rather than Python tensor operations.
T: Any = _language


@T.macro
def ue8m0_exp_scale(scale: T.float32) -> T.int32:
    """Round scale up to a power of two: returns -ceil(log2(scale)), the exponent of the
    quantization multiplier 2^exp_scale (the ue8m0 dequant sf byte is -exp_scale + 127)."""
    scale_bits = T.reinterpret("uint32", scale)
    return -T.reinterpret("int32", ((scale_bits - 1) >> 23) + 1 - 127)
