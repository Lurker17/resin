# Adapted from DeepInfer, Apache-2.0; see LICENSE in this directory.
# https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/common.py
# quant
FP4_AMAX_MARGIN = 6.0 * (2**-126)
FLOAT4_E2M1_MAX = 6.0
FP8_AMAX_MARGIN = 1e-4
FLOAT8E4M3_MAX = 448.0
FLOAT8E4M3_MIN_POSITIVE = 2**-9


def is_power_of_two(x: int) -> bool:
    return x & (x - 1) == 0


def ceil_div(x: int, align: int) -> int:
    return (x + align - 1) // align


def align(x: int, align: int) -> int:
    return ceil_div(x, align) * align
