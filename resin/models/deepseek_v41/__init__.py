"""DeepSeek V4.1 checkpoint configuration, independent of accelerator libraries."""

from resin.models.deepseek_v41.codec import DeepSeekV41Codec
from resin.models.deepseek_v41.config import DeepSeekV41Config

__all__ = ["DeepSeekV41Config", "DeepSeekV41Codec"]
