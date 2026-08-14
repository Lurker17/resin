"""DeepSeek-V4 model support.

The CUDA runner is intentionally imported from its backend module so importing
``resin.models.deepseek_v4`` does not initialize PyTorch or TileLang.
"""

from resin.models.deepseek_v4.config import DeepSeekV4Config

__all__ = ["DeepSeekV4Config"]
