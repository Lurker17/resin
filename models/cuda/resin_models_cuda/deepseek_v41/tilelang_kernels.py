"""TileKernels quantization and import compatibility for DeepSeek V4.1."""

import sys
from typing import Any

if sys.platform == "linux":
    from tilelang.backend import target

    # TileKernels 1.0 uses the path renamed in TileLang 0.1.14.
    sys.modules.setdefault("tilelang.utils.target", target)
    from tile_kernels.quant import per_token_cast
else:
    per_token_cast: Any = None
