"""Backend-neutral DeepSeek-V4 conversation recipe."""

from typing import Any

from resin.recipe import Recipe


class DeepSeekV4Codec(Recipe):
    """Render and incrementally parse DeepSeek's reasoning and DSML format."""

    def __init__(self, tokenizer: Any) -> None:
        super().__init__(tokenizer, model="deepseek_v4")


__all__ = ["DeepSeekV4Codec"]
