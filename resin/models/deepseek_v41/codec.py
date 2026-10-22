"""The 1017 V4.1 recipe, distinct from the earlier V4 checkpoint grammar."""

from typing import Any

from resin.recipe import Recipe


class DeepSeekV41Codec(Recipe):
    def __init__(self, tokenizer: Any) -> None:
        super().__init__(tokenizer, model="deepseek_v41")


__all__ = ["DeepSeekV41Codec"]
