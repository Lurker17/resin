"""Native DeepSeek V4.1 image geometry and prepared media payloads."""

from resin._resin import (
    DeepSeekV41Image,
    DeepSeekV41ImageProcessor,
    DeepSeekV41Media,
    image_token_types,
)

IMAGE_TOKEN = "<｜deepseek_image｜>"
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

__all__ = [
    "DeepSeekV41Image",
    "DeepSeekV41ImageProcessor",
    "DeepSeekV41Media",
    "image_token_types",
]
