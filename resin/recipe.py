"""Typed conversations, native model recipes, and incremental response parsers."""

from collections.abc import Sequence
from typing import Protocol

from resin._resin import (
    Audio,
    DeepSeekV4Recipe,
    DeepSeekV41Recipe,
    Gemma4Recipe,
    GptOssRecipe,
    Image,
    InklingRecipe,
    InklingResponseParser,
    Lfm2Recipe,
    Message,
    ParserEvent,
    Qwen35Recipe,
    Qwen38Recipe,
    ReasoningDelta,
    RecipeResponseParser,
    RenderedPrompt,
    RenderOptions,
    Role,
    Text,
    TextDelta,
    Thinking,
    Tool,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallStart,
    ToolResult,
    Video,
)
from resin._resin import RecipeTokenizer as Tokenizer
from resin.protocols import (
    FinalizingResponseParser,
    MediaInput,
    ResponseParser,
    TokenSpan,
)

ContentBlock = Text | Thinking | Image | Video | Audio | ToolCall | ToolResult


class Recipe(Protocol):
    """Render requests for a model and construct an incremental response parser.

    Requests are given as a typed list of messages, standardized across model
    families. A recipe describes how to translate a conversation into tokens
    and media spans, similar to a Hugging Face "chat template."

    We recommend using one of the recipes in the `resin-recipe` crate, which
    includes optimized incremental parsers in Rust.
    """

    def render(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = (),
        options: RenderOptions | None = None,
    ) -> RenderedPrompt: ...

    def create_parser(self, prompt: RenderedPrompt) -> ResponseParser: ...


__all__ = [
    "Audio",
    "ContentBlock",
    "DeepSeekV4Recipe",
    "DeepSeekV41Recipe",
    "FinalizingResponseParser",
    "Gemma4Recipe",
    "GptOssRecipe",
    "Image",
    "InklingRecipe",
    "InklingResponseParser",
    "Lfm2Recipe",
    "MediaInput",
    "Message",
    "ParserEvent",
    "Qwen35Recipe",
    "Qwen38Recipe",
    "Recipe",
    "ReasoningDelta",
    "RecipeResponseParser",
    "RenderedPrompt",
    "RenderOptions",
    "ResponseParser",
    "Role",
    "Text",
    "TextDelta",
    "Thinking",
    "Tokenizer",
    "TokenSpan",
    "Tool",
    "ToolCall",
    "ToolCallArgumentsDelta",
    "ToolCallStart",
    "ToolResult",
    "Video",
]
