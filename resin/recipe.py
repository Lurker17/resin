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
    MediaTokens,
    Message,
    ParserEvent,
    ProcessedMedia,
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
    MediaTokenSpan,
    ResponseParser,
    TokenSpan,
)

MediaBlock = Image | Video | Audio
ContentBlock = Text | Thinking | MediaBlock | ToolCall | ToolResult


class MediaProcessor(Protocol):
    """Optional preprocessing boundary for custom Python recipe implementations.

    Processing owns pixel/audio geometry; binding adds final prompt positions.
    Request state belongs to ProcessedMedia, so processors can be reused.
    Stock recipes perform these phases internally in Rust.
    """

    def process(self, media: Sequence[MediaBlock]) -> ProcessedMedia:
        """Return one token expansion per media block, in the same order."""
        ...

    def bind(
        self,
        processed: ProcessedMedia,
        token_ids: Sequence[int],
        spans: Sequence[MediaTokenSpan],
    ) -> MediaInput:
        """Bind the payload to final token ranges, ordered like the input blocks."""
        ...


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
    ) -> RenderedPrompt:
        """Prepare media and render tokens with their final media positions."""
        ...

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
    "MediaBlock",
    "MediaInput",
    "MediaProcessor",
    "MediaTokens",
    "MediaTokenSpan",
    "Message",
    "ParserEvent",
    "ProcessedMedia",
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
