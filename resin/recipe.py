"""Model formatting primitives backed by the ``resin-recipe`` Rust crate.

Prepare a tokenizer once per codec, then create one parser per generated
response. Parsers implement :class:`resin.protocols.ResponseParser` and emit
semantic events shared by streaming and complete API responses.
"""

from resin._resin import InklingResponseParser
from resin._resin import RecipeTokenizer as Tokenizer

__all__ = ["InklingResponseParser", "Tokenizer"]
