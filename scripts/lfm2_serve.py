"""Serve the basic LFM2 Torch runner through Resin's local HTTP APIs."""

import torch
from transformers import AutoTokenizer

from resin import Engine
from resin.recipe import Recipe
from resin.server import HttpServer
from resin_models_torch.lfm2 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Lfm2TorchRunner,
)


def main() -> None:
    torch.set_num_threads(1)

    model = DEFAULT_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model, revision=DEFAULT_REVISION)
    codec = Recipe(tokenizer, model="lfm2")
    runner = Lfm2TorchRunner(model, device="cpu")
    engine = Engine(runner, eos_token_id=runner.eos_token_id)

    print(f"Serving {model} at http://127.0.0.1:8000")
    HttpServer(engine, codec=codec, model=model).serve()


if __name__ == "__main__":
    main()
