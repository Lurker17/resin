"""Chat with the LFM2 Torch runner using Resin's native conversation recipe."""

import torch
from tokenizers import Tokenizer

import resin
from resin import Engine
from resin.recipe import Lfm2Recipe
from resin_models_torch.lfm2 import DEFAULT_MODEL, DEFAULT_REVISION, Lfm2TorchRunner


def main() -> None:
    torch.set_num_threads(1)

    model = DEFAULT_MODEL
    tokenizer = Tokenizer.from_pretrained(model, revision=DEFAULT_REVISION)
    recipe = Lfm2Recipe(tokenizer)
    conversation = [
        resin.recipe.Message(
            resin.recipe.Role.SYSTEM,
            [resin.recipe.Text("You are a helpful assistant trained by Liquid AI.")],
        ),
        resin.recipe.Message(
            resin.recipe.Role.USER,
            [resin.recipe.Text("What is the capital of France?")],
        ),
    ]
    prompt = recipe.render(conversation)
    parser = recipe.create_parser(prompt)

    runner = Lfm2TorchRunner(model, device="cpu")
    with Engine(runner, eos_token_id=runner.eos_token_id) as engine:
        request = engine.submit(
            prompt.token_ids,
            max_new_tokens=256,
            stop_token_ids=prompt.stop_token_ids,
        )
        while engine.has_work():
            for token in engine.step():
                events = parser.feed([token.token_id])
                if token.finished:
                    reason = token.finish_reason
                    if reason == "stop":
                        events.extend(parser.finish("stop"))
                    elif reason == "length":
                        events.extend(parser.finish("length"))
                    else:
                        raise RuntimeError(f"unexpected completion reason: {reason!r}")
                for event in events:
                    if isinstance(event, resin.recipe.TextDelta):
                        print(event.delta, end="", flush=True)
        request.take_result()
    print()


if __name__ == "__main__":
    main()
