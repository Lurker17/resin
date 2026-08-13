"""Vibe-test Resin with external Hugging Face tokenization.

`tokenizers` is an optional dependency in the `demo` group. This is the most
basic example of using Resin with a Torch-based forward pass.
"""

from tokenizers import Tokenizer

from resin import Engine
from resin.backends.lfm2 import DEFAULT_MODEL, Lfm2TorchRunner


def main() -> None:
    tokenizer = Tokenizer.from_pretrained("LiquidAI/LFM2.5-230M")
    prompt = (
        "<|startoftext|><|im_start|>system\n"
        "You are a helpful assistant trained by Liquid AI.<|im_end|>\n"
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids

    runner = Lfm2TorchRunner(DEFAULT_MODEL)
    engine = Engine(runner, eos_token_id=runner.eos_token_id)
    request = engine.submit(input_ids, max_new_tokens=256)
    while engine.has_work():
        engine.step()
    output_ids = request.take_result()
    print(tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
