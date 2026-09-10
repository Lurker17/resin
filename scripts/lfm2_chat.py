"""Vibe-test Resin with external Hugging Face tokenization.

`tokenizers` is an optional dependency in the `dev` group. This is the most
basic example of using Resin with a Torch-based forward pass.
"""

import torch
from tokenizers import Tokenizer

from resin import Engine
from resin_models_torch.lfm2 import DEFAULT_MODEL, Lfm2TorchRunner


def main() -> None:
    torch.set_num_threads(1)

    model = DEFAULT_MODEL
    tokenizer = Tokenizer.from_pretrained(model)
    prompt = (
        "<|startoftext|><|im_start|>system\n"
        "You are a helpful assistant trained by Liquid AI.<|im_end|>\n"
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids

    runner = Lfm2TorchRunner(model, device="cpu")
    engine = Engine(runner, eos_token_id=runner.eos_token_id)
    request = engine.submit(input_ids, max_new_tokens=256)
    while engine.has_work():
        engine.step()
    output_ids = request.take_result()
    print(tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
