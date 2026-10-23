"""Exact runtime-width candidate reduction and mutable graph qualification."""

import torch


def _reference(logits, lengths, block_size):
    """Independent row-wise oracle for the initialized live score range."""
    blocks = (logits.shape[1] + block_size - 1) // block_size
    scores = torch.full((len(lengths), blocks), -torch.inf, dtype=torch.bfloat16)
    counts = torch.empty_like(lengths, device="cpu")
    source = logits.cpu()
    for row, length in enumerate(lengths.cpu().tolist()):
        count = (length + block_size - 1) // block_size
        counts[row] = count
        if count:
            if count > 1:
                scores[row, : count - 1] = (
                    source[row, : (count - 1) * block_size]
                    .reshape(-1, block_size)
                    .amax(-1)
                )
            scores[row, count - 1] = torch.inf
    return scores, counts


def _equal(output, expected):
    scores, counts = (value.cpu() for value in output)
    expected_scores, expected_counts = expected
    assert torch.equal(counts, expected_counts)
    valid = torch.arange(expected_scores.shape[1])[None] < counts[:, None]
    assert torch.equal(
        scores[:, : expected_scores.shape[1]][valid], expected_scores[valid]
    )


@torch.inference_mode()
def run_probe():
    from resin_models_cuda.deepseek_v41.upstream.candidate_blocks import (
        candidate_block_max,
    )
    from resin_models_cuda.deepseek_v41.upstream.cuda.candidate_blocks import (
        candidate_block_max_tl,
    )

    torch.manual_seed(583)
    rows = 257  # Odd row and a second preparation CTA (128 request pairs/CTA).
    results = []
    for block_size in (8, 16):
        kernel = candidate_block_max_tl(block_size)
        for width in (17, 16383, 16384, 16385, 65539, 131073):
            # Contiguous columns with a padded row stride exercise runtime shape
            # and stride inference independently, including partial 16K tasks.
            stride = (width + 15) // 16 * 16
            logits = torch.randn(rows, stride, device="cuda", dtype=torch.bfloat16)[
                :, :width
            ]
            original = logits.clone()
            lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
            pattern = torch.tensor(
                [
                    0,
                    1,
                    block_size - 1,
                    block_size,
                    block_size + 1,
                    16383,
                    16384,
                    16385,
                    width - 1,
                    width,
                ],
                device="cuda",
                dtype=torch.int32,
            ).clamp_max_(width)

            def update(iteration):
                lengths.copy_(
                    pattern[
                        (torch.arange(rows, device="cuda") + iteration) % len(pattern)
                    ]
                )
                # Future keys deliberately outrank valid keys. Live counts must
                # exclude their blocks; the newest partial block remains +inf.
                logits.copy_(original)
                logits.masked_fill_(
                    torch.arange(width, device="cuda")[None] >= lengths[:, None], 1000
                )

            def call():
                return candidate_block_max(logits, lengths, block_size)

            update(0)
            _equal(call(), _reference(logits, lengths, block_size))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call()
                call()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = call()
            torch.cuda.current_stream().wait_stream(stream)
            for iteration in range(3):
                update(iteration + 1)
                graph.replay()
                torch.cuda.synchronize()
                _equal(output, _reference(logits, lengths, block_size))
            # Exercise an empty work queue after nonempty graph replays.
            lengths.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert (output[1] == 0).all()
            assert candidate_block_max_tl(block_size) is kernel
            results.append(
                dict(
                    block_size=block_size,
                    width=width,
                    rows=rows,
                    exact_live_scores=True,
                    graph_replays=4,
                )
            )
    return dict(
        hardware=torch.cuda.get_device_name(),
        same_compiled_factories=True,
        cases=results,
    )
