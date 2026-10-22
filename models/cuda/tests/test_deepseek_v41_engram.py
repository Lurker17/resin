"""CPU checks against scalar n-gram math, independent of CUDA kernels."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_deepseek_v41_model import implementation as implementation
from tokenizers import Tokenizer, decoders, models
from torch import nn

from resin_models_cuda.deepseek_v41.engram import (
    Engram,
    EngramLayout,
    NgramHash,
    ParallelEngramEmbedding,
    build_compressed_token_map,
    compute_hash_multipliers,
)


def tokenizer():
    return Tokenizer(
        models.WordLevel(
            dict(
                zip(
                    (
                        "[UNK]",
                        "<pad>",
                        "The",
                        "the",
                        "ＴＨＥ",
                        "Thé",
                        " ",
                        "",
                        "x",
                        "y",
                        "z",
                    ),
                    range(11),
                    strict=True,
                )
            ),
            unk_token="[UNK]",
        )
    )


def args(**changes):
    result = dict(
        engram_layer_ids=(1, 14),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_vocab_size=7,
        engram_num_embeddings=(400, 600),
        engram_compressed_vocab_size=8,
        engram_pad_id=1,
        dim=4,
        hc_mult=2,
        norm_eps=1e-5,
    )
    result.update(changes)
    return SimpleNamespace(**result)


class PagedTokens:
    """Two requests with deliberately unrelated physical page orders."""

    def __init__(self):
        self.pages = torch.tensor([[5, 1, 7, 2], [6, 4, 0, 3]])
        self.token_values = torch.full((32,), -999, dtype=torch.int64)

    def token_rows(self, positions):
        pages = self.pages.gather(1, (positions // 4).flatten(1)).view_as(positions)
        return pages * 4 + positions % 4


def scalar_hash(module, sequences, positions, dead=None):
    mapping, multipliers = module.token_map.tolist(), module.multipliers.tolist()
    result = []
    for batch, row in enumerate(positions.tolist()):
        output = []
        for position in row:
            tokens, blocked = [], False
            for shift in range(module.layout.max_ngram_size):
                source = position - shift
                blocked |= source < 0 or (dead is not None and dead[batch][source])
                tokens.append(
                    module.pad_id if blocked else mapping[sequences[batch][source]]
                )
            layers = []
            for layer, primes in enumerate(module.layout.primes):
                rolling, columns, offset = tokens[0] * multipliers[layer][0], [], 0
                for shift, buckets in enumerate(primes, 1):
                    rolling ^= tokens[shift] * multipliers[layer][shift]
                    for modulus in buckets:
                        columns.append(rolling % modulus + offset)
                        offset += modulus
                layers.append(columns)
            output.append(layers)
        result.append(output)
    return torch.tensor(result)


def test_normalization_collapses_case_accents_width_but_keeps_space_and_empty():
    mapping, count = build_compressed_token_map(tokenizer())
    assert mapping[2:6] == [2] * 4
    assert mapping[6] != mapping[7]
    assert count == 8


def test_partial_utf8_tokens_are_keyed_by_raw_form():
    backend = Tokenizer(
        models.BPE(
            {"[UNK]": 0, "<0xC3>": 1, "<0xA9>": 2, "é": 3, "e": 4},
            [],
            unk_token="[UNK]",
        )
    )
    backend.decoder = decoders.ByteFallback()
    mapping, count = build_compressed_token_map(backend)
    assert backend.decode([1]) == backend.decode([2]) == "\ufffd"
    assert mapping[1] != mapping[2]
    assert mapping[3] == mapping[4]
    assert count == 4


def test_prime_buckets_are_globally_disjoint_and_follow_official_order():
    layout = EngramLayout.from_args(args())
    assert layout.primes == (
        ((7, 11), (13, 17), (19, 23)),
        ((29, 31), (37, 41), (43, 47)),
    )
    released = EngramLayout.from_args(
        args(
            engram_vocab_size=16000000,
            engram_n_heads=8,
            engram_num_embeddings=(384006168, 384016682),
        )
    )
    assert [sum(sum(row) for row in layer) for layer in released.primes] == [
        384006168,
        384016682,
    ]


def test_multiplier_rng_matches_official_per_layer_numpy_seed():
    actual = compute_hash_multipliers((1, 14), 4, 99092)
    bound = (np.iinfo(np.int64).max // 99092) // 2
    expected = np.stack(
        [
            np.random.default_rng(10007 * layer).integers(0, bound, 4) * 2 + 1
            for layer in (1, 14)
        ]
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert (actual % 2 == 1).all()
    assert int(actual.max()) * 99092 < 2**63


@pytest.mark.parametrize("chunk", [1, 3, 5, 12])
def test_paged_prefill_and_decode_match_scalar_history_for_each_request(chunk):
    module, cache = NgramHash(args(), tokenizer()), PagedTokens()
    sequences = [
        [2, 8, 9, 10, 3, 9, 8, 10, 2, 8, 9, 10],
        [10, 9, 8, 2, 10, 8, 9, 3, 8, 2, 10, 9],
    ]
    ids = torch.tensor(sequences)
    outputs = []
    for start in range(0, ids.shape[1], chunk):
        positions = torch.arange(start, min(start + chunk, ids.shape[1])).expand(2, -1)
        result = module(ids[:, start : start + chunk], positions, cache)
        torch.testing.assert_close(result, scalar_hash(module, sequences, positions))
        outputs.append(result)
    assert torch.cat(outputs, 1).shape == (2, 12, 2, 6)
    logical = cache.token_rows(torch.arange(12).expand(2, -1))
    torch.testing.assert_close(cache.token_values[logical], module.token_map[ids])
    assert set(module.state_dict()) == set()  # no hidden cursor or history checkpoint


def test_rejected_draft_tail_is_overwritten_without_mutating_accepted_history():
    module, cache = NgramHash(args(), tokenizer()), PagedTokens()
    original = torch.tensor(
        [[2, 8, 9, 10, 2, 9, 8, 10, 2, 9], [9, 2, 8, 10, 3, 10, 9, 8, 2, 10]]
    )
    module(original, torch.arange(10).expand(2, -1), cache)
    accepted = cache.token_values[
        cache.token_rows(torch.arange(7).expand(2, -1))
    ].clone()
    replacement = torch.tensor([[10, 8, 2], [8, 10, 9]])
    positions = torch.arange(7, 10).expand(2, -1)
    actual = module(replacement, positions, cache)
    history = torch.cat((original[:, :7], replacement), 1)
    torch.testing.assert_close(actual, scalar_hash(module, history.tolist(), positions))
    torch.testing.assert_close(
        cache.token_values[cache.token_rows(torch.arange(7).expand(2, -1))], accepted
    )


def test_dead_token_blocks_lookback_across_chunk_boundary():
    module, cache = NgramHash(args(), tokenizer()), PagedTokens()
    ids = torch.tensor([[2, 8, 9, 10, 2, 9], [9, 2, 8, 10, 3, 9]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 2] = False
    module(ids[:, :3], torch.arange(3).expand(2, -1), cache, mask[:, :3])
    positions = torch.arange(3, 6).expand(2, -1)
    actual = module(ids[:, 3:], positions, cache, mask[:, 3:])
    torch.testing.assert_close(
        actual, scalar_hash(module, ids.tolist(), positions, (~mask).tolist())
    )


def test_compressed_vocabulary_mismatch_is_rejected_before_inference():
    with pytest.raises(ValueError, match="compressed vocabulary"):
        NgramHash(args(engram_compressed_vocab_size=9), tokenizer())


def test_row_shards_keep_fp8_storage_and_mask_other_ranks():
    class LocalLookup:
        def __init__(self, rank):
            self.group = SimpleNamespace(size=2, local_rank=rank)

        def __call__(self, indices, local):
            return local(indices)

    shards = []
    for rank in range(2):
        table = ParallelEngramEmbedding(5, 32, LocalLookup(rank))
        assert table.weight.shape == (3, 32)
        assert table.weight.dtype == torch.float8_e4m3fn
        assert table.scale.dtype == torch.float8_e8m0fnu
        with torch.no_grad():
            table.weight.fill_(rank + 1)
            table.scale.fill_(2)
        shards.append(table(torch.arange(5)))
    torch.testing.assert_close(
        sum(shards),
        torch.tensor([2, 2, 2, 4, 4], dtype=torch.bfloat16)[:, None].expand(-1, 32),
    )


def test_engram_gate_normalizes_each_hc_copy_and_preserves_mask(monkeypatch):
    import resin_models_cuda.deepseek_v41.linear as linear

    def projection(inputs, outputs):
        return nn.Linear(inputs, outputs, bias=False)

    monkeypatch.setattr(linear, "Linear", projection)
    module = Engram(args(), 14)
    assert module.layer_hash_index == 1
    monkeypatch.setattr(module, "embed", nn.Embedding(600, 32))
    generator = torch.Generator().manual_seed(13)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    hidden = torch.randn(2, 3, 2, 4, generator=generator)
    hashes = torch.randint(0, 600, (2, 3, 6), generator=generator)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    key, value = module.wkv(module.embed(hashes).flatten(-2)).split([8, 4], -1)
    key = key.reshape(2, 3, 2, 4)
    expected = hidden.clone()
    for batch in range(2):
        for position in range(3):
            for hc in range(2):
                if not mask[batch, position]:
                    continue
                h, k = hidden[batch, position, hc], key[batch, position, hc]
                dot = (h * (module.q_weight[hc] * module.k_weight[hc]) * k).sum()
                dot *= (
                    (h.square().mean() + module.eps).rsqrt()
                    * (k.square().mean() + module.eps).rsqrt()
                    / 2
                )
                gate = torch.sigmoid(
                    torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
                )
                expected[batch, position, hc] += gate * value[batch, position]
    torch.testing.assert_close(module(hidden, hashes, mask), expected)


@pytest.mark.parametrize("prefill", [False, True])
def test_idle_rank_joins_engram_before_each_layer_collective(
    implementation, prefill, monkeypatch
):
    Transformer = implementation.Transformer
    calls = []

    class Layer(nn.Module):
        def __init__(self, index, *, engram=False):
            super().__init__()
            self.engram = (
                SimpleNamespace(
                    hash_columns=24,
                    embed=SimpleNamespace(
                        forward_empty=lambda columns: calls.append(
                            ("engram", index, columns)
                        )
                    ),
                )
                if engram
                else None
            )
            self.ffn = SimpleNamespace(
                forward_empty=lambda: calls.append(("ffn", index))
            )

            def empty_ffn(hidden, ids):
                assert hidden.shape == (0, 8)
                assert hidden.dtype == torch.bfloat16
                assert ids.shape == (0,) and ids.dtype == torch.int64
                calls.append(("ffn", index))

            self.prefill_ffn = empty_ffn

    model = Transformer.__new__(Transformer)
    nn.Module.__init__(model)
    monkeypatch.setattr(
        model,
        "layers",
        nn.ModuleList([Layer(0), Layer(1, engram=True), Layer(2, engram=True)]),
        raising=False,
    )
    monkeypatch.setattr(model, "mtp", nn.ModuleList([Layer(3)]), raising=False)
    monkeypatch.setattr(model, "args", SimpleNamespace(dim=8), raising=False)
    monkeypatch.setattr(
        model, "embed", nn.Embedding(4, 8, dtype=torch.bfloat16), raising=False
    )
    Transformer.forward_experts_empty(model, prefill=prefill)
    assert calls == [
        ("ffn", 0),
        ("engram", 1, 24),
        ("ffn", 1),
        ("engram", 2, 24),
        ("ffn", 2),
    ]
    calls.clear()
    Transformer.forward_experts_empty(model, stage="draft")
    assert calls == [("ffn", 3)]
