"""The synthetic GPU workload changes construction, never model execution."""

import json

import torch
from tokenizers import Tokenizer
from torch import nn

from deploy import v41_engine_smoke as smoke
from resin.models.deepseek_v41.codec import DeepSeekV41Codec


def test_engine_smoke_runs_both_natural_widths_with_checkpoint_geometry(monkeypatch):
    monkeypatch.setattr(smoke.torch.cuda, "device_count", lambda: 2)
    widths = []

    def spawn(worker, *, args, nprocs, join):
        _, output, model, checkpoint, cases, topology, options = args
        assert nprocs == 2 and join and worker is smoke._smoke_rank
        assert model == "deepseek-v41" and topology.attn_dp == topology.ep == 2
        assert not (checkpoint / "model.safetensors.index.json").exists()
        resolved, config = smoke.SyntheticV41Runner._checkpoint(checkpoint)
        assert resolved == checkpoint
        assert (config.hidden_size, config.head_dim, config.num_attention_heads) == (
            5120,
            512,
            64,
        )
        assert config.moe_intermediate_size == 2304 and not config.engram_layer_ids
        assert config.dspark_noise_token_id < config.vocab_size
        tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        prompt = DeepSeekV41Codec(tokenizer).render_ids(
            [{"role": "user", "content": "Count from 1 to 20."}]
        )
        assert len(prompt) > 8 and all(
            0 <= token < config.vocab_size for token in prompt
        )
        assert not options["balanced_experts"] and options["accepted_drafts"] is None
        assert options["validate_model"] and options["expert_backend"] == "deepgemm"
        assert (
            options["prefill_graph"] == "breakable"
            and options["decode_graph"] == "full"
        )
        widths.append(options["draft_tokens"])
        output.write_text(
            json.dumps(
                [{"ok": True, "drivers": [{"rank": i, "ok": True} for i in range(2)]}]
            )
        )

    monkeypatch.setattr(smoke.torch.multiprocessing, "spawn", spawn)
    result = smoke.run_smoke()
    assert result["ok"] and result["synthetic"] and widths == [0, 4]
    assert [case["draft_tokens"] for case in result["cases"]] == widths


def test_synthetic_weights_are_finite_nonzero_reproducible_and_shard_distinct():
    def fixture():
        model = nn.Module()
        model.register_parameter(
            "replicated", nn.Parameter(torch.empty(8, 8, dtype=torch.bfloat16))
        )
        model.norm = nn.LayerNorm(8)
        experts = nn.ModuleDict()
        model.experts = experts
        for expert in (0, 8):
            layer = nn.Module()
            layer.weight = nn.Parameter(
                torch.empty(8, 4, dtype=torch.float4_e2m1fn_x2), requires_grad=False
            )
            layer.scale = nn.Parameter(
                torch.empty(8, 1, dtype=torch.float8_e8m0fnu), requires_grad=False
            )
            layer.fp8 = nn.Parameter(
                torch.empty(8, 8, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            experts[str(expert)] = layer
        runner = object.__new__(smoke.SyntheticV41Runner)
        runner._model = model
        report = runner._load_weights()
        assert report["synthetic"] and report["bytes"] > 0
        return dict(model.named_parameters())

    first, second = fixture(), fixture()
    for name, parameter in first.items():
        assert torch.equal(parameter.view(torch.uint8), second[name].view(torch.uint8))
        if parameter.dtype != torch.float4_e2m1fn_x2:
            assert torch.isfinite(parameter.float()).all()
    assert first["replicated"].count_nonzero() > 0
    assert torch.equal(first["norm.weight"], torch.ones(8))
    assert not torch.equal(
        first["experts.0.weight"].view(torch.uint8),
        first["experts.8.weight"].view(torch.uint8),
    )
