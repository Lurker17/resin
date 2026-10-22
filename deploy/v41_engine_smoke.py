"""Synthetic EP2/ADP2 smoke through the actual V4.1 runner, pages and Engine.

This exercises integration and graph/scoring parity, not checkpoint quality.
Only weight construction changes; all execution uses production model paths.
"""

from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
from typing import Any
from zlib import crc32

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from deploy.bench import _rank
from resin import Parallelism
from resin.models.deepseek_v41.config import DeepSeekV41Config
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner


def settings():
    args: dict[str, Any] = dict(
        vocab_size=1024,
        dim=5120,
        moe_inter_dim=2304,
        n_layers=3,
        n_mtp_layers=1,
        n_heads=64,
        n_routed_experts=16,
        n_activated_experts=6,
        n_shared_experts=1,
        dspark_n_routed_experts=16,
        dspark_n_activated_experts=3,
        dspark_block_size=5,
        dspark_noise_token_id=999,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=256,
        q_lora_rank=1280,
        head_dim=512,
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,
        norm_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10,
        dtype="fp8",
        expert_dtype="fp4",
        compress_ratios=[2, 2, 2, 0],
        kv_source_layers=[0],
        index_source_layers=[0],
        index_n_heads=8,
        index_head_dim=128,
        index_topk=8,
        candidate_source_layer=0,
        candidate_topk_blocks=4,
        candidate_block_size=8,
        engram_layer_ids=[],
        engram_num_embeddings=[],
    )
    names = dict(
        hidden_size="dim",
        num_hidden_layers="n_layers",
        num_attention_heads="n_heads",
        sliding_window="window_size",
        num_routed_experts="n_routed_experts",
        num_experts_per_token="n_activated_experts",
        moe_intermediate_size="moe_inter_dim",
    )
    args.update(engram_max_ngram_size=4, engram_n_heads=8, engram_head_dim=256)
    values = {**args, "num_key_value_heads": 1}
    data: dict[str, Any] = {
        field: tuple(value) if isinstance(value, list) else value
        for field in DeepSeekV41Config.__dataclass_fields__
        for value in [values[names.get(field, field)]]
    }
    config = DeepSeekV41Config(**data)
    config.validate()
    renamed = dict(
        rope_head_dim="qk_rope_head_dim",
        kv_source_layers="kv_source_layer_ids",
        index_source_layers="index_source_layer_ids",
        num_routed_experts="n_routed_experts",
        num_experts_per_token="num_experts_per_tok",
        candidate_source_layer="candidate_source_layer_id",
    )
    public = dict(
        model_type="deepseek_v41",
        quantization_config=dict(
            weight_block_size=[32, 32], scale_fmt="ue8m0", expert_dtype="fp4"
        ),
        text_config={
            renamed.get(name, name): getattr(config, name)
            for name in config.__dataclass_fields__
        },
    )
    return args, public


class SyntheticV41Runner(DeepSeekV41CudaRunner):
    @staticmethod
    def _checkpoint(model):
        # Weights are generated below; published-shard readiness is inapplicable.
        path = Path(model)
        config = DeepSeekV41Config.from_hugging_face(
            json.loads((path / "config.json").read_text())
        )
        return path, config

    @torch.no_grad()
    def _load_weights(self):
        total = 0
        for name, value in self._model.named_parameters():
            # Global expert names differ between shards; replicated names match.
            generator = torch.Generator(device=value.device).manual_seed(
                4141 + crc32(name.encode())
            )
            if value.dtype == torch.float4_e2m1fn_x2:
                value.view(torch.uint8).random_(0, 256, generator=generator)
            elif value.dtype == torch.float8_e8m0fnu:
                value.fill_(2**-6)
            elif name.endswith("norm.weight"):
                value.fill_(1)
            elif name.endswith(("_base", ".bias")):
                value.zero_()
            elif name.endswith("_scale"):
                value.fill_(0.1)
            else:
                value.copy_(
                    torch.randn(value.shape, device=value.device, generator=generator)
                    * 0.01
                )
            total += value.numel() * value.element_size()
        return {"synthetic": True, "seed": 4141, "bytes": total}


def _runner(model, checkpoint, *, device, groups, cases, options, context_capacity):
    return SyntheticV41Runner(
        checkpoint,
        device=device,
        parallel_groups=groups,
        expert_backend="deepgemm",
        max_seq_len=context_capacity,
        max_batch_size=1,
        speculative_tokens=options["draft_tokens"],
        prefill_chunk_size=8,
        cuda_graph_prefill="breakable",
        cuda_graph_decode="full",
        cuda_graph_decode_shapes=((1, context_capacity),),
        cache_parent_capacity=32,
        fast_allreduce=False,
    )


def _smoke_rank(rank, *args):
    _rank(rank, *args, runner_factory=_runner)


def run_smoke():
    if torch.cuda.device_count() != 2:
        raise ValueError("the synthetic V4.1 Engine smoke requires two visible GPUs")
    args, public = settings()
    topology = Parallelism(world_size=2, attn_dp=2, ep=2)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="v41-engine-smoke-") as directory:
        checkpoint = Path(directory)
        (checkpoint / "inference").mkdir()
        (checkpoint / "inference/config.json").write_text(json.dumps(args))
        (checkpoint / "config.json").write_text(json.dumps(public))
        markers = [
            "<unk>",
            "<｜end▁of▁sentence｜>",
            "<｜begin▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜System｜>",
            "<think>",
            "</think>",
        ]
        vocab = {
            token: index
            for index, token in enumerate(
                markers + [f"token{i}" for i in range(len(markers), 1024)]
            )
        }
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(markers)
        tokenizer.save(str(checkpoint / "tokenizer.json"))
        for width in (0, 4):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            output = checkpoint / f"k{width}.json"
            options = dict(
                draft_tokens=width,
                accepted_drafts=None,
                balanced_experts=False,
                warmup_steps=0,
                measure_steps=4,
                prefill_chunk_size=8,
                prefill_graph="breakable",
                decode_graph="full",
                gpu_memory_utilization=0.9,
                validate_model=True,
                expert_backend="deepgemm",
                metadata={"synthetic": True, "seed": 4141, "engram": False},
            )
            torch.multiprocessing.spawn(
                _smoke_rank,
                args=(
                    port,
                    output,
                    "deepseek-v41",
                    checkpoint,
                    [dict(phase="decode", context_length=64, batch_size=1)],
                    topology,
                    options,
                ),
                nprocs=2,
                join=True,
            )
            results.append(
                {"draft_tokens": width, "reports": json.loads(output.read_text())}
            )
    return {
        "ok": all(record["ok"] for case in results for record in case["reports"]),
        "synthetic": True,
        "geometry": args,
        "parallelism": dict(world_size=2, attn_dp=2, attn_tp=1, cp=1, ep=2),
        "cases": results,
    }


if __name__ == "__main__":
    import sys

    report = run_smoke()
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["ok"]:
        raise SystemExit("synthetic V4.1 Engine validation failed")
