"""V4.1 construction over the shared DeepSeek paged serving runtime."""

from pathlib import Path

import torch
from tokenizers import Tokenizer

from resin.backend.topology import ParallelismCapabilities
from resin.backend.torch import use_default_dtype
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner
from resin_models_cuda.deepseek_v41.checkpoint import load_config, resolve_model_path


class DeepSeekV41CudaRunner(DeepSeekV4CudaRunner):
    """Text-only V4.1 Flash; source-owned paged KV, CED, Engram and DSpark."""

    parallelism_capabilities = ParallelismCapabilities(
        attn_tp=False, attn_dp=True, cp=True, ep=True, expert_tp=True
    )

    def __init__(self, *args, expert_backend="flashinfer", **kwargs):
        if expert_backend not in ("flashinfer", "deepgemm"):
            raise ValueError(f"unknown expert backend: {expert_backend}")
        if expert_backend == "deepgemm":
            groups = kwargs.get("parallel_groups")
            if groups is None:
                raise ValueError("DeepGEMM requires explicit parallel groups")
            self.parallelism_capabilities.validate(
                groups.topology, model="DeepSeek-V4.1"
            )
            topology = groups.topology
            if (
                topology.attn_dp != topology.world_size
                or topology.ep != topology.world_size
            ):
                raise ValueError("DeepGEMM requires ADP=EP=world_size and expert_tp=1")
        self.expert_backend = expert_backend
        super().__init__(*args, **kwargs)

    @staticmethod
    def _checkpoint(model):
        path = resolve_model_path(model)
        return path, load_config(path)

    @staticmethod
    def _cache_backend():
        from resin_models_cuda.deepseek_v41 import cache

        return cache

    def _build_cache_geometry(self, args):
        return self._cache_backend().CacheGeometry.build(
            args,
            prefill_chunk_size=self.prefill_chunk_size,
            max_batch_size=self.max_batch_size,
        )

    def _build_model(self, values):
        from resin_models_cuda.deepseek_v41.model import ModelArgs, Transformer

        args = ModelArgs(
            **{
                name: value
                for name, value in values.items()
                if name in ModelArgs.__dataclass_fields__
            }
        )
        tokenizer = Tokenizer.from_file(str(Path(self.model_path) / "tokenizer.json"))
        with use_default_dtype(torch.bfloat16), torch.device(self.device):
            model = Transformer(
                args,
                tensor_parallel=self.tensor_parallel,
                parallel_groups=self.parallel_groups,
                tokenizer=tokenizer,
            )
        return args, model

    def _load_weights(self):
        from resin_models_cuda.deepseek_v41.weights import load_weights

        return load_weights(
            self.model_path,
            self._model,
            rank=self.tensor_parallel_rank,
            tensor_parallel_size=self.tensor_parallel_size,
            device=self.device,
        )

    def _prepare_model_weights(self):
        self._model.prepare_weights(
            self._allreduce.sum,
            expert_backend=self.expert_backend,
            max_tokens=max(
                self.prefill_chunk_size,
                self.max_batch_size * (self.speculative_tokens + 1),
            ),
        )

    def _configure_prefill_attention_graphs(self):
        from resin.backend.cuda.tensor_graphs import CudaGraphTensorRegion, TensorSpec

        regions = []
        for layer in self._model.layers:
            attention = layer.attn
            width = (
                attention.n_local_heads // attention.n_local_groups * attention.head_dim
            )

            def project(hidden, attention=attention):
                return tuple(
                    value.squeeze(0) for value in attention._project(hidden[None])
                )

            def finish(values, scales, attention=attention):
                return attention.wo_b(attention.wo_a(values, scales).flatten(1))

            regions.extend(
                (
                    CudaGraphTensorRegion(
                        f"layer.{layer.layer_id}.project",
                        project,
                        (TensorSpec((attention.dim,)),),
                        (
                            TensorSpec((attention.q_lora_rank,)),
                            TensorSpec((attention.head_dim,)),
                        ),
                    ),
                    CudaGraphTensorRegion(
                        f"layer.{layer.layer_id}.query",
                        attention.wq_b,
                        (TensorSpec((attention.q_lora_rank,)),),
                        TensorSpec((attention.n_local_heads * attention.head_dim,)),
                    ),
                    CudaGraphTensorRegion(
                        f"layer.{layer.layer_id}.finish",
                        finish,
                        (
                            TensorSpec(
                                (attention.n_local_groups, width), torch.float8_e4m3fn
                            ),
                            TensorSpec(
                                (attention.n_local_groups, width // 128), torch.int32
                            ),
                        ),
                        TensorSpec((attention.dim,)),
                    ),
                )
            )
        callbacks = self._configure_prefill_tensor_graphs(
            tuple(regions),
            max_tokens=self.prefill_chunk_size,
            pad_up_to=0,
            capture_context=self._prefill_compute_context,
        )
        for index, layer in enumerate(self._model.layers):
            project, query, finish = callbacks[3 * index : 3 * index + 3]
            layer.attn.prefill_project = lambda hidden, invoke=project: tuple(
                value[None] for value in invoke(hidden.squeeze(0))
            )
            layer.attn.prefill_query = lambda qr, invoke=query: invoke(qr.squeeze(0))[
                None
            ]
            layer.attn.prefill_finish = lambda output, positions, invoke=finish: invoke(
                *output
            )[None]

    def _empty_prefill_regions(self):
        if not self._prefill_graph_active:
            raise RuntimeError("empty prefill regions require an aligned graph bucket")
        self._model.forward_experts_empty(prefill=True)

    def _validate_plan(self, plan):
        if 129264 in plan.token_ids:
            raise ValueError("DeepSeek-V4.1 image inputs require the vision runner")
        return super()._validate_plan(plan)
