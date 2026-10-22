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

    def _validate_plan(self, plan):
        if 129264 in plan.token_ids:
            raise ValueError("DeepSeek-V4.1 image inputs require the vision runner")
        return super()._validate_plan(plan)
