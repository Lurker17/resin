"""V4.1 construction over the shared DeepSeek paged serving runtime."""

from pathlib import Path

import torch
from tokenizers import Tokenizer

from resin.backend.topology import ParallelismCapabilities
from resin.backend.torch import use_default_dtype
from resin.models.deepseek_v41.media import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    DeepSeekV41Media,
)
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner
from resin_models_cuda.deepseek_v41.checkpoint import load_config, resolve_model_path


class DeepSeekV41CudaRunner(DeepSeekV4CudaRunner):
    """V4.1 Flash with paged KV, CED, Engram, DSpark and image embeddings."""

    parallelism_capabilities = ParallelismCapabilities(
        attn_tp=False, attn_dp=True, cp=True, ep=True, expert_tp=True
    )

    def __init__(
        self,
        *args,
        expert_backend="flashinfer",
        swa_replay_tokens=140,
        swa_replay_mode="compact",
        **kwargs,
    ):
        if type(swa_replay_tokens) is not int or swa_replay_tokens < 0:
            raise ValueError("swa_replay_tokens must be a nonnegative integer")
        if swa_replay_mode not in ("off", "mask", "compact"):
            raise ValueError("swa_replay_mode must be off, mask, or compact")
        self._swa_replay_tokens = swa_replay_tokens
        self._swa_replay_mode = swa_replay_mode
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
        self._media_features: dict[
            int, tuple[DeepSeekV41Media, dict[int, torch.Tensor]]
        ] = {}
        super().__init__(*args, **kwargs)

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._media_features.pop(request_id, None)

    def _media_embeddings(self, plan, slot, input_ids):
        media = plan.media[slot]
        if media is None:
            return None
        start = int(plan.context_lengths[slot])
        end = start + input_ids.shape[1]
        overlaps = [
            (index, image)
            for index, image in enumerate(media.images)
            if image.start < end and start < image.start + len(image.types)
        ]
        if not overlaps:
            return None
        _, features = self._media_features.setdefault(
            int(plan.sequence_ids[slot]), (media, {})
        )
        hidden = self._model.embed(input_ids)
        for index, image in overlaps:
            if index not in features:
                patches = image.patches.to(self.device)
                encoded = self._model.vision(patches, image.n_vit_h, image.n_vit_w)
                encoded = self._model.aligner(encoded, image.n_vit_h, image.n_vit_w)
                span = encoded.new_empty(len(image.types), encoded.shape[-1])
                for kind, value in (
                    (IMAGE, encoded),
                    (IMAGE_START, self._model.image_start),
                    (IMAGE_END, self._model.image_end),
                    (IMAGE_NEW_LINE, self._model.image_newline),
                ):
                    indices = [
                        i
                        for i, token_type in enumerate(image.types)
                        if token_type == kind
                    ]
                    span[indices] = value
                features[index] = span
            lower, upper = (
                max(start, image.start),
                min(end, image.start + len(image.types)),
            )
            hidden[0, lower - start : upper - start] = features[index][
                lower - image.start : upper - image.start
            ]
        return hidden

    @staticmethod
    def _checkpoint(model):
        path = resolve_model_path(model)
        return path, load_config(path)

    @staticmethod
    def _cache_backend():
        from resin_models_cuda.deepseek_v41 import cache

        return cache

    def _build_cache_geometry(self, args):
        boundary = self._model.prefill_boundary
        replay = (
            self._swa_replay_tokens
            if boundary is not None and self._swa_replay_mode != "off"
            else 0
        )
        if replay:
            if replay < args.window_size + self.speculative_tokens + 1:
                raise ValueError(
                    "SWA replay must cover the window and speculative rollback"
                )
            if (
                any(layer > boundary for layer in args.kv_source_layers)
                or any(layer >= boundary for layer in args.engram_layer_ids)
                or any(layer < boundary for layer in self._model.target_layer_ids)
            ):
                raise ValueError(
                    "SWA replay requires producer KV/Engram and consumer DSpark taps"
                )
        return self._cache_backend().CacheGeometry.build(
            args,
            prefill_chunk_size=self.prefill_chunk_size,
            max_batch_size=self.max_batch_size,
            replay_suffix_tokens=replay,
        )

    def _replay_start(self, plan, slot, tokens):
        """A host-only boundary; scored histories stay full."""
        if (
            not self._swa_replay_tokens
            or self._swa_replay_mode == "off"
            or self._model.prefill_boundary is None
            or plan.needs_prompt_logprobs
        ):
            return None
        prompt = int(plan.prefill_lengths[slot])
        if int(plan.context_lengths[slot]) + tokens > prompt:
            return None
        return max(0, prompt - self._swa_replay_tokens)

    def _prefill_region_tokens(self, plan, rows):
        boundary = self._model.prefill_boundary
        if (
            not self._swa_replay_tokens
            or boundary is None
            or self._swa_replay_mode == "off"
        ):
            return {}
        demand = 1
        if plan is not None:
            for slot, (begin, end) in enumerate(rows):
                tokens = end - begin
                lower = self._replay_start(plan, slot, tokens)
                clipped = (
                    min(max(lower - int(plan.context_lengths[slot]), 0), tokens - 1)
                    if lower is not None and self._swa_replay_mode == "compact"
                    else 0
                )
                demand = max(demand, tokens - clipped)
        return {range(boundary, len(self._model.layers)): demand}

    def _prefill_row(self, plan, slot, input_ids, positions, cache):
        context = int(plan.context_lengths[slot])
        replay_start = self._replay_start(plan, slot, input_ids.shape[1])
        logits, main = self._model.forward_prefill(
            input_ids,
            context,
            cache=cache,
            return_hidden=plan.needs_prompt_logprobs,
            replay_start=replay_start,
            compact_replay=self._swa_replay_mode == "compact",
            inputs_embeds=self._media_embeddings(plan, slot, input_ids),
        )
        if self.speculative_tokens:
            if main is None:
                raise RuntimeError("target did not produce DSpark hidden states")
            offset = input_ids.shape[1] - main.shape[1]
            valid = (
                (positions[offset:] >= replay_start)
                if replay_start is not None
                else None
            )
            self._model.commit_dspark_main_block(
                main, positions[offset : offset + 1], cache=cache, valid_tokens=valid
            )
        return logits

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
                from resin_models_cuda.deepseek_v41.attention import finish_output

                return finish_output(attention, (values, scales))

            query_outputs = (
                TensorSpec((attention.n_local_heads * attention.head_dim,)),
            )
            if hasattr(attention, "indexer"):
                query_outputs += (
                    TensorSpec(
                        (attention.indexer.local_heads * attention.indexer.head_dim,)
                    ),
                )

            def query(qr, attention=attention):
                main, index = attention.project_query(qr)
                return (main,) if index is None else (main, index)

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
                        query,
                        (TensorSpec((attention.q_lora_rank,)),),
                        query_outputs,
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

            def prefill_query(qr, invoke=query):
                values = invoke(qr.squeeze(0))
                return values[0][None], values[1][None] if len(values) > 1 else None

            layer.attn.prefill_query = prefill_query
            layer.attn.prefill_finish = lambda output, positions, invoke=finish: invoke(
                *output
            )[None]

    def _empty_prefill_regions(self):
        if not self._prefill_graph_active:
            raise RuntimeError("empty prefill regions require an aligned graph bucket")
        self._model.forward_experts_empty(prefill=True)

    def _validate_plan(self, plan):
        rows = super()._validate_plan(plan)
        image_token_id = self._model.args.image_token_id
        for slot, (begin, end) in enumerate(rows):
            media = plan.media[slot]
            tokens = plan.token_ids[begin:end]
            if media is None:
                if image_token_id in tokens:
                    raise ValueError("DeepSeek-V4.1 image tokens require image media")
                continue
            if not isinstance(media, DeepSeekV41Media):
                raise TypeError("DeepSeekV41CudaRunner media must be DeepSeekV41Media")
            if self._model.vision is None or media.image_token_id != image_token_id:
                raise ValueError(
                    "image media does not match the V4.1 vision checkpoint"
                )
            existing = self._media_features.get(int(plan.sequence_ids[slot]))
            if existing is not None:
                cached_media = existing[0]
                # Follower ranks unpickle a fresh object on each plan. Compare
                # its prepared identity without rehashing image tensors.
                if cached_media is not media and (
                    cached_media.digest != media.digest
                    or cached_media.token_ids != media.token_ids
                    or cached_media.image_token_id != media.image_token_id
                ):
                    raise ValueError(
                        "a live DeepSeek V4.1 sequence changed its media or prompt"
                    )
            start = int(plan.context_lengths[slot])
            prompt_tokens = min(len(tokens), max(0, len(media.token_ids) - start))
            if (
                tuple(tokens[:prompt_tokens])
                != media.token_ids[start : start + prompt_tokens]
            ):
                raise ValueError(
                    "scheduled token does not match DeepSeekV41Media prompt"
                )
        return rows
