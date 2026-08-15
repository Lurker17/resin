"""Paged FA4/FlashInfer execution for the Resin-owned PyTorch LFM2 model.

This deliberately specializes one model and one CUDA stack. Rust supplies the
complete immutable page, append, prefill, and decode subplans. This runner only
tensorizes that metadata and addresses the corresponding physical K/V slab while
mapping short-convolution state into one native-StatePool-owned physical slab.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from resin_models_torch.lfm2.reference import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    load_model,
    resolve_model_path,
)

from resin import BatchPhase, CacheGroupLayout, CacheLayout, StateMappingConfig
from resin.backend.checks import nonempty_plan
from resin.backend.checks import token_ids as check_token_ids
from resin.backend.cuda.attention import paged_fa4
from resin.backend.cuda.execution import CudaCompletion, CudaExecutor
from resin.backend.cuda.mapped_state import MappedStateArena
from resin.backend.cuda.paged_kv import PagedKvArena, PagedKvCsr
from resin.backend.cuda.sampling import FlashInferSampler
from resin.backend.cuda.staging import TensorStager
from resin_models_cuda.lfm2.plan import CacheMetadata, build_cache_metadata

if TYPE_CHECKING:
    from resin import BatchPlan


_RECURRENT_STATE_MAPPING_PROTOCOL = "lfm2.conv_state.v1"


class _InPlaceRecurrentState:
    """Recurrent-state facade that cannot detach from its mapped slab view."""

    __slots__ = ("_state",)

    def __init__(self, state) -> None:
        self._state = state

    @property
    def value(self):
        return self._state

    @value.setter
    def value(self, value) -> None:
        if value is None:
            raise ValueError("mapped convolution state cannot be cleared")
        self._state.copy_(value)


class Lfm2CudaRunner:
    """BF16 LFM2 runner with FA4 prefill and FlashInfer paged decode."""

    supports_continuation_prefill = True
    supports_recurrent_state_mapping = True
    recurrent_state_mapping = StateMappingConfig(
        _RECURRENT_STATE_MAPPING_PROTOCOL,
        ["write"],
    )

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        page_size: int = 16,
        max_pages: int = 256,
        workspace_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        import flashinfer
        from flash_attn.cute import flash_attn_varlen_func

        if not torch.cuda.is_available():
            raise RuntimeError("Lfm2CudaRunner requires CUDA")
        if torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("FlashAttention-4 requires an SM90-or-newer GPU")
        if page_size <= 0 or max_pages <= 0:
            raise ValueError("page_size and max_pages must be positive")

        self._torch = torch
        self._flashinfer = flashinfer
        self._fa4 = flash_attn_varlen_func
        self._sampler = FlashInferSampler(torch)
        self.device = torch.device("cuda")
        self._stream = torch.cuda.Stream(device=self.device)
        self.page_size = page_size
        self.max_pages = max_pages
        self.cache_layout = CacheLayout(
            max_pages,
            [CacheGroupLayout(page_size)],
        )
        self.model_path = resolve_model_path(model, revision)
        # Persistent CUDA state is created on the same private stream used for
        # every launch, so dispatch never needs an implicit default-stream handoff.
        with torch.cuda.stream(self._stream):
            self._torch, self._model, self.config = load_model(
                self.model_path, device=self.device
            )
            self.eos_token_id = self.config.get("eos_token_id")
            self.vocab_size = self.config["vocab_size"]
            args = self._model.args
            self.num_query_heads = args.num_attention_heads
            self.num_kv_heads = args.num_key_value_heads
            self.head_dim = args.hidden_size // args.num_attention_heads
            self.scale = self.head_dim**-0.5
            self._attention_layers = [
                index
                for index, layer_type in enumerate(args.layer_types)
                if layer_type == "full_attention"
            ]
            self._attention_slot = {
                layer_index: slot
                for slot, layer_index in enumerate(self._attention_layers)
            }
            self._conv_layers = [
                index
                for index, layer_type in enumerate(args.layer_types)
                if layer_type != "full_attention"
            ]
            self._conv_slot = {
                layer_index: slot for slot, layer_index in enumerate(self._conv_layers)
            }
            self.hidden_size = args.hidden_size
            self.conv_cache_length = args.conv_cache_length

            # FA4's SM90 paged path may issue TMA loads for the full final
            # physical page and mask scores afterwards. Zero the unwritten tail
            # so a masked lane can never feed NaN/garbage into the PV reduction.
            self._kv = PagedKvArena(
                torch,
                self.device,
                self.cache_layout,
                layer_count=len(self._attention_layers),
                page_size=page_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=torch.bfloat16,
                resource="LFM2 KV",
            )
            self._decode_workspace = torch.zeros(
                workspace_bytes, dtype=torch.uint8, device=self.device
            )
            self._decode = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                self._decode_workspace, "NHD"
            )
        self._state = MappedStateArena(
            torch,
            self.device,
            self._stream,
            protocol=_RECURRENT_STATE_MAPPING_PROTOCOL,
            layer_indices=self._conv_layers,
            row_shape=(self.hidden_size, self.conv_cache_length - 1),
            dtype=torch.bfloat16,
            label="LFM2 recurrent state",
        )
        self._executor = CudaExecutor(
            torch,
            self.device,
            self._stream,
            resources=self._completion_resources,
        )

    def bind_cache_and_recurrent_state(
        self,
        cache_pool_id: int,
        layout: CacheLayout,
        state_pool_id: int,
        state_capacity: int,
        protocol: str,
        required_arena_id: int | None,
    ) -> tuple[int, int, int, str, int]:
        """Atomically bind the cache and recurrent-state physical arenas."""

        cache_id = self._kv.binding.validate(cache_pool_id, layout)
        state_binding, prepared, arena_id = self._state.prepare_binding(
            state_pool_id,
            state_capacity,
            protocol,
            required_arena_id,
        )
        self._state.commit_binding(state_binding, prepared)
        self._kv.binding.commit(cache_id)
        return (
            cache_id,
            state_pool_id,
            state_capacity,
            protocol,
            arena_id,
        )

    def _preflight_cache(self, plan: BatchPlan) -> CacheMetadata:
        """Validate and derive the complete cache launch before CUDA acceptance."""

        self._kv.check(plan)
        groups = tuple(plan.cache_groups)
        if len(groups) != 1 or groups[0].group_index != 0:
            raise ValueError("BatchPlan does not contain LFM2 cache group zero")
        derived = build_cache_metadata(plan, self.page_size)
        self._kv.check_pages(derived.block_ids)
        return derived

    def _metadata(
        self,
        plan: BatchPlan,
        derived: CacheMetadata | None = None,
        staging: TensorStager | None = None,
    ):
        torch = self._torch
        if derived is None:
            derived = self._preflight_cache(plan)
        prefill = derived.prefill

        if staging is None:
            staging = TensorStager(torch, self.device)
        metadata = {"_pinned": staging.host_tensors, "_staging": staging}
        metadata.update(
            staging.tensors(
                {
                    "block_indptr": (derived.block_indptr, torch.int32),
                    "block_ids": (derived.block_ids, torch.int32),
                    "last_block_lengths": (derived.last_block_lengths, torch.int32),
                    "append_batch_indices": (
                        derived.append_batch_indices,
                        torch.int32,
                    ),
                    "append_positions": (derived.append_positions, torch.int32),
                    "prefill_token_indices": (
                        prefill.dense_token_indices,
                        torch.long,
                    ),
                    "prefill_cu_seqlens": (
                        prefill.dense_cu_seqlens,
                        torch.int32,
                    ),
                    "decode_token_indices": (
                        derived.decode_token_indices,
                        torch.long,
                    ),
                },
                device=True,
            )
        )
        metadata["prefill_max_q_len"] = prefill.dense_max_q_len

        if prefill.continuation_max_q_len:
            continuation_page_table = staging.device(
                "continuation_prefill_page_table",
                prefill.continuation_page_table,
                torch.int32,
            ).view(
                len(prefill.continuation_seqused_k),
                prefill.continuation_page_table_width,
            )
            metadata.update(
                {
                    "continuation_prefill_token_indices": staging.device(
                        "continuation_prefill_token_indices",
                        prefill.continuation_token_indices,
                        torch.long,
                    ),
                    "continuation_prefill_cu_seqlens": staging.device(
                        "continuation_prefill_cu_seqlens",
                        prefill.continuation_cu_seqlens,
                        torch.int32,
                    ),
                    "continuation_prefill_page_table": continuation_page_table,
                    "continuation_prefill_seqused_k": staging.device(
                        "continuation_prefill_seqused_k",
                        prefill.continuation_seqused_k,
                        torch.int32,
                    ),
                    "continuation_prefill_max_q_len": (prefill.continuation_max_q_len),
                    "continuation_prefill_min_seqlen_k": (
                        prefill.continuation_min_seqlen_k
                    ),
                    "continuation_prefill_max_seqlen_k": (
                        prefill.continuation_max_seqlen_k
                    ),
                }
            )
        else:
            metadata["continuation_prefill_max_q_len"] = 0

        metadata.update(
            staging.tensors(
                {
                    "decode_block_indptr": (
                        derived.decode_block_indptr,
                        torch.int32,
                    ),
                    "decode_block_ids": (derived.decode_block_ids, torch.int32),
                    "decode_last_block_lengths": (
                        derived.decode_last_block_lengths,
                        torch.int32,
                    ),
                },
                device=False,
            )
        )
        metadata["paged_kv"] = PagedKvCsr(
            metadata["block_indptr"],
            metadata["block_ids"],
            metadata["last_block_lengths"],
            metadata["append_batch_indices"],
            metadata["append_positions"],
        )
        return metadata

    def _plan_decode(self, plan: BatchPlan, metadata) -> bool:
        torch = self._torch
        if plan.phase != BatchPhase.DECODE:
            return False
        self._decode.plan(
            metadata["decode_block_indptr"],
            metadata["decode_block_ids"],
            metadata["decode_last_block_lengths"],
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            sm_scale=self.scale,
            data_type=torch.bfloat16,
            q_data_type=torch.bfloat16,
        )
        return True

    def _attention(self, layer_index, layer, hidden, positions, metadata, decode):
        torch = self._torch
        token_count = hidden.shape[0]
        queries = layer.q_proj(hidden).view(
            token_count, self.num_query_heads, self.head_dim
        )
        keys = layer.k_proj(hidden).view(token_count, self.num_kv_heads, self.head_dim)
        values = layer.v_proj(hidden).view(
            token_count, self.num_kv_heads, self.head_dim
        )
        queries = layer.q_layernorm(queries)
        keys = layer.k_layernorm(keys)
        queries, keys = layer._apply_rope(
            queries.transpose(0, 1).unsqueeze(0),
            keys.transpose(0, 1).unsqueeze(0),
            positions.unsqueeze(0),
        )
        queries = queries.squeeze(0).transpose(0, 1).contiguous()
        keys = keys.squeeze(0).transpose(0, 1).contiguous()

        cache_slot = self._attention_slot[layer_index]
        k_cache = self._kv.keys[cache_slot]
        v_cache = self._kv.values[cache_slot]
        paged = metadata["paged_kv"]
        self._flashinfer.append_paged_kv_cache(
            keys,
            values,
            paged.append_batch_indices,
            paged.append_positions,
            (k_cache, v_cache),
            paged.page_ids,
            paged.indptr,
            paged.last_page_lengths,
            kv_layout="NHD",
        )

        output = torch.empty_like(queries)
        if metadata["prefill_max_q_len"]:
            prefill_token_indices = metadata["prefill_token_indices"]
            prefill_output = self._fa4(
                queries[prefill_token_indices],
                keys[prefill_token_indices],
                values[prefill_token_indices],
                cu_seqlens_q=metadata["prefill_cu_seqlens"],
                cu_seqlens_k=metadata["prefill_cu_seqlens"],
                max_seqlen_q=metadata["prefill_max_q_len"],
                max_seqlen_k=metadata["prefill_max_q_len"],
                softmax_scale=self.scale,
                causal=True,
                pack_gqa=True,
            )
            if isinstance(prefill_output, tuple):
                prefill_output = prefill_output[0]
            output[prefill_token_indices] = prefill_output

        if metadata["continuation_prefill_max_q_len"]:
            continuation_token_indices = metadata["continuation_prefill_token_indices"]
            continuation_output = paged_fa4(
                self._fa4,
                queries[continuation_token_indices],
                k_cache,
                v_cache,
                cu_seqlens_q=metadata["continuation_prefill_cu_seqlens"],
                max_seqlen_q=metadata["continuation_prefill_max_q_len"],
                page_table=metadata["continuation_prefill_page_table"],
                seqused_k=metadata["continuation_prefill_seqused_k"],
                min_seqlen_k=metadata["continuation_prefill_min_seqlen_k"],
                max_seqlen_k=metadata["continuation_prefill_max_seqlen_k"],
                softmax_scale=self.scale,
            )
            output[continuation_token_indices] = continuation_output

        if decode:
            decode_token_indices = metadata["decode_token_indices"]
            decode_output = self._decode.run(
                queries[decode_token_indices], (k_cache, v_cache)
            )
            output[decode_token_indices] = decode_output

        return layer.out_proj(output.reshape(token_count, -1))

    def _convolution(self, layer_index, layer, hidden, plan, recurrent_rows, ephemeral):
        conv_slot = self._conv_slot[layer_index]
        outputs = []
        for index, row in enumerate(recurrent_rows):
            start, end = plan.sequence_indptr[index : index + 2]
            state = self._state.view(conv_slot, row).unsqueeze(0)
            cache = _InPlaceRecurrentState(state)
            # The facade turns ShortConv's ``state.value = next_state`` into an
            # in-place copy and therefore cannot detach this generation from its
            # fixed physical slab address. Retain both view and facade until the
            # terminal event proves every dependent kernel complete.
            ephemeral.extend((state, cache))
            outputs.append(layer(hidden[start:end].unsqueeze(0), cache).squeeze(0))
        return self._torch.cat(outputs, dim=0)

    def _run_model(
        self,
        plan,
        tokens,
        positions,
        final_indices,
        metadata,
        decode,
        recurrent_rows,
        ephemeral,
    ):
        torch = self._torch
        hidden = self._model.model.embed_tokens(tokens)
        for layer_index, layer in enumerate(self._model.model.layers):
            normalized = layer.operator_norm(hidden)
            if layer.is_attention_layer:
                residual = self._attention(
                    layer_index,
                    layer.self_attn,
                    normalized,
                    positions,
                    metadata,
                    decode,
                )
            else:
                residual = self._convolution(
                    layer_index,
                    layer.conv,
                    normalized,
                    plan,
                    recurrent_rows,
                    ephemeral,
                )
            hidden = hidden + residual
            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))
        hidden = self._model.model.embedding_norm(hidden)
        logits = torch.nn.functional.linear(
            hidden[final_indices], self._model.model.embed_tokens.weight
        )
        output = self._sampler(logits, plan)
        # Keep the terminal activation chain and device result explicitly alive.
        # PyTorch's stream-aware allocator safely recycles earlier layer-local
        # temporaries in order on this same stream.
        ephemeral.extend((hidden, logits))
        return output

    def _completion_resources(self):
        """Return the persistent CUDA objects one accepted launch must lease."""

        return (
            self._torch,
            self._flashinfer,
            self._fa4,
            self._model,
            self._kv,
            self._kv.storage,
            self._kv.keys,
            self._kv.values,
            self._decode_workspace,
            self._decode,
            self._stream,
            self._state,
            self._state.slab,
            self._state.generations,
            self._state.owners,
            self._state.binding,
        )

    def dispatch(self, plan: BatchPlan):
        """Enqueue one model step on the private stream and return its capsule."""

        if self._executor.inflight is not None:
            raise RuntimeError("Lfm2CudaRunner already has an in-flight batch")
        nonempty_plan(plan)
        check_token_ids(plan.token_ids, self.vocab_size)
        cache_metadata = self._preflight_cache(plan)
        recurrent_rows = self._state.rows(plan)

        torch = self._torch

        def launch(completion: CudaCompletion):
            self._state.prepare_rows(recurrent_rows)
            metadata = self._metadata(plan, cache_metadata, completion.staging)
            completion.retain(metadata, recurrent_rows)
            tokens = completion.staging.device("tokens", plan.token_ids, torch.long)
            positions = completion.staging.device(
                "positions", plan.positions, torch.long
            )
            final_indices = completion.staging.device(
                "final_indices",
                [end - 1 for end in plan.sequence_indptr[1:]],
                torch.long,
            )
            completion.retain(tokens, positions, final_indices)
            decode = self._plan_decode(plan, metadata)
            return self._run_model(
                plan,
                tokens,
                positions,
                final_indices,
                metadata,
                decode,
                recurrent_rows,
                completion.ephemeral,
            )

        return self._executor.dispatch(plan, launch)

    def poll(self, completion) -> list[int] | None:
        """Return None while pending, then retire and return one token per row."""

        return self._executor.poll(completion)
