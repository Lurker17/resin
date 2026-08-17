"""Paged FA4/FlashInfer execution for the Resin-owned PyTorch LFM2 model.

This deliberately specializes one model and one CUDA stack. Rust supplies the
complete immutable page, append, prefill, and decode subplans. Attention history
and recurrent checkpoints share one Engine-owned pool of fungible parents.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from resin import (
    BatchPhase,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    ResourceBinding,
    RunnerConfig,
)
from resin.backend.cuda.paged_kv import PagedKvCsr
from resin.backend.cuda.sampling import FlashInferSampler
from resin.backend.cuda.staging import TensorStager
from resin.backend.plans import BatchView, CacheGroupView
from resin.protocols import SyncRunner
from resin_models_cuda.lfm2.plan import CacheMetadata, build_cache_metadata
from resin_models_torch.lfm2.reference import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    load_model,
    resolve_model_path,
)

if TYPE_CHECKING:
    from resin import BatchPlan


_STATE_CHECKPOINT_TOKENS = 128


class _PostAcceptCudaError(RuntimeError):
    """Tell the Engine that launched CUDA work could not be quiesced."""

    _resin_post_accept = True


class _Lfm2CacheArena:
    """Model-specific views over one fungible parent allocation."""

    def __init__(
        self,
        torch,
        device,
        layout,
        *,
        attention_layers: int,
        conv_layers: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        state_length: int,
        dtype,
    ) -> None:
        from resin.backend.resources import CachePoolBinding

        self.binding = CachePoolBinding(layout, resource="LFM2 cache")
        self.parent_capacity = layout.parent_capacity
        self.state_packing = layout.groups[1].blocks_per_slab
        kv_elements = 2 * attention_layers * page_size * num_kv_heads * head_dim
        state_elements = conv_layers * hidden_size * state_length
        if kv_elements != self.state_packing * state_elements:
            raise ValueError("LFM2 cache groups do not exactly fill one parent")
        self.storage = torch.zeros(
            (self.parent_capacity + 1, kv_elements), dtype=dtype, device=device
        )
        kv = self.storage.view(
            self.parent_capacity + 1,
            2,
            attention_layers,
            page_size,
            num_kv_heads,
            head_dim,
        ).permute(1, 2, 0, 3, 4, 5)
        self.keys = kv[0]
        self.values = kv[1]
        self._state = self.storage[1:].view(
            self.parent_capacity,
            self.state_packing,
            conv_layers,
            hidden_size,
            state_length,
        )

    def bind(self, pool_id: int, layout) -> int:
        return self.binding.bind(pool_id, layout)

    def state(self, address: tuple[int, int]):
        parent, child_slot = address
        return self._state[parent - 1, child_slot]


class _StateRow:
    __slots__ = ("input_address", "output_address")

    def __init__(
        self, input_address: tuple[int, int] | None, output_address: tuple[int, int]
    ) -> None:
        self.input_address = input_address
        self.output_address = output_address


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


class Lfm2CudaRunner(SyncRunner):
    """BF16 LFM2 runner with FA4 prefill and FlashInfer paged decode."""

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
        self.model_path = resolve_model_path(model, revision)
        # Persistent CUDA state is created on the same private stream used for
        # every forward call, avoiding an implicit default-stream handoff.
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

            kv_elements = (
                2
                * len(self._attention_layers)
                * page_size
                * self.num_kv_heads
                * self.head_dim
            )
            state_elements = (
                len(self._conv_layers) * self.hidden_size * (self.conv_cache_length - 1)
            )
            if state_elements == 0 or kv_elements % state_elements:
                raise ValueError("LFM2 recurrent state does not pack into a KV parent")
            state_packing = kv_elements // state_elements
            self.cache_layout = CacheLayout(
                max_pages,
                [
                    CacheGroupLayout(page_size),
                    CacheGroupLayout(
                        _STATE_CHECKPOINT_TOKENS,
                        blocks_per_slab=state_packing,
                        family=CacheGroupFamily.STATE,
                    ),
                ],
            )
            self.runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=_STATE_CHECKPOINT_TOKENS,
                cache_layout=self.cache_layout,
            )

            # FA4's SM90 paged path may issue TMA loads for the full final
            # physical page and mask scores afterwards. Zero the unwritten tail
            # so a masked lane can never feed NaN/garbage into the PV reduction.
            self._cache = _Lfm2CacheArena(
                torch,
                self.device,
                self.cache_layout,
                attention_layers=len(self._attention_layers),
                conv_layers=len(self._conv_layers),
                page_size=page_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                hidden_size=self.hidden_size,
                state_length=self.conv_cache_length - 1,
                dtype=torch.bfloat16,
            )
            self._decode_workspace = torch.zeros(
                workspace_bytes, dtype=torch.uint8, device=self.device
            )
            self._decode = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                self._decode_workspace, "NHD"
            )

    def bind_resources(self, binding: ResourceBinding) -> None:
        """Bind the unified physical arena to its Engine cache pool."""

        if binding.cache_pool_id is None or binding.cache_layout is None:
            raise ValueError("LFM2 requires managed cache resources")
        self._cache.bind(binding.cache_pool_id, binding.cache_layout)

    def _preflight_cache(self, plan: BatchPlan) -> CacheMetadata:
        """Validate and derive the complete cache launch before CUDA acceptance."""

        groups = tuple(plan.cache_groups)
        if len(groups) != 2 or tuple(group.group_index for group in groups) != (0, 1):
            raise ValueError("BatchPlan does not contain LFM2 history and state groups")
        return build_cache_metadata(
            plan,
            self.page_size,
            parent_capacity=self._cache.parent_capacity,
        )

    def _state_rows(self, plan: BatchPlan) -> tuple[_StateRow, ...]:
        batch = BatchView.from_plan(plan)
        if len(batch.cache_groups) != 2:
            raise ValueError("LFM2 requires History and State cache groups")
        group = CacheGroupView.from_plan(
            batch.cache_groups[1],
            batch,
            group_index=1,
            blocks_per_slab=self._cache.state_packing,
            parent_capacity=self._cache.parent_capacity,
            block_tokens=_STATE_CHECKPOINT_TOKENS,
        )
        block_indptr = group.block_indptr
        block_parents = group.block_slab_ids
        block_slotnums = group.block_slotnums
        first_blocks = group.block_table_start_indices
        result = []
        for index, context in enumerate(batch.context_lengths):
            context = int(context)
            token_start, token_end = map(int, batch.token_indptr[index : index + 2])
            post_length = context + token_end - token_start
            first = int(first_blocks[index])
            page_start, page_end = map(int, block_indptr[index : index + 2])
            addresses = tuple(
                zip(
                    map(int, block_parents[page_start:page_end]),
                    map(int, block_slotnums[page_start:page_end]),
                )
            )

            def address(logical: int) -> tuple[int, int]:
                column = logical - first
                if column < 0 or column >= len(addresses):
                    raise ValueError("LFM2 state table does not cover a checkpoint")
                return addresses[column]

            input_address = (
                None
                if context == 0
                else address((context - 1) // _STATE_CHECKPOINT_TOKENS)
            )
            output_address = address((post_length - 1) // _STATE_CHECKPOINT_TOKENS)
            result.append(_StateRow(input_address, output_address))
        return tuple(result)

    def _prepare_state_rows(self, rows: tuple[_StateRow, ...]) -> None:
        for row in rows:
            output = self._cache.state(row.output_address)
            if row.input_address is None:
                output.zero_()
            elif row.input_address != row.output_address:
                output.copy_(self._cache.state(row.input_address))

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
            staging = TensorStager(self.device)
        metadata = {"_pinned": staging.host_tensors, "_staging": staging}
        metadata.update(
            staging.tensors(
                {
                    "block_indptr": (derived.block_indptr, torch.int32),
                    "page_indices": (derived.page_indices, torch.int32),
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
                    "decode_page_indices": (derived.decode_page_indices, torch.int32),
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
            metadata["page_indices"],
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
            metadata["decode_page_indices"],
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
        k_cache = self._cache.keys[cache_slot]
        v_cache = self._cache.values[cache_slot]
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
            continuation_output = self._fa4(
                queries[continuation_token_indices],
                k_cache,
                v_cache,
                cu_seqlens_q=metadata["continuation_prefill_cu_seqlens"],
                cu_seqlens_k=None,
                max_seqlen_q=metadata["continuation_prefill_max_q_len"],
                max_seqlen_k=metadata["continuation_prefill_max_seqlen_k"],
                min_seqlen_k=metadata["continuation_prefill_min_seqlen_k"],
                seqused_q=None,
                seqused_k=metadata["continuation_prefill_seqused_k"],
                page_table=metadata["continuation_prefill_page_table"],
                softmax_scale=self.scale,
                causal=True,
                window_size=(None, None),
                softcap=0.0,
                num_splits=1,
                pack_gqa=True,
                deterministic=False,
                return_lse=False,
            )
            if isinstance(continuation_output, tuple):
                continuation_output = continuation_output[0]
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
            start, end = map(int, plan.token_indptr[index : index + 2])
            state = self._cache.state(row.output_address)[conv_slot].unsqueeze(0)
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

    def _synchronize_failed_forward(self) -> None:
        """Make a failed launch quiescent or force the Engine to quarantine it."""

        try:
            self._stream.synchronize()
        except BaseException as synchronization_error:
            raise _PostAcceptCudaError(
                "LFM2 execution failed after launch and its CUDA stream could "
                "not be quiesced; resources must remain quarantined"
            ) from synchronization_error

    def forward(self, plan: BatchPlan) -> list[int]:
        """Execute one immutable Engine plan to completion."""

        cache_metadata = self._preflight_cache(plan)
        recurrent_rows = self._state_rows(plan)
        torch = self._torch
        staging = TensorStager(self.device)
        ephemeral = []
        try:
            with torch.cuda.stream(self._stream), torch.inference_mode():
                self._prepare_state_rows(recurrent_rows)
                metadata = self._metadata(plan, cache_metadata, staging)
                ephemeral.extend((metadata, recurrent_rows))
                tokens = staging.device("tokens", plan.token_ids, torch.long)
                positions = staging.device("positions", plan.positions, torch.long)
                final_indices = staging.device(
                    "final_indices",
                    plan.token_indptr[1:] - 1,
                    torch.long,
                )
                ephemeral.extend((tokens, positions, final_indices))
                decode = self._plan_decode(plan, metadata)
                output = self._run_model(
                    plan,
                    tokens,
                    positions,
                    final_indices,
                    metadata,
                    decode,
                    recurrent_rows,
                    ephemeral,
                )
                ephemeral.append(output)
            self._stream.synchronize()
            return output.tolist()
        except BaseException:
            self._synchronize_failed_forward()
            raise
