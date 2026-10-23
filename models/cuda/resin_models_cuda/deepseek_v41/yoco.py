"""Retained consumer queries over complete producer KV and Engine page ownership."""

import torch

from resin_models_cuda.deepseek_v41.batch import ForwardMetadata


def compact_metadata(
    frame: ForwardMetadata, query_indices: torch.Tensor, *, swa_start: int
):
    """Select shared query columns; negative/out-of-range entries are CP padding.

    A real placeholder before ``swa_start`` keeps its compressed causal history,
    but cannot read or write consumer SWA. Original metadata is never mutated.
    """
    if (
        query_indices.ndim != 1
        or query_indices.dtype != torch.long
        or not query_indices.numel()
        or query_indices.device != frame.positions.device
    ):
        raise ValueError("query_indices must be a nonempty device int64 vector")
    if type(swa_start) is not int or swa_start < 0:
        raise ValueError("swa_start must be a nonnegative integer")
    batch, queries = frame.positions.shape
    safe = query_indices.clamp(0, queries - 1)
    present = (query_indices >= 0) & (query_indices < queries)

    def select(value):
        return value.index_select(1, safe)

    positions = select(frame.positions)
    valid = select(frame.valid_token_mask.reshape(batch, queries)) & present[None]
    writable = valid & (positions >= swa_start)

    def stores(value):
        return torch.where(writable, select(value), -1)

    indices = select(frame.kv_swa_index)
    window = indices.shape[-1]
    end = positions + 1
    logical = (
        end[..., None]
        - end.clamp(0, window)[..., None]
        + torch.arange(window, device=positions.device)
    )
    indices = torch.where(valid[..., None] & (logical >= swa_start), indices, -1)
    tables = frame.block_tables.reshape(batch, queries, -1).index_select(1, safe)
    return ForwardMetadata(
        positions=positions,
        context_lens=select(frame.context_lens),
        valid_token_mask=valid.flatten(),
        slot_mapping_store_swa=stores(frame.slot_mapping_store_swa),
        kv_swa_index=indices,
        positions_c={
            ratio: select(value) for ratio, value in frame.positions_c.items()
        },
        slot_mapping_store_ca={
            ratio: stores(value) for ratio, value in frame.slot_mapping_store_ca.items()
        },
        klen={
            ratio: torch.where(valid, select(value), 0)
            for ratio, value in frame.klen.items()
        },
        state_block_idx=stores(frame.state_block_idx),
        query_indices=select(frame.query_indices.reshape(batch, queries)).flatten(),
        block_tables=tables.flatten(0, 1),
        draft_slots=stores(frame.draft_slots),
        draft_indices=torch.where(writable[..., None], select(frame.draft_indices), -1),
    )


def compact_runtime(runtime, query_indices: torch.Tensor, *, swa_start: int):
    """Rebuild query-local schedules after selecting the retained consumer rows."""
    frame = runtime["metadata"]
    batch, queries = frame.positions.shape
    result = dict(runtime)
    result["metadata"] = compact_metadata(frame, query_indices, swa_start=swa_start)
    safe = query_indices.clamp(0, queries - 1)
    present = (query_indices >= 0) & (query_indices < queries)
    for name in ("indices", "candidate_blocks"):
        if name not in runtime:
            continue
        value = runtime[name]
        # Top-k indices retain [B,S,K]; CED candidates use flattened [B*S,K].
        flat = name == "candidate_blocks"
        shaped = value.reshape(batch, queries, -1) if flat else value
        selected = shaped.index_select(1, safe)
        selected = torch.where(present[None, :, None], selected, -1)
        result[name] = selected.flatten(0, 1) if flat else selected
    for name in ("index_metadata", "prefill_layout", "prefill_compressed"):
        result.pop(name, None)
    result["swa_start"] = swa_start
    return result
