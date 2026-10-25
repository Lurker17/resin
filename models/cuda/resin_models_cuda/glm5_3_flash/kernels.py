"""GLM KDA and sparse-index operations, with independent small CPU references.

Semantics follow Transformers' GLM5-Next implementation at
50e35cc0ca18ec19b80653690de553e6decf74f5 (Apache-2.0). CUDA uses the
pinned FLA KDA and FlashMLA implementations; page adapters are Resin-owned.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F


def convolution(
    qkv: Tensor, weight: Tensor, cache, *, planar: bool = False
) -> tuple[Tensor, Tensor]:
    """Causal conv4 + SiLU, optionally returning contiguous Q/K/V planes.

    The default result retains the ``[tokens,3,channels]`` API. ``planar``
    returns ``[3,tokens,channels]`` without changing any values or state updates.
    """
    if qkv.is_cuda:
        from resin_models_cuda.glm5_3_flash.kernels_cuda import convolution_cuda

        return convolution_cuda(qkv, weight, cache, planar=planar)
    output = torch.empty_like(qkv)
    final = cache.convolution.clone()
    offsets = cache.cu_seqlens.tolist()
    for row, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if end == start:
            continue
        history = cache.convolution[row]
        source = torch.cat((history, qkv[start:end].permute(1, 2, 0)), -1)
        groups, channels, _ = source.shape
        value = F.conv1d(
            source.reshape(1, groups * channels, -1).float(),
            weight.reshape(groups * channels, 1, -1).float(),
            groups=groups * channels,
        )
        output[start:end] = F.silu(value).reshape(groups, channels, -1).permute(2, 0, 1)
        final[row].copy_(source[..., -history.shape[-1] :])
    return (output.permute(1, 0, 2).contiguous() if planar else output), final


def reference_kda(q, k, v, g, beta, state, offsets):
    """Sequential FP32 state oracle; intentionally used only for CPU tests."""
    output = torch.empty_like(v)
    final = state.clone()
    qf, kf = q.float(), k.float()
    qf = qf / (qf.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    kf = kf / (kf.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    qf = qf * q.shape[-1] ** -0.5
    bounds = offsets.tolist()
    for row, (start, end) in enumerate(zip(bounds, bounds[1:])):
        current = state[row].float()
        for token in range(start, end):
            current = current * g[token].float().exp().unsqueeze(-1)
            prediction = (current * kf[token].unsqueeze(-1)).sum(-2)
            delta = (v[token].float() - prediction) * beta[token].float().unsqueeze(-1)
            current = current + kf[token].unsqueeze(-1) * delta.unsqueeze(-2)
            output[token] = (current * qf[token].unsqueeze(-1)).sum(-2)
        final[row] = current
    return output, final


def kda(q, k, v, forget, beta, a_log, dt_bias, lower_bound, cache):
    """KDA uses per-key decay, unlike Qwen's scalar-per-head delta rule."""
    # Serving reference: SGLang 5971f47's KDA keeps sigmoid(beta) in FP32
    # for both chunk prefill and recurrent decode. Rounding it to BF16 here
    # changes every recurrent update, including speculative accepted-prefix replay.
    if not q.is_cuda:
        beta = beta.float().sigmoid()
        gate = forget.float() + dt_bias.float().reshape(1, *forget.shape[-2:])
        decay = a_log.float().reshape(1, -1, 1).exp()
        gate = (
            lower_bound * torch.sigmoid(decay * gate)
            if lower_bound is not None
            else -decay * F.softplus(gate)
        )
        return reference_kda(q, k, v, gate, beta, cache.recurrent, cache.cu_seqlens)
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    from resin_models_cuda.glm5_3_flash.state_cuda import beta_sigmoid, preserve_idle

    beta = beta_sigmoid(beta)

    small = q.shape[0] <= (cache.cu_seqlens.shape[0] - 1) * 8
    # FLA input_guard loses its wrapped callable signature in upstream typing.
    kernel = cast(
        Callable[..., tuple[Tensor, Tensor]],
        fused_recurrent_kda if small else chunk_kda,
    )
    kwargs = {}
    if not small:
        from resin_models_cuda.glm5_3_flash.kernels_cuda import l2norm_cuda

        # FLA 0.5.2's standalone L2 norm specializes on ceil(T*H/65536),
        # recompiling each new 1024-token band for GLM's 64 heads. Normalize at
        # the same BF16 boundary with runtime row counts before entering FLA.
        q, k = l2norm_cuda(q), l2norm_cuda(k)
        # Prepared host offsets avoid FLA's GPU-to-host metadata synchronization.
        if cache.cu_seqlens_cpu is None:
            raise ValueError("KDA prefill requires prepared CPU sequence offsets")
        kwargs = {
            "cu_seqlens_cpu": cache.cu_seqlens_cpu,
            "safe_gate": lower_bound is not None,
        }
    output, final = kernel(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        forget.unsqueeze(0),
        beta.unsqueeze(0),
        A_log=a_log.reshape(-1).float(),
        dt_bias=dt_bias.reshape(-1).float(),
        initial_state=cache.recurrent.contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=small,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cache.cu_seqlens,
        **kwargs,
    )
    # FLA's recurrent kernel returns before writing an empty sequence's final
    # state. Preserve it explicitly for idle/padded attention replicas.
    final = preserve_idle(cache.recurrent, final, cache.cu_seqlens)
    return output.squeeze(0), final


def store_sparse(latent: Tensor, keys: Tensor, gates: Tensor, ape: Tensor, cache):
    """Write newly appended rows and compress only newly completed pools."""
    if latent.is_cuda:
        from resin_models_cuda.glm5_3_flash.kernels_cuda import store_sparse_cuda

        store_sparse_cuda(latent, keys, gates, ape, cache)
        return
    origin = getattr(cache, "storage_origin", 0)
    for token, (seq, pos) in enumerate(
        zip(cache.sequence_indices.tolist(), cache.positions.tolist())
    ):
        pos += origin
        page = int(cache.page_table[seq, pos // cache.page_tokens])
        if page <= 0:
            continue
        offset = pos % cache.page_tokens
        cache.latent[page, offset] = latent[token]
        cache.index_keys[page, offset] = keys[token]
        cache.index_gates[page, offset] = gates[token]
    for seq, pos in zip(cache.sequence_indices.tolist(), cache.positions.tolist()):
        if (pos + 1) % 4:
            continue
        pool_slot = pos // 4 + origin
        page = int(cache.page_table[seq, pool_slot // (cache.page_tokens // 4)])
        if page <= 0:
            continue
        # Pool membership is logical; shifted MTP groups can span two pages.
        source = torch.arange(pos - 3 + origin, pos + 1 + origin)
        source_pages = cache.page_table[seq, source // cache.page_tokens].long()
        offsets = source % cache.page_tokens
        pool_keys = cache.index_keys[source_pages, offsets]
        logits = cache.index_gates[source_pages, offsets].float() + ape.float()
        probs = logits.softmax(0).to(pool_keys.dtype)
        cache.pooled_keys[page, pool_slot % (cache.page_tokens // 4)] = (
            probs * pool_keys
        ).sum(0)


def select_indices(
    query: Tensor,
    weights: Tensor,
    positions: Tensor,
    sequences: Tensor,
    cache,
    topk: int,
):
    """Score complete pools, expand selected pools, and append the causal tail."""
    if query.is_cuda:
        from resin_models_cuda.glm5_3_flash.kernels_cuda import select_indices_cuda

        return select_indices_cuda(query, weights, positions, sequences, cache, topk)
    origin = getattr(cache, "storage_origin", 0)
    pools = cache.page_table.shape[1] * cache.page_tokens // 4
    scores = torch.full((query.shape[0], pools), -torch.inf, device=query.device)
    for row, (seq, pos) in enumerate(zip(sequences.tolist(), positions.tolist())):
        for pool in range((pos + 1) // 4):
            pool_slot = pool + origin
            page = int(cache.page_table[seq, pool_slot // (cache.page_tokens // 4)])
            if page <= 0:
                continue
            key = cache.pooled_keys[page, pool_slot % (cache.page_tokens // 4)]
            per_head = F.relu(
                (query[row].float() * key.float()).sum(-1) * query.shape[-1] ** -0.5
            )
            scores[row, pool] = (
                per_head * weights[row].float() * query.shape[1] ** -0.5
            ).sum()
    selected = scores.topk(min(topk // 4, pools), dim=-1, sorted=False).indices
    logical = selected[..., None] * 4 + torch.arange(4, device=query.device)
    logical = logical.reshape(query.shape[0], -1)
    visible = (selected + 1) * 4 <= positions[:, None] + 1
    logical = torch.where(visible.repeat_interleave(4, -1), logical, -1)
    tail_start = (positions + 1) // 4 * 4
    tail = tail_start[:, None] + torch.arange(3, device=query.device)
    tail = torch.where(tail <= positions[:, None], tail, -1)
    logical = torch.cat(
        (F.pad(logical, (0, topk - logical.shape[1]), value=-1), tail), -1
    )
    output = torch.full((query.shape[0], (topk + 10) // 8 * 8), -1, dtype=torch.int32)
    stride = cache.latent.stride(0) // cache.latent.shape[-1]
    for row, seq in enumerate(sequences.tolist()):
        for col, pos in enumerate(logical[row].tolist()):
            if pos < 0:
                continue
            pos += origin
            page = int(cache.page_table[seq, pos // cache.page_tokens])
            if page > 0:
                output[row, col] = page * stride + pos % cache.page_tokens
    return output


def latent_envelope(latent: Tensor) -> Tensor:
    """View the live cache allocation in latent-width rows, including page gaps."""
    pages, tokens, width = latent.shape
    if latent.stride(-1) != 1 or latent.stride(1) != width or latent.stride(0) % width:
        raise ValueError("MLA cache requires compact tokens and aligned physical pages")
    rows = ((pages - 1) * latent.stride(0) + tokens * width) // width
    return latent.as_strided((rows, 1, width), (width, width, 1))


def sparse_attention(query: Tensor, indices: Tensor, latent: Tensor, scale: float):
    """Sparse absorbed MLA; never expand full history into 64 K/V heads."""
    envelope = latent_envelope(latent)
    if query.is_cuda:
        from flash_mla import flash_mla_sparse_fwd

        # The pinned SM100 head64 sparse kernel reads complete 64-index tiles.
        # GLM's 2048 selected tokens plus causal tail must therefore be padded
        # to 2112, even though the general interface documents no such bound.
        padding = (-indices.shape[-1]) % 64
        if padding:
            indices = F.pad(indices, (0, padding), value=-1)
        return flash_mla_sparse_fwd(
            query.contiguous(),
            envelope,
            indices[:, None].contiguous(),
            scale,
            d_v=latent.shape[-1],
        )[0]
    output = torch.empty_like(query)
    for row in range(query.shape[0]):
        valid = indices[row][indices[row] >= 0].long()
        selected = envelope[valid, 0].float()
        if not valid.numel():
            output[row].zero_()
            continue
        logits = torch.matmul(query[row].float(), selected.T) * scale
        output[row] = torch.matmul(logits.softmax(-1), selected)
    return output
