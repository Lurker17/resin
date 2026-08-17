from __future__ import annotations

import sys
from types import ModuleType

import torch

from resin_models_cuda.kimi_k25.vision import _flash_attention_4


def test_flash_attention_4_adapter_uses_keyword_varlen_metadata(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_flash_attention(query, key, value, **kwargs):
        del key, value
        calls.append(kwargs)
        return query

    interface = ModuleType("flash_attn.cute.interface")
    interface.flash_attn_varlen_func = fake_flash_attention
    cute = ModuleType("flash_attn.cute")
    flash_attn = ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)

    query = torch.zeros(12, 3, 8)
    cu_seqlens = torch.tensor([0, 5, 12], dtype=torch.int32)
    output = _flash_attention_4(
        query,
        query,
        query,
        q_cu_seqlens=cu_seqlens,
        k_cu_seqlens=cu_seqlens,
        max_seqlen_q=torch.tensor(7),
        max_seqlen_k=torch.tensor(7),
        deterministic=True,
    )

    assert output.shape == (12, 24)
    assert calls == [
        {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": 12,
            "max_seqlen_k": 12,
            "causal": False,
            "deterministic": True,
        }
    ]
