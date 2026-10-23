"""A small actual-Engine replay checks the CPU benchmark's cache/speculation setup."""

from scripts.bench_engine_control_plane import run_case


def test_cpu_replay_uses_shared_pages_and_commits_full_speculative_rows():
    result = run_case(2, 3, context=256, shared=128, accepted=3, lookahead=3, warmup=1)
    assert result["cached_tokens_min"] == result["cached_tokens_max"] == 128
    assert result["history_blocks_per_sequence"] == 2
    assert result["accepted_tokens_per_sequence"] == 3
    assert result["page_packings"] == (1, 16, 1, 2048)
    assert len(result["samples"]) == 3
    for sample in result["samples"]:
        assert sample["whole_ms"] >= sample["prepare_ms"] > 0
        assert sample["callback_and_complete_ms"] >= 0
