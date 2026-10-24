"""Blackwell primitive probes and real-checkpoint eager/captured validation."""

from contextlib import contextmanager

import torch


def _check(actual, expected, *, atol=0, rtol=0):
    actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    difference = actual - expected
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-20)).item(),
        "values": actual.numel(),
    }


@contextmanager
def _bf16_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _fp8_reference(values, block=32):
    grouped = values.float().unflatten(-1, (-1, block))
    scales = (grouped.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
    quantized = (grouped / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.flatten(-2), scales


def _primitives():
    from resin_models_cuda.deepseek_v41.kernels import (
        act_quant,
        fp4_act_quant,
        fp8_gemm,
        hc_split_sinkhorn,
    )

    values = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
    values[0].zero_()
    values[1].fill_(1e-6)
    values[2] = torch.linspace(-6, 6, 128, device="cuda")
    expected_q, expected_s = _fp8_reference(values)
    quantized, scales = act_quant(values, 32, "ue8m0", torch.float8_e8m0fnu)
    result = {
        "fp8_values": _check(quantized, expected_q),
        "fp8_scales": _check(scales, expected_s),
    }
    inplace = values.clone()
    act_quant(inplace, 32, "ue8m0", torch.float8_e8m0fnu, True)
    result["fp8_inplace"] = _check(
        inplace,
        (expected_q.float() * expected_s.repeat_interleave(32, -1)).bfloat16(),
    )

    weight = torch.randn(160, 128, device="cuda").to(torch.float8_e4m3fn)
    weight_scales = torch.exp2(
        torch.arange(20, device="cuda").reshape(5, 4).remainder(5).float() - 2
    ).to(torch.float8_e8m0fnu)
    with _bf16_default():
        output = fp8_gemm(
            quantized,
            scales,
            weight,
            weight_scales,
            torch.float8_e8m0fnu,
            block_size=32,
        )
    left = quantized.float() * scales.float().repeat_interleave(32, -1)
    right = weight.float() * weight_scales.float().repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    result["fp8_gemm"] = _check(
        output, left.bfloat16() @ right.bfloat16().T, atol=0.0625, rtol=0.01
    )

    values = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    values[0].zero_()
    values[1].fill_(1e-6)
    # Unit scale with exact FP4 midpoints exercises ties-to-even, both signs.
    edges = torch.tensor(
        [
            0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5,
            6,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5,
        ],
        device="cuda",
    )
    values[2] = edges.repeat(32)
    grouped = values.float().unflatten(-1, (-1, 16))
    expected_s = (
        (grouped.abs().amax(-1).clamp_min(6 * 2**-9) / 6)
        .to(torch.float8_e4m3fn)
        .float()
    )
    normalized = (grouped / expected_s[..., None]).clamp(-6, 6)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    distances = (normalized.abs()[..., None] - levels).abs()
    nearest = distances.argmin(-1)
    ties = (distances == distances.amin(-1, keepdim=True)).sum(-1) > 1
    nearest += (ties & (nearest.remainder(2) == 1)).long()
    expected = (
        (levels[nearest] * normalized.sign() * expected_s[..., None])
        .flatten(-2)
        .bfloat16()
    )
    packed, scales = fp4_act_quant(values, 16, scale_dtype=torch.float8_e4m3fn)
    result["fp4_scales"] = _check(scales, expected_s)
    # Independently unpack the two E2M1 nibbles, including negative zero.
    raw = packed.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
    unpacked = levels[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    result["fp4_packed"] = _check(
        (unpacked * expected_s.repeat_interleave(16, -1)).bfloat16(), expected
    )
    fp4_act_quant(values, 16, True, scale_dtype=torch.float8_e4m3fn)
    result["fp4_inplace"] = _check(values, expected)

    mixes = torch.randn(2, 7, 24, device="cuda", dtype=torch.float32)
    mixes[0, 0].zero_()
    mixes[0, 1] = torch.linspace(-80, 80, 24, device="cuda")
    scale = torch.tensor([0.4, -1.5, 1.2], device="cuda")
    base = torch.linspace(-3, 3, 24, device="cuda")
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    expected = (mixes[..., 8:] * scale[2] + base[8:]).unflatten(-1, (4, 4)).softmax(
        -1
    ) + 1e-6
    expected /= expected.sum(-2, keepdim=True) + 1e-6
    for _ in range(19):
        expected /= expected.sum(-1, keepdim=True) + 1e-6
        expected /= expected.sum(-2, keepdim=True) + 1e-6
    result["hc_pre"] = _check(
        pre,
        (mixes[..., :4] * scale[0] + base[:4]).sigmoid() + 1e-6,
        atol=2e-6,
        rtol=1e-5,
    )
    result["hc_post"] = _check(
        post,
        2 * (mixes[..., 4:8] * scale[1] + base[4:8]).sigmoid(),
        atol=2e-6,
        rtol=1e-5,
    )
    result["hc_sinkhorn"] = _check(comb, expected, atol=2e-6, rtol=1e-5)
    return result


def _attention_reference(query, cache, sinks, indices, scale):
    keys = cache.view(-1, query.shape[-1])[indices.clamp_min(0).long()].float()
    scores = torch.einsum("bshd,bskd->bshk", query.float(), keys) * scale
    scores.masked_fill_(indices[:, :, None] < 0, -torch.inf)
    scores = torch.cat(
        (scores, sinks[None, None, :, None].expand(*scores.shape[:-1], 1)), -1
    )
    probabilities = scores.softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probabilities, keys).bfloat16()


def _attention():
    from deploy.flashmla_validation import run_probe

    return run_probe()


@torch.inference_mode()
def run_probes():
    """Return JSON-serializable metrics; assertion failures stop the GPU job."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("V4.1 probes require a real Blackwell CUDA GPU")
    torch.manual_seed(41)
    result = {"hardware": torch.cuda.get_device_name(), "torch": torch.__version__}
    result["primitives"] = _primitives()
    print({"primitives": result["primitives"]}, flush=True)
    result["attention"] = _attention()
    torch.cuda.synchronize()
    result["ok"] = True
    return result


@contextmanager
def _validation_stage(stage, coordinate):
    try:
        yield
    except Exception as error:
        if coordinate is not None:
            coordinate(stage, error)
        raise
    else:
        if coordinate is not None:
            coordinate(stage, None)


def validate_overlap(runner, engine, prompt, *, coordinate=None, generated=160):
    """Check natural GPU feedback against synchronous captured execution.

    Token stops and scores deliberately remain disabled so the Engine can keep
    a decode in flight. The fixed output span crosses a 128-token page boundary.
    """
    from resin import SamplingParams

    local = getattr(engine, "local", engine)
    if len(prompt) + generated + runner.speculative_tokens > local.context_length:
        raise ValueError("overlap validation needs space for 160 generated tokens")

    def generate(overlap):
        name = "overlapped" if overlap else "synchronous captured"
        with _validation_stage(name + " admission", coordinate):
            local.clear_prefix_cache()
            before = local.overlap_stats
            request = engine.submit(
                prompt,
                sampling_params=SamplingParams(
                    max_new_tokens=generated, stop_token_ids=[], temperature=0, seed=41
                ),
            )
        tokens, steps = [], 0
        try:
            while engine.has_work():
                events = engine.step(allow_overlap=overlap)
                steps += 1
                with _validation_stage(name + " events", coordinate):
                    for event in events:
                        if event.request_id != request.id or event.index != len(tokens):
                            raise AssertionError("overlap returned misordered events")
                        tokens.append(event.token_id)
            with _validation_stage(name + " result", coordinate):
                if len(tokens) != generated or request.take_result() != tokens:
                    raise AssertionError("overlap lost committed tokens")
                after = local.overlap_stats
                continuations = after[1] - before[1]
                if overlap and not continuations:
                    raise AssertionError("overlap validation never continued on GPU")
                return {
                    "token_ids": tokens,
                    "steps": steps,
                    "enqueues": after[0] - before[0],
                    "continuations": continuations,
                }
        finally:
            with _validation_stage(name + " cleanup", coordinate):
                if not request.done and not getattr(local, "poisoned", False):
                    engine.cancel(request)

    reference, overlapped = generate(False), generate(True)
    with _validation_stage("overlap parity", coordinate):
        if reference["token_ids"] != overlapped["token_ids"]:
            index = next(
                i
                for i, pair in enumerate(
                    zip(reference["token_ids"], overlapped["token_ids"], strict=True)
                )
                if pair[0] != pair[1]
            )
            raise AssertionError(f"synchronous/overlapped tokens differ at {index}")
    return {"ok": True, "reference": reference, "overlapped": overlapped}


def validate_model(
    runner,
    engine,
    *,
    prompt_tokens,
    synchronize=torch.cuda.synchronize,
    coordinate=None,
):
    """Compare the same loaded shards through eager and captured execution.

    Call on every attention-DP driver in the same order; followers remain in
    DistributedWorker.serve(). Policy votes put the entire domain into eager
    execution when a driver disables graphs. Kernels and catalogs stay intact.
    Both runs use the Engine's actual chunk size, fresh scored prefixes, greedy
    target sampling, normal expert routes, and natural speculative acceptance.
    EOS is honored; the prompt must elicit at least three decode steps.
    """
    with _validation_stage("setup", coordinate):
        import math
        from pathlib import Path

        from tokenizers import Tokenizer

        from resin import BatchPhase, SamplingParams

        prompt = list(prompt_tokens)
        local = getattr(engine, "local", engine)
        if len(prompt) < 2 or local.has_work():
            raise ValueError(
                "validation requires a fresh engine and at least two prompt tokens"
            )
        policies = runner.cuda_graph_prefill, runner.cuda_graph_decode
        if policies != ("breakable", "full"):
            raise ValueError(
                "capture breakable prefill and full decode graphs before validation"
            )
        if any(
            getattr(module, "benchmark_balanced_experts", False)
            for module in runner._model.modules()
        ):
            raise ValueError("validation requires natural expert routing")
        if runner._speculation.benchmark_accepted_drafts is not None:
            raise ValueError("validation requires natural speculative acceptance")
        width = runner.speculative_tokens
        # Reserve three decode replays even if every draft matches; early EOS fails
        # the replay-count check instead of validating an out-of-distribution tail.
        generated = max(8, 3 * (width + 1) + 1)
        if len(prompt) + generated + width > local.context_length:
            raise ValueError(
                "validation prompt and generated tokens exceed Engine context capacity"
            )
        tokenizer = Tokenizer.from_file(str(Path(runner.model_path) / "tokenizer.json"))

    def counts():
        return {
            mode: stats["replay_count"]
            for mode, stats in runner.cuda_graph_stats.items()
        }

    def finite(value):
        if not math.isfinite(value) or value > 1e-5:
            raise AssertionError(f"expected a finite raw logprob <= 0, got {value}")
        return float(value)

    def generate(eager):
        stage = "eager" if eager else "captured"
        with _validation_stage(stage + " admission", coordinate):
            local.clear_prefix_cache()
            runner.cuda_graph_prefill, runner.cuda_graph_decode = (
                ("disabled", "disabled") if eager else policies
            )
            before = counts()
            accepted = runner._speculation.snapshot()
            request = engine.submit(
                prompt,
                sampling_params=SamplingParams(
                    max_new_tokens=generated,
                    stop_token_ids=[runner.eos_token_id],
                    temperature=0,
                    seed=41,
                    logprobs=True,
                    top_logprobs=3,
                    prompt_logprobs=True,
                ),
            )
        steps = {"prefill": 0, "decode": 0}
        tokens, scores, tops = [], [], []
        try:
            while engine.has_work():
                with _validation_stage(stage + " phase", coordinate):
                    phase = local.next_phase()
                events = engine.step()
                with _validation_stage(stage + " events", coordinate):
                    synchronize()
                    if phase is not None:
                        steps[
                            "prefill" if phase == BatchPhase.PREFILL else "decode"
                        ] += 1
                    for event in events:
                        if event.request_id != request.id or event.index != len(tokens):
                            raise AssertionError(
                                "isolated generation returned misordered token events"
                            )
                        if (
                            event.logprobs is None
                            or len(event.logprobs.top_logprobs) != 3
                        ):
                            raise AssertionError(
                                "generation omitted selected or top token scores"
                            )
                        tokens.append(event.token_id)
                        scores.append(finite(event.logprobs.logprob))
                        rows = [
                            (token, finite(score))
                            for token, score in event.logprobs.top_logprobs
                        ]
                        if not any(token == event.token_id for token, _ in rows):
                            raise AssertionError(
                                "greedy token is absent from the top three"
                            )
                        tops.append(rows)
            with _validation_stage(stage + " result", coordinate):
                ended_at_eos = bool(tokens) and tokens[-1] == runner.eos_token_id
                if request.take_result() != tokens or (
                    len(tokens) != generated and not ended_at_eos
                ):
                    raise AssertionError(
                        "generation did not produce the requested scored tokens"
                    )
                values = request.prompt_logprobs
                if (
                    values is None
                    or len(values) != len(prompt)
                    or values[0] is not None
                ):
                    raise AssertionError(
                        "prompt scores lost input positions or initial None"
                    )
                prompt_scores = [None]
                for value in values[1:]:
                    if value is None:
                        raise AssertionError(
                            "a prompt token is missing its predecessor score"
                        )
                    prompt_scores.append(finite(value.logprob))
                if request.cached_tokens:
                    raise AssertionError(
                        "the scored validation prefix was unexpectedly reused"
                    )
        finally:
            with _validation_stage(stage + " cleanup", coordinate):
                if not request.done and not getattr(local, "poisoned", False):
                    engine.cancel(request)
        with _validation_stage(stage + " graphs", coordinate):
            after = counts()
            graphs = {mode: after[mode] - before[mode] for mode in before}
            if eager and any(graphs.values()):
                raise AssertionError(
                    f"eager reference replayed captured work: {graphs}"
                )
            if not eager and (
                graphs.get("decode", 0) != steps["decode"]
                or steps["decode"] < 3
                or not graphs.get("prefill", 0)
            ):
                reason = (
                    "validation answer reached EOS before three decode steps"
                    if ended_at_eos and steps["decode"] < 3
                    else "captured workload did not replay its prefill/decode graphs"
                )
                raise AssertionError(f"{reason}: {graphs}, {steps}")
            speculation = runner._speculation.snapshot()
            return {
                "token_ids": tokens,
                "text": tokenizer.decode(tokens, skip_special_tokens=False),
                "logprobs": scores,
                "top_logprobs": tops,
                "prompt_logprobs": prompt_scores,
                "finished_by": "eos" if ended_at_eos else "length",
                "steps": steps,
                "graph_replays": graphs,
                "speculation": {
                    key: speculation[key] - accepted[key]
                    for key in ("blocks", "drafts", "accepted", "emitted")
                },
            }

    try:
        eager, captured = generate(True), generate(False)
    finally:
        runner.cuda_graph_prefill, runner.cuda_graph_decode = policies
    if eager["token_ids"] != captured["token_ids"]:
        import json

        index = next(
            (
                i
                for i, pair in enumerate(zip(eager["token_ids"], captured["token_ids"]))
                if pair[0] != pair[1]
            ),
            min(len(eager["token_ids"]), len(captured["token_ids"])),
        )
        difference = {"index": index}
        for name, output in (("eager", eager), ("captured", captured)):
            if index < len(output["token_ids"]):
                top = output["top_logprobs"][index]
                difference[name] = {
                    "token_id": output["token_ids"][index],
                    "top_logprobs": top,
                    "top1_top2_margin": top[0][1] - top[1][1],
                }
            else:
                difference[name] = {"finished_by": output["finished_by"]}
        raise AssertionError(
            "eager/captured greedy tokens differ at a shared prefix: "
            + json.dumps(difference)
        )
    selected = _check(
        torch.tensor(captured["logprobs"]),
        torch.tensor(eager["logprobs"]),
        atol=0.05,
        rtol=0.01,
    )
    prompt_scores = _check(
        torch.tensor(captured["prompt_logprobs"][1:]),
        torch.tensor(eager["prompt_logprobs"][1:]),
        atol=0.05,
        rtol=0.01,
    )
    overlap = (
        validate_overlap(runner, engine, prompt, coordinate=coordinate)
        if getattr(local, "overlap", False)
        else None
    )
    return {
        "ok": True,
        "prompt_tokens": len(prompt),
        "prompt_token_ids": prompt,
        "prompt_text": tokenizer.decode(prompt, skip_special_tokens=False),
        "speculative_tokens": width,
        "prefill_chunk_size": runner.prefill_chunk_size,
        "eager": eager,
        "captured": captured,
        "selected_comparison": selected,
        "prompt_comparison": prompt_scores,
        "tolerance": {"atol": 0.05, "rtol": 0.01},
        **({"overlap": overlap} if overlap is not None else {}),
    }
