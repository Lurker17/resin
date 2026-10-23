"""Checkpoint mask/compact qualification using actual Engine-owned cache leases.

Use ProbeRunner as deploy.bench._rank's runner_factory implementation and pass
run_validation as its validation callback. Enable prefix caching and admit two
requests per driver. Paired modes require ADP=world; CP uses separate runs with
one constructor-fixed mode and compares the exported scores/state hashes.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch

from deploy.deepseek_v41_validation import _validation_stage
from resin import SamplingParams
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner


def _sample_positions(end, *, start=0):
    if end <= start:
        return []
    # Cover the oldest keys, both sides of Engine pages, and the live tail.
    values = {start, end - 1, max(start, end - 2)}
    for point in (1, 127, 128, 129, end // 2, end - 128, end - 127):
        if start <= point < end:
            values.add(point)
    return sorted(values)


def _cache_snapshot(runner, cache, end, replay_start):
    """Canonical logical samples, independent of allocator page placement."""
    from resin_models_cuda.deepseek_v41.attention import gather_kv

    device = cache.storage.device
    result = {}
    for source in cache.geometry.kv_sources:
        ratio = cache.geometry.ratios[source]
        logical = (
            torch.tensor(
                _sample_positions(end // ratio), device=device, dtype=torch.long
            )[None]
            * ratio
        )
        rows = cache.compressed_rows(source, ratio, logical).flatten().long()
        values = cache.compressed_cache(source)
        result[f"main.{source}"] = values[
            rows // values.shape[1], rows % values.shape[1], 0
        ].cpu()
        values, scales = cache.index_parts(source)
        result[f"index.{source}.values"] = values[
            rows // values.shape[1], rows % values.shape[1]
        ].cpu()
        result[f"index.{source}.scales"] = scales[
            rows // values.shape[1], rows % values.shape[1]
        ].cpu()
    tail = torch.arange(max(0, end - 2), end, device=device)[None]
    for source in cache.geometry.compressor_sources:
        values, scores = cache.compressor_rows(source, tail)
        result[f"compressor.{source}.values"] = cache.compressor_values[
            values.long()
        ].cpu()
        result[f"compressor.{source}.scores"] = cache.compressor_values[
            scores.long()
        ].cpu()
    boundary = runner._model.prefill_boundary
    for layer in range(cache.geometry.attention_layers):
        start = max(0, end - cache.geometry.window_size)
        if boundary is not None and layer >= boundary:
            start = max(start, replay_start)
        logical = torch.tensor(
            _sample_positions(end, start=start), device=device, dtype=torch.long
        )[None]
        rows = cache.window_rows(layer, logical)
        result[f"window.{layer}"] = gather_kv(cache.window_cache(layer), rows).cpu()
    return result


class ProbeRunner(DeepSeekV41CudaRunner):
    """Observe the production row path after its normal DSpark cache commit."""

    _yoco_rows: list[int]
    _yoco_snapshots: list[dict]

    def _prefill_row(self, plan, slot, input_ids, positions, cache):
        before = len(getattr(self, "_yoco_rows", ()))
        output = super()._prefill_row(plan, slot, input_ids, positions, cache)
        if getattr(self, "_yoco_observe", False):
            endpoint = int(plan.context_lengths[slot]) + input_ids.shape[1]
            prompt = int(plan.prefill_lengths[slot])
            lower = (
                0
                if plan.needs_prompt_logprobs
                else max(0, prompt - self._swa_replay_tokens)
            )
            self._yoco_snapshots.append(
                dict(
                    prompt=prompt,
                    endpoint=endpoint,
                    input_rows=input_ids.shape[1],
                    consumer_rows=self._yoco_rows[before:],
                    prompt_scores=plan.needs_prompt_logprobs,
                    state=_cache_snapshot(self, cache, endpoint, lower),
                )
            )
        return output


def _tensor_report(actual, expected, *, atol=0, rtol=0, label="tensor"):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    if actual.numel():
        difference = actual.double() - expected.double()
        maximum = difference.abs().max().item()
        relative = (
            difference.norm() / expected.double().norm().clamp_min(1e-20)
        ).item()
    else:
        maximum = relative = 0.0
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(
            f"{label}: max_abs={maximum}, relative_l2={relative}: {error}"
        ) from error
    return dict(max_abs=maximum, relative_l2=relative, values=actual.numel())


def _state_hashes(snapshot):
    return {
        key: hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for key, value in snapshot.items()
    }


def _compare(left, right, *, atol=0, rtol=0, label="case"):
    assert left["requests"].keys() == right["requests"].keys()
    scores = {}
    for key, a in left["requests"].items():
        b = right["requests"][key]
        assert a["tokens"] == b["tokens"], f"{label} token mismatch for {key}"
        assert a["top_ids"] == b["top_ids"], f"{label} top token mismatch for {key}"
        assert a["cached_tokens"] == b["cached_tokens"]
        scores[str(key)] = _tensor_report(
            torch.tensor(a["scores"], dtype=torch.float64),
            torch.tensor(b["scores"], dtype=torch.float64),
            atol=atol,
            rtol=rtol,
            label=f"{label} request {key} scores",
        )
    assert left["snapshots"].keys() == right["snapshots"].keys()
    state = {}
    for key, a in left["snapshots"].items():
        b = right["snapshots"][key]
        assert a["state"].keys() == b["state"].keys()
        state[str(key)] = {
            name: _tensor_report(
                value,
                b["state"][name],
                label=f"{label} snapshot {key} {name}",
                **({} if not value.is_floating_point() else dict(atol=atol, rtol=rtol)),
            )
            for name, value in a["state"].items()
        }
    return dict(scores=scores, state=state)


def _public(result):
    return {
        **{key: value for key, value in result.items() if key != "snapshots"},
        "snapshots": [
            dict(
                **{name: value for name, value in snapshot.items() if name != "state"},
                state_sha256=_state_hashes(snapshot["state"]),
            )
            for key, snapshot in result["snapshots"].items()
        ],
    }


def run_probe(
    runner,
    engine,
    prompt_tokens,
    *,
    coordinate=None,
    modes=None,
    generated=160,
    atol=0,
    rtol=0,
):
    """Qualify fresh, restored, scored, unequal/idle, and natural decode cohorts.

    Paired mask/compact comparisons are exact by default. Report wall time only
    as qualification time: these runs deliberately copy small state samples.
    """
    local = getattr(engine, "local", engine)
    groups = runner.parallel_groups
    world = 1 if groups is None else groups.topology.world_size
    driver = 0 if groups is None else groups.drivers.local_rank
    modes = tuple(modes or ("mask", "compact"))
    if len(modes) > 1 and groups is not None and groups.drivers.size != world:
        raise ValueError("CP/TP qualification needs constructor-fixed mode runs")
    if (
        groups is not None
        and groups.drivers.size != world
        and modes != (runner._swa_replay_mode,)
    ):
        raise ValueError("CP/TP mode must match every runner's constructor mode")
    if not isinstance(runner, ProbeRunner) or runner.max_batch_size < 2:
        raise ValueError(
            "YOCO qualification requires ProbeRunner with max_batch_size>=2"
        )
    if any(
        getattr(module, "benchmark_balanced_experts", False)
        for module in runner._model.modules()
    ):
        raise ValueError("mask/compact qualification requires natural expert routing")
    if runner._speculation.benchmark_accepted_drafts is not None:
        raise ValueError("mask/compact qualification requires natural acceptance")
    if runner._swa_replay_tokens != 140 or any(
        mode not in ("mask", "compact") for mode in modes
    ):
        raise ValueError("qualification requires mask/compact with140 retained tokens")
    width = runner.prefill_chunk_size
    if width < 256:
        raise ValueError("qualification requires a prefill chunk of at least256 tokens")
    # A capped restore must still include a completed chunk checkpoint. The
    # second prompt leaves more than140 tokens after its first full chunk.
    lengths = (2 * width + 129, width + 257)
    if max(lengths) + generated + runner.speculative_tokens >= local.context_length:
        raise ValueError(
            "qualification context must fit two chunks, page tail and160 decode tokens"
        )
    if not prompt_tokens:
        raise ValueError("qualification needs reproducible nonempty prompt tokens")
    prompts = {
        length: (list(prompt_tokens) * math.ceil(length / len(prompt_tokens)))[:length]
        for length in lengths
    }
    boundary = runner._model.layers[runner._model.prefill_boundary]
    runner._yoco_rows, runner._yoco_snapshots = [], []
    hook = boundary.register_forward_pre_hook(
        lambda _module, inputs: runner._yoco_rows.append(inputs[0].shape[1])
    )
    original_mode = runner._swa_replay_mode

    def cohort(name, chosen, *, restored=False, scored=False, overlap=False):
        runner._yoco_rows.clear()
        runner._yoco_snapshots.clear()
        runner._yoco_observe = True
        requests = {}
        before = local.overlap_stats
        speculation_before = runner._speculation.snapshot()
        with _validation_stage(name + " admission", coordinate):
            for length in chosen:
                request = engine.submit(
                    prompts[length],
                    max_new_tokens=generated,
                    stop_token_ids=[],
                    sampling_params=SamplingParams(
                        temperature=0,
                        seed=583,
                        logprobs=not overlap,
                        top_logprobs=0 if overlap else 3,
                        prompt_logprobs=scored,
                    ),
                )
                requests[request.id] = (length, request)
        records = {length: dict(tokens=[], scores=[], top_ids=[]) for length in chosen}
        steps, started = 0, time.perf_counter()
        try:
            while engine.has_work():
                events = engine.step(allow_overlap=overlap)
                steps += 1
                with _validation_stage(name + " events", coordinate):
                    for event in events:
                        length, _ = requests[event.request_id]
                        record = records[length]
                        assert event.index == len(record["tokens"])
                        record["tokens"].append(event.token_id)
                        if not overlap:
                            assert event.logprobs is not None
                            record["scores"].append(float(event.logprobs.logprob))
                            record["top_ids"].append(
                                [int(token) for token, _ in event.logprobs.top_logprobs]
                            )
                            record["scores"].extend(
                                float(score) for _, score in event.logprobs.top_logprobs
                            )
                    assert steps <= (max(lengths) // width + generated + 8)
            with _validation_stage(name + " result", coordinate):
                for length, request in requests.values():
                    record = records[length]
                    assert len(record["tokens"]) == generated
                    assert request.take_result() == record["tokens"]
                    record["cached_tokens"] = request.cached_tokens
                    if restored:
                        assert 0 < request.cached_tokens <= length - 140
                    else:
                        assert request.cached_tokens == 0
                    if scored:
                        values = request.prompt_logprobs
                        assert (
                            values is not None
                            and len(values) == length
                            and values[0] is None
                        )
                        assert all(value is not None for value in values[1:])
                        record["scores"].extend(
                            float(value.logprob) for value in values[1:]
                        )
                for snapshot in runner._yoco_snapshots:
                    context = snapshot["endpoint"] - snapshot["input_rows"]
                    expected = (
                        snapshot["input_rows"]
                        if (scored or runner._swa_replay_mode == "mask")
                        else max(
                            1,
                            snapshot["endpoint"]
                            - max(context, snapshot["prompt"] - 140),
                        )
                    )
                    assert snapshot["consumer_rows"] == [expected]
                after = local.overlap_stats
                speculation_after = runner._speculation.snapshot()
                if overlap and chosen:
                    assert after[1] > before[1], "no real GPU continuation occurred"
                return dict(
                    requests=records,
                    steps=steps,
                    qualification_seconds=time.perf_counter() - started,
                    enqueues=after[0] - before[0],
                    continuations=after[1] - before[1],
                    speculation={
                        key: speculation_after[key] - speculation_before[key]
                        for key in ("blocks", "drafts", "accepted", "emitted")
                    },
                    snapshots={
                        (row["prompt"], row["endpoint"]): row
                        for row in runner._yoco_snapshots
                    },
                )
        finally:
            runner._yoco_observe = False
            with _validation_stage(name + " cleanup", coordinate):
                for _, request in requests.values():
                    if not request.done and not getattr(local, "poisoned", False):
                        engine.cancel(request)

    results = {}
    try:
        for mode in modes:
            with _validation_stage(mode + " setup", coordinate):
                assert not local.has_work()
                runner._swa_replay_mode = mode
                local.clear_prefix_cache()
            results[mode] = {}
            fresh = cohort(mode + " fresh", lengths)
            results[mode]["fresh"] = fresh
            restored = cohort(mode + " restored", lengths, restored=True)
            results[mode]["restored"] = restored
            # Fresh/restored follow the same logical contexts and must agree.
            with _validation_stage(mode + " restored parity", coordinate):
                for length in lengths:
                    assert (
                        fresh["requests"][length]["tokens"]
                        == restored["requests"][length]["tokens"]
                    ), f"{mode} fresh/restored prompt {length} tokens"
                    assert (
                        fresh["requests"][length]["top_ids"]
                        == restored["requests"][length]["top_ids"]
                    ), f"{mode} fresh/restored prompt {length} top token IDs"
                    _tensor_report(
                        torch.tensor(fresh["requests"][length]["scores"]),
                        torch.tensor(restored["requests"][length]["scores"]),
                        atol=atol,
                        rtol=rtol,
                        label=f"{mode} fresh/restored prompt {length} scores",
                    )
                    a = fresh["snapshots"][(length, length)]["state"]
                    b = restored["snapshots"][(length, length)]["state"]
                    for name, value in a.items():
                        _tensor_report(
                            value,
                            b[name],
                            label=f"{mode} fresh/restored prompt {length} {name}",
                            **(
                                {}
                                if not value.is_floating_point()
                                else dict(atol=atol, rtol=rtol)
                            ),
                        )
                local.clear_prefix_cache()
            scored = cohort(mode + " scored fallback", (lengths[1],), scored=True)
            results[mode]["scored"] = scored
            with _validation_stage(mode + " unequal setup", coordinate):
                local.clear_prefix_cache()
            count = 2 if driver == 0 else 1 if driver == 1 else 0
            unequal = cohort(
                mode + " unequal idle overlap", lengths[:count], overlap=True
            )
            results[mode]["unequal"] = unequal
        comparisons = {}
        if len(modes) == 2:
            with _validation_stage("mask compact comparisons", coordinate):
                comparisons = {
                    name: _compare(
                        results[modes[0]][name],
                        results[modes[1]][name],
                        atol=atol,
                        rtol=rtol,
                        label=name,
                    )
                    for name in results[modes[0]]
                }
        return dict(
            ok=True,
            modes=modes,
            retained_tokens=140,
            generated=generated,
            comparisons=comparisons,
            results={
                mode: {name: _public(value) for name, value in rows.items()}
                for mode, rows in results.items()
            },
        )
    except Exception as error:
        raise ProbeFailure(
            str(error),
            {
                mode: {name: _public(value) for name, value in rows.items()}
                for mode, rows in results.items()
            },
        ) from error
    finally:
        runner._swa_replay_mode = original_mode
        runner._yoco_observe = False
        hook.remove()


class ProbeFailure(RuntimeError):
    def __init__(self, message, partial_results):
        super().__init__(message)
        self.partial_results = partial_results


class ValidationFailure(RuntimeError):
    def __init__(self, records):
        ranks = records[0]["ranks"]
        super().__init__(
            "YOCO checkpoint qualification failed: "
            + str(
                [(rank["rank"], rank.get("error")) for rank in ranks if not rank["ok"]]
            )
        )
        self.validation_records = records


def run_validation(raw, engine, checkpoint, metadata, groups):
    """deploy.bench._rank callback; each attention-DP driver joins each stage."""
    from tokenizers import Tokenizer

    from resin.models.deepseek_v41.codec import DeepSeekV41Codec

    rank = 0 if groups is None else groups.rank
    drivers = None if groups is None else groups.drivers

    def gather(stage, value):
        if drivers is None or drivers.size == 1:
            return [value]
        values = [None] * drivers.size
        groups.distributed.all_gather_object(
            values, (stage, value), group=drivers.handle
        )
        assert all(row is not None and row[0] == stage for row in values)
        return [row[1] for row in values if row is not None]

    def coordinate(stage, error):
        errors = gather(
            stage, None if error is None else f"{type(error).__name__}: {error}"
        )
        if any(error is not None for error in errors):
            raise RuntimeError(f"YOCO validation {stage}: {errors}")

    try:
        recipe = DeepSeekV41Codec(
            Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
        )
        prompt = recipe.render_ids(
            [dict(role="user", content="Count from1 to20, one number per line.")]
        )
        modes = (
            None
            if groups is None or groups.drivers.size == groups.topology.world_size
            else (raw._swa_replay_mode,)
        )
        result = dict(
            rank=rank,
            **run_probe(raw, engine, prompt, coordinate=coordinate, modes=modes),
        )
    except Exception as error:
        result = dict(
            rank=rank,
            ok=False,
            error=f"{type(error).__name__}: {error}",
            partial_results=getattr(error, "partial_results", {}),
        )
    reports = gather("reports", result)
    records = [
        dict(
            schema_version=1,
            phase="validation",
            validation="bounded_swa_replay",
            metadata=metadata,
            ranks=reports,
        )
    ]
    if any(not row["ok"] for row in reports):
        raise ValidationFailure(records)
    return records


def compare_fixed_mode_reports(mask, compact, *, atol=0, rtol=0):
    """Compare exported constructor-fixed CP runs without loading a checkpoint."""
    assert len(mask) == len(compact) == 1
    mask, compact = mask[0], compact[0]
    for field in (
        "git_commit",
        "source_digest",
        "checkpoint_revision",
        "parallelism",
        "expert_backend",
        "quant",
        "hardware",
    ):
        assert mask["metadata"][field] == compact["metadata"][field], field
    left = {rank["rank"]: rank for rank in mask["ranks"]}
    right = {rank["rank"]: rank for rank in compact["ranks"]}
    assert left.keys() == right.keys()
    compared = 0
    for rank, a in left.items():
        b = right[rank]
        assert a["ok"] and b["ok"]
        assert tuple(a["modes"]) == ("mask",) and tuple(b["modes"]) == ("compact",)
        assert (
            a["generated"] == b["generated"]
            and a["retained_tokens"] == b["retained_tokens"] == 140
        )
        for case, reference in a["results"]["mask"].items():
            candidate = b["results"]["compact"][case]
            assert reference["requests"].keys() == candidate["requests"].keys()
            for request, expected in reference["requests"].items():
                actual = candidate["requests"][request]
                for field in ("tokens", "top_ids", "cached_tokens"):
                    assert expected[field] == actual[field], (
                        rank,
                        case,
                        request,
                        field,
                    )
                _tensor_report(
                    torch.tensor(actual["scores"], dtype=torch.float64),
                    torch.tensor(expected["scores"], dtype=torch.float64),
                    atol=atol,
                    rtol=rtol,
                    label=f"rank {rank} {case} request {request} scores",
                )
            states = {
                (row["prompt"], row["endpoint"]): row["state_sha256"]
                for row in reference["snapshots"]
            }
            next_states = {
                (row["prompt"], row["endpoint"]): row["state_sha256"]
                for row in candidate["snapshots"]
            }
            assert states == next_states, (rank, case, "canonical cache state hashes")
            compared += sum(len(values) for values in states.values())
    return dict(ok=True, ranks=sorted(left), exact_state_hashes=compared)
