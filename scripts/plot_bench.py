"""Plot Resin JSONL sweeps as decode and prefill throughput/latency curves."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

FIXED_METADATA = (
    "model",
    "checkpoint_revision",
    "git_commit",
    "source_digest",
    "hardware",
    "parallelism",
    "graphs",
    "quant",
    "expert_backend",
)
FIXED_CASE = (
    "draft_tokens",
    "accepted_drafts",
    "balanced_experts",
    "shared_prefix_tokens",
)


def _number(value, name, *, integer=False, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(
            f"{name} must be a finite {'integer' if integer else 'number'} >= {minimum}"
        )
    return value


def _validate(record):
    if not isinstance(record, dict):
        raise ValueError("benchmark record must be a JSON object")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("expected benchmark schema_version=1")
    if record.get("phase") not in {"decode", "prefill"}:
        raise ValueError("phase must be decode or prefill")
    for name in (
        "context_length",
        "batch_size",
        "global_batch_size",
        "gpu_count",
        "attn_dp",
        "measure_steps",
    ):
        _number(record[name], name, integer=True, minimum=1)
    for name in ("draft_tokens", "warmup_steps", "input_tokens", "committed_tokens"):
        _number(record[name], name, integer=True)
    shared = _number(
        record.setdefault("shared_prefix_tokens", 0),
        "shared_prefix_tokens",
        integer=True,
    )
    if shared >= record["context_length"]:
        raise ValueError("shared_prefix_tokens must be less than context_length")
    if shared and any(
        record.get(name) != shared
        for name in ("cached_tokens_min", "cached_tokens_max")
    ):
        raise ValueError("shared-prefix records must verify exact cached token counts")
    _number(record["prefill_tokens"], "prefill_tokens")
    _number(record["window_seconds"], "window_seconds", minimum=math.nextafter(0, 1))
    steps = record["step_seconds"]
    if not isinstance(steps, list) or len(steps) != record["measure_steps"]:
        raise ValueError("step_seconds must contain measure_steps measurements")
    for value in steps:
        _number(value, "step_seconds", minimum=math.nextafter(0, 1))
    accepted = record["accepted_drafts"]
    if (
        accepted is not None
        and _number(accepted, "accepted_drafts", integer=True) > record["draft_tokens"]
    ):
        raise ValueError("accepted_drafts must not exceed draft_tokens")
    measured = record["measured_accept_rate"]
    if measured is not None and _number(measured, "measured_accept_rate") > 1:
        raise ValueError("measured_accept_rate must be between 0 and 1")
    if type(record["balanced_experts"]) is not bool:
        raise ValueError("balanced_experts must be a boolean")
    if record["global_batch_size"] != record["batch_size"] * record["attn_dp"]:
        raise ValueError("global_batch_size must equal batch_size * attn_dp")
    metadata = record["metadata"]
    if isinstance(metadata, dict):
        # Earlier schema-1 measurements had only the FlashInfer expert path.
        metadata.setdefault("expert_backend", "flashinfer")
    if not isinstance(metadata, dict) or any(
        name not in metadata for name in FIXED_METADATA
    ):
        raise ValueError(f"metadata must include {', '.join(FIXED_METADATA)}")
    topology = metadata["parallelism"]
    if (
        not isinstance(topology, dict)
        or topology.get("world_size") != record["gpu_count"]
        or topology.get("attn_dp") != record["attn_dp"]
    ):
        raise ValueError("metadata.parallelism must agree with gpu_count and attn_dp")
    if record["phase"] == "prefill":
        if record["input_tokens"] <= 0 or not math.isclose(
            record["prefill_tokens"], record["input_tokens"] / len(steps), rel_tol=1e-9
        ):
            raise ValueError(
                "prefill_tokens must be the measured mean input tokens per step"
            )
    elif (
        record["input_tokens"]
        or record["prefill_tokens"]
        or not record["committed_tokens"]
    ):
        raise ValueError(
            "decode requires positive committed_tokens and zero prefill input tokens"
        )


def load_records(paths: list[Path]) -> list[dict]:
    """Validate every row before selecting the latest repeat of a sweep point."""
    records = []
    configs = {}
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                _validate(record)
                fixed = {name: record[name] for name in FIXED_CASE}
                fixed.update(
                    {name: record["metadata"][name] for name in FIXED_METADATA}
                )
                fixed["prefill_chunk_size"] = record["metadata"].get(
                    "prefill_chunk_size"
                )
                fixed["max_batch_tokens"] = record["metadata"].get("max_batch_tokens")
                # Context is the only fixed configuration allowed to form separate curves.
                fixed.update(gpu_count=record["gpu_count"], attn_dp=record["attn_dp"])
                signature = json.dumps(fixed, sort_keys=True)
                previous = configs.setdefault(record["phase"], signature)
                if signature != previous:
                    raise ValueError(
                        "mixed model/topology/graphs/quant/kernels/balance/acceptance configurations; use separate files"
                    )
                records.append(record)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not records:
        raise ValueError("no benchmark records found")
    latest = {}
    for record in records:
        key = (
            record["phase"],
            record["context_length"],
            record["batch_size"],
            record["prefill_tokens"],
        )
        latest[key] = record
    return list(latest.values())


def axes(records, phase, *, draft_tokens=4, accept_rate=0.75):
    """Normalize measured engine steps; decode acceptance is an explicit projection."""
    _number(draft_tokens, "draft_tokens", integer=True)
    if _number(accept_rate, "accept_rate") > 1:
        raise ValueError("accept_rate must be between 0 and 1")
    result = {}
    for record in sorted(
        records,
        key=lambda r: (r["context_length"], r["prefill_tokens"], r["batch_size"]),
    ):
        if record["phase"] != phase:
            continue
        seconds, steps = record["window_seconds"], len(record["step_seconds"])
        if phase == "decode":
            if record["draft_tokens"] != draft_tokens:
                raise ValueError(
                    "--draft-tokens must match the measured decode draft_tokens"
                )
            speed = steps / seconds * (1 + draft_tokens * accept_rate)
            point = (
                speed,
                speed * record["global_batch_size"] / record["gpu_count"],
                f"{record['batch_size']:,}",
            )
        else:
            batch = record["prefill_tokens"]
            label = f"{batch:,.0f}" if batch == int(batch) else f"{batch:,.1f}"
            point = (
                seconds / steps * 1000,
                record["input_tokens"] / seconds / record["gpu_count"],
                label,
            )
        result.setdefault(record["context_length"], []).append(point)
    if not result:
        raise ValueError(f"no {phase} measurements found")
    return result


def plot(records, phase, output_dir, *, draft_tokens=4, accept_rate=0.75):
    curves = axes(records, phase, draft_tokens=draft_tokens, accept_rate=accept_rate)
    # Loading/validating JSONL and --help do not need Matplotlib installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    selected = [record for record in records if record["phase"] == phase]
    first = selected[0]
    metadata = first["metadata"]
    topology = metadata["parallelism"]
    mesh = " ".join(
        f"{name}={topology[name]}"
        for name in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
        if name in topology
    )
    graphs = ", ".join(
        f"{name}: {policy}" for name, policy in metadata["graphs"].items()
    )
    subtitle = f"{metadata['model']} | {mesh} | quant={metadata['quant']} | experts={metadata['expert_backend']} | graphs={graphs} | balanced experts={first['balanced_experts']}"
    if shared := first.get("shared_prefix_tokens", 0):
        subtitle += f" | shared prefix={shared:,} tokens"
    figure, axis = plt.subplots(figsize=(9, 6.4))
    try:
        for curve_index, (context, points) in enumerate(sorted(curves.items())):
            xs, ys, labels = zip(*points, strict=True)
            (line,) = axis.plot(xs, ys, "o-", linewidth=2, label=f"Context {context:,}")
            for x, y, label in points:
                axis.annotate(
                    label,
                    (x, y),
                    xytext=(0, 9 if curve_index % 2 == 0 else -16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=line.get_color(),
                )
        decode = phase == "decode"
        figure.suptitle(
            "Decode: output throughput vs. interactivity"
            if decode
            else "Prefill: input throughput vs. latency",
            x=0.08,
            ha="left",
            fontsize=15,
            weight="bold",
        )
        axis.set_title(
            textwrap.fill(subtitle, 105),
            loc="left",
            fontsize=9,
            color="#555555",
            pad=12,
        )
        axis.set_xlabel(
            "Per-user output (tokens/s)" if decode else "Prefill latency (ms/step)"
        )
        axis.set_ylabel("Output tokens/s/GPU" if decode else "Input tokens/s/GPU")
        axis.set_xlim(0, axis.dataLim.xmax * 1.12)
        axis.set_ylim(0, axis.dataLim.ymax * 1.2)
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=9)
        if decode:
            measured = [
                r["measured_accept_rate"]
                for r in selected
                if r["measured_accept_rate"] is not None
            ]
            observed = (
                "unavailable"
                if not measured
                else f"{min(measured):.1%}–{max(measured):.1%}"
            )
            fixed = (
                "natural"
                if first["accepted_drafts"] is None
                else f"fixed {first['accepted_drafts']}/{first['draft_tokens']}"
            )
            note = f"Projected acceptance {accept_rate:.0%}, {draft_tokens} drafts → τ={1 + draft_tokens * accept_rate:g}; measured acceptance {observed} ({fixed}).\nPoint labels: requests per attention-DP replica."
            if not draft_tokens:
                note = "Ordinary decode (no drafts), τ=1.\nPoint labels: requests per attention-DP replica."
        else:
            note = "Point labels: measured unique input tokens per global prefill step."
        figure.text(0.5, 0.025, note, ha="center", fontsize=9, color="#555555")
        figure.tight_layout(rect=(0, 0.09, 1, 0.94))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = tuple(output_dir / f"{phase}.{suffix}" for suffix in ("png", "svg"))
        for path in paths:
            figure.savefig(path, dpi=180)
        return paths
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="benchmark JSONL files with matching fixed configurations",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("both", "decode", "prefill"), default="both")
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--accept-rate", type=float, default=0.75)
    args = parser.parse_args(argv)
    try:
        records = load_records(args.results)
        phases = ("decode", "prefill") if args.mode == "both" else (args.mode,)
        output = args.output_dir or args.results[0].parent / "plots"
        for phase in phases:
            for path in plot(
                records,
                phase,
                output,
                draft_tokens=args.draft_tokens,
                accept_rate=args.accept_rate,
            ):
                print(path.resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
