"""Run resident prefill/decode benchmarks locally or on Modal.

uv run scripts/bench.py --model deepseek-v41 --backend modal --output bench/v41.jsonl
The measurement API lives in resin.bench; model definitions are shared with serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

# Direct script execution must also make the repository packages importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

import resin
from deploy.models import MODEL_SPECS, create_runner, get_model, validate_gpu
from resin import Engine, Parallelism, SchedulerConfig
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups
from resin.bench import perfetto_trace

ROOT = Path(__file__).resolve().parents[1]


def _runner(model, checkpoint, *, device, groups, cases, options, context_capacity):
    spec = get_model(model)
    batches = sorted({case.batch_size for case in cases})
    # GPT-OSS sizes its flattened input buffer at construction; other runners
    # own their input budget and expose it through runner_config.
    chunk = options["prefill_chunk_size"]
    return create_runner(
        spec,
        checkpoint,
        device=device,
        groups=groups,
        context_length=context_capacity,
        max_batch_size=max(batches),
        prefill_chunk_size=chunk * max(batches) if spec.family == "gpt_oss" else chunk,
        draft_tokens=options["draft_tokens"],
        prefill_graph=options["prefill_graph"],
        decode_graph=options["decode_graph"],
        gpu_memory_utilization=options["gpu_memory_utilization"],
        decode_batch_sizes=batches,
        expert_backend=options.get("expert_backend"),
        profile_graph_phases=options.get("profile_graph_phases", False),
        assistant_checkpoint=options.get("assistant_checkpoint"),
    )


def _result_callback(report_path):
    """Keep a valid snapshot of completed cases and print untimed progress."""
    records = []
    path = Path(report_path)

    def save(record):
        records.append(record)
        pending = path.with_suffix(path.suffix + ".tmp")
        pending.write_text(json.dumps(records))
        pending.replace(path)
        print("[bench-result] " + json.dumps(record), flush=True)

    return save


def _cache_preflight(raw, cases, options, groups=None):
    """Reject a partial decode cohort before spending time on capture or prefill."""
    plan = getattr(raw, "cache_capacity_plan", None)
    if plan is None:
        return
    layout = raw.cache_layout
    requirements = []
    for case in cases:
        context = case.context_length
        if case.phase == "decode":
            context += (
                options["warmup_steps"]
                + options["measure_steps"]
                + 1
                + (2 if options.get("trace_dir") else 0)
            ) * (options["draft_tokens"] + 1) + options["draft_tokens"]
        parents = layout.estimated_parent_count(
            [context] * case.batch_size, shared_prefix_tokens=case.shared_prefix_tokens
        )
        requirements.append(
            dict(
                **asdict(case),
                peak_context=context,
                required_parents=parents,
            )
        )
    rank = 0 if groups is None else groups.rank
    print(
        "[bench-cache] "
        + json.dumps(
            dict(
                rank=rank,
                capacity=asdict(plan),
                cases=requirements,
            )
        ),
        flush=True,
    )
    failures = [
        f"rank {rank} {row['phase']} ctx={row['context_length']} "
        f"batch={row['batch_size']} needs {row['required_parents']} cache parents, "
        f"capacity={layout.parent_capacity} (peak context {row['peak_context']})"
        for row in requirements
        if row["required_parents"] > layout.parent_capacity
    ]
    errors = [failures]
    if groups is not None and groups.domain.size > 1:
        errors = [None] * groups.domain.size
        groups.distributed.all_gather_object(
            errors, failures, group=groups.domain.handle
        )
    failures = [failure for values in errors if values for failure in values]
    if failures:
        raise ValueError("benchmark cache preflight failed: " + "; ".join(failures))


def _rank(
    rank,
    port,
    report_path,
    model,
    checkpoint,
    case_values,
    topology,
    options,
):
    from resin.bench import (
        BenchCase,
        BenchmarkRunner,
        distributed_reducer,
        run_sweep,
    )

    if model.startswith("gpt-oss"):
        from deploy.jit_cache import KERNEL_CACHE_MOUNT, install_flashinfer_hash_cache

        # The durable Modal cache owns a fixed mounted path. Local GPU runs use
        # the user's ordinary FlashInfer cache unless that mount is configured.
        if os.environ.get("CUDA_CACHE_PATH", "").startswith(KERNEL_CACHE_MOUNT + "/"):
            install_flashinfer_hash_cache()
    torch.cuda.set_device(rank)
    groups = None
    if topology.world_size > 1:
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=topology.world_size,
            device_id=torch.device("cuda", rank),
            timeout=timedelta(minutes=30),
        )
        groups = ParallelGroups.create(topology)
    cases = [BenchCase(**case) for case in case_values]
    prefill_budget = (
        max(case.batch_size for case in cases) * options["prefill_chunk_size"]
    )
    # Reserve real generated positions, including draft write lookahead. Context
    # lengths on the charts remain the prompt lengths recorded by the harness.
    headroom = (options["warmup_steps"] + options["measure_steps"] + 4) * (
        options["draft_tokens"] + 1
    )
    capacity = max(case.context_length for case in cases) + headroom
    if rank == 0:
        print(f"[bench] loading {model} over {topology.world_size} GPUs", flush=True)
    started = time.perf_counter()
    raw = _runner(
        model,
        checkpoint,
        device=f"cuda:{rank}",
        groups=groups,
        cases=cases,
        options=options,
        context_capacity=capacity,
    )
    # A batch is a cohort per attention replica; never exceed the runner's
    # supported flattened row count when prefill spans several requests.
    prefill_budget = min(prefill_budget, raw.runner_config.max_batch_tokens)
    _cache_preflight(raw, cases, options, groups)
    observed = BenchmarkRunner(
        raw,
        accepted_drafts=options["accepted_drafts"],
        balanced_experts=options["balanced_experts"],
    )
    torch.cuda.synchronize()
    loaded_seconds = time.perf_counter() - started
    if rank == 0:
        print(
            f"[bench] weights ready in {loaded_seconds:.1f}s; capturing graphs",
            flush=True,
        )
    started = time.perf_counter()
    raw.capture_cuda_graphs()
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - started
    if rank == 0:
        print(f"[bench] graphs ready in {capture_seconds:.1f}s", flush=True)
    overlap = options.get("overlap", True) and callable(getattr(raw, "enqueue", None))
    metadata = {
        **options["metadata"],
        "model": model,
        "checkpoint": str(checkpoint),
        "hardware": torch.cuda.get_device_name(rank),
        "parallelism": {
            axis: getattr(topology, axis)
            for axis in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
        },
        "graphs": {
            "prefill": options["prefill_graph"],
            "decode": options["decode_graph"],
        },
        "quant": "checkpoint",
        "expert_backend": options["expert_backend"],
        "load_seconds": loaded_seconds,
        "capture_seconds": capture_seconds,
        "profile_graph_phases": options.get("profile_graph_phases", False),
        "overlap": overlap,
        "timing": "continuous_completions" if overlap else "synchronized_steps",
        "prefill_chunk_size": options["prefill_chunk_size"],
        "context_capacity": capacity,
        "max_batch_tokens": prefill_budget,
    }
    engine_options: dict[str, Any] = dict(
        max_running_requests=max(case.batch_size for case in cases),
        context_length=capacity,
        prefix_cache=any(case.shared_prefix_tokens for case in cases),
        enable_overlap=options.get("overlap", True),
        scheduler_config=SchedulerConfig(
            max_batch_tokens=prefill_budget,
            prefill_chunk_size=options["prefill_chunk_size"],
        ),
    )
    worker = None if groups is None else DistributedWorker(observed, groups)
    if groups is not None and worker is not None and not groups.is_driver:
        worker.serve()
        os._exit(0)
    engine = (
        Engine(observed, **engine_options)
        if worker is None
        else DistributedEngine(worker, **engine_options)
    )
    try:
        # A seeded synthetic prompt avoids tokenization overhead and special
        # token stops. Every replica has distinct, reproducible input tokens.
        generator = torch.Generator().manual_seed(41 + rank)
        prompt = torch.randint(
            1000, min(30_000, raw.vocab_size), (capacity,), generator=generator
        ).tolist()
        run_sweep(
            engine,
            cases,
            runner=observed,
            prompt_tokens=prompt,
            warmup_steps=options["warmup_steps"],
            measure_steps=options["measure_steps"],
            gpu_count=topology.world_size,
            attn_dp=topology.attn_dp,
            synchronize=torch.cuda.synchronize,
            reduce_step=None if groups is None else distributed_reducer(groups),
            metadata=metadata,
            on_result=_result_callback(report_path) if rank == 0 else None,
            trace_step=(
                None
                if options.get("trace_dir") is None
                else perfetto_trace(
                    options["trace_dir"], rank, metadata, decode_passes=2 + int(overlap)
                )
            ),
        )
    finally:
        # Closing a poisoned domain still tells followers to stop. Preserve the
        # original failure/report if close repeats that same domain error.
        failed = engine.poisoned or sys.exc_info()[0] is not None
        try:
            engine.close()
        except Exception:
            if not failed:
                raise
    # Spawned process exit releases graph-owned NCCL communicators together.
    if groups is not None:
        os._exit(0)


def _validate_configuration(model, cases, topology, options):
    from resin.bench import BenchCase

    spec = get_model(model)
    draft_tokens = options["draft_tokens"]
    accepted_drafts = options["accepted_drafts"]
    expert_backend = options["expert_backend"]
    prefill_chunk_size = options["prefill_chunk_size"]
    numa = options["numa"]
    for name, minimum in (
        ("warmup_steps", 0),
        ("measure_steps", 1),
        ("prefill_chunk_size", 1),
        ("draft_tokens", 0),
    ):
        value = options[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if accepted_drafts is not None and type(accepted_drafts) is not int:
        raise ValueError("accepted_drafts must be an integer or None")
    if not 0 < options["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if options["balanced_experts"] and spec.family not in {
        "gpt_oss",
        "deepseek_v41",
    }:
        raise ValueError(f"{model} does not support balanced expert routing")
    if spec.family == "inkling" and (
        draft_tokens != 5 or topology.world_size != 1 or prefill_chunk_size != 4096
    ):
        raise ValueError(
            "Inkling requires one GPU, five draft tokens, and prefill_chunk_size=4096"
        )
    if spec.family in {"gemma4", "qwen3_5"} and prefill_chunk_size > 4096:
        raise ValueError(f"{model} supports prefill chunks up to 4096 tokens")
    if spec.family == "glm5_3_flash":
        if (
            topology.attn_tp != 1
            or topology.cp != 1
            or topology.attn_dp != topology.world_size
            or topology.ep != topology.world_size
        ):
            raise ValueError("GLM requires attention-DP=EP=world size, TP1 and CP1")
        if prefill_chunk_size % 64:
            raise ValueError("GLM prefill chunks must align to 64-token pages")
        if draft_tokens > 7:
            raise ValueError("GLM native MTP supports at most seven draft tokens")
    if numa not in ("auto", "off", "required"):
        raise ValueError("numa must be 'auto', 'off', or 'required'")
    if expert_backend not in ("flashinfer", "deepgemm"):
        raise ValueError(f"unknown expert backend: {expert_backend}")
    if expert_backend == "deepgemm":
        if model != "deepseek-v41":
            raise ValueError("the DeepGEMM expert backend currently requires V4.1")
        if topology.world_size == 1:
            raise ValueError(
                "DeepGEMM sweeps require at least two ranks with explicit parallel groups"
            )
        if (
            topology.attn_tp != 1
            or topology.cp != 1
            or topology.attn_dp != topology.world_size
            or topology.ep != topology.world_size
        ):
            raise ValueError("DeepGEMM requires ADP=EP=world_size, attn_tp=1, and cp=1")
    if not cases:
        raise ValueError("the benchmark needs at least one case")
    for values in cases:
        BenchCase(**values)
    if spec.family in {"gpt_oss", "gemma4"} and options["decode_graph"] != "disabled":
        from resin.backend.cuda.graphs import DEFAULT_MAX_NUM_SEQS

        if max(case["batch_size"] for case in cases) > DEFAULT_MAX_NUM_SEQS:
            raise ValueError(
                f"{model} decode graph batch capacity is {DEFAULT_MAX_NUM_SEQS}; disable decode graphs for larger cohorts"
            )
    if (
        draft_tokens < 0
        or accepted_drafts is not None
        and not 0 <= accepted_drafts <= draft_tokens
    ):
        raise ValueError("accepted drafts must be between zero and draft tokens")
    if model.startswith("gpt-oss") and draft_tokens:
        raise ValueError("GPT-OSS has no draft model; pass --draft-tokens 0")
    if model.startswith("gpt-oss") and topology.world_size == 1:
        from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

        limit = FlashInferSM90MXFP4Kernel.max_tokens
        budget = max(case["batch_size"] for case in cases) * prefill_chunk_size
        if budget > limit:
            raise ValueError(
                f"single-GPU GPT-OSS prefill budget {budget} exceeds the kernel's "
                f"{limit}-token limit; reduce batch size or prefill_chunk_size"
            )


def run_benchmark(
    model: str,
    checkpoint: str | Path,
    cases: list[dict],
    *,
    parallelism: Parallelism | None = None,
    draft_tokens: int | None = None,
    accepted_drafts: int | None = None,
    balanced_experts: bool = False,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    prefill_chunk_size: int = 4096,
    prefill_graph: str = "breakable",
    decode_graph: str = "full",
    gpu_memory_utilization: float = 0.9,
    metadata: dict | None = None,
    expert_backend: str | None = None,
    profile_graph_phases: bool = False,
    overlap: bool = True,
    numa: Literal["auto", "off", "required"] = "auto",
    trace_dir: str | None = None,
    assistant_checkpoint: str | Path | None = None,
    progress_path: str | Path | None = None,
) -> list[dict]:
    """One load and graph catalog for all compatible points in this sweep.

    ``progress_path`` optionally retains a JSON array of completed cases even
    if a later case or worker fails. It must be a new file. Updates happen
    outside timing; successful results also include worker NUMA placement.
    """
    spec = get_model(model)
    topology = parallelism or Parallelism()
    draft_tokens = spec.draft_tokens if draft_tokens is None else draft_tokens
    expert_backend = spec.expert_backend if expert_backend is None else expert_backend
    options = dict(
        draft_tokens=draft_tokens,
        accepted_drafts=accepted_drafts,
        balanced_experts=balanced_experts,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        prefill_chunk_size=prefill_chunk_size,
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        gpu_memory_utilization=gpu_memory_utilization,
        metadata=metadata or {},
        expert_backend=expert_backend,
        profile_graph_phases=profile_graph_phases,
        overlap=overlap,
        numa=numa,
        trace_dir=trace_dir,
        assistant_checkpoint=None
        if assistant_checkpoint is None
        else str(assistant_checkpoint),
    )
    _validate_configuration(model, cases, topology, options)
    if torch.cuda.device_count() < topology.world_size:
        raise ValueError(f"the sweep requires {topology.world_size} visible GPUs")
    with tempfile.TemporaryDirectory(prefix="resin-bench-") as directory:
        output = (
            Path(directory) / "results.json"
            if progress_path is None
            else Path(progress_path).expanduser().resolve()
        )
        if progress_path is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                stream.write("[]")
        port = 0
        if topology.world_size > 1:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        processes = resin.process.spawn(
            _rank,
            args=(port, output, model, checkpoint, cases, topology, options),
            devices=[f"cuda:{rank}" for rank in range(topology.world_size)],
            numa=numa,
        )
        records = json.loads(output.read_text())
        placement = {
            "mode": numa,
            "bindings": [
                None if binding is None else asdict(binding)
                for binding in processes.placements
            ],
        }
        for record in records:
            record["numa"] = placement
        if progress_path is not None:
            pending = output.with_suffix(output.suffix + ".tmp")
            pending.write_text(json.dumps(records))
            pending.replace(output)
        return records


def write_results(path: str | Path, records: list[dict]) -> None:
    """Append raw points; plotting deliberately resolves repeated points later."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def source_metadata() -> dict:
    """Identify the submitted source, including local edits and new source files."""

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)

    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain"))
        source = hashlib.sha256(revision.encode())
        source.update(git("diff", "--binary", "HEAD"))
        for name in sorted(
            git("ls-files", "--others", "--exclude-standard").decode().splitlines()
        ):
            path = ROOT / name
            if path.suffix in {".py", ".rs", ".toml", ".lock", ".cu", ".cuh"}:
                source.update(name.encode())
                source.update(path.read_bytes())
        return dict(git_commit=revision, dirty=dirty, source_digest=source.hexdigest())
    except (OSError, subprocess.CalledProcessError):
        return dict(git_commit=None, dirty=None, source_digest=None)


def _integers(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from None
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return list(dict.fromkeys(values))


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="local checkpoint directory; required for local runs",
    )
    parser.add_argument(
        "--assistant-checkpoint",
        type=Path,
        help="local assistant/draft checkpoint when required",
    )
    parser.add_argument(
        "--gpu", help="Modal GPU type; defaults to the model's serving hardware"
    )
    for axis in ("world-size", "attn-dp", "attn-tp", "cp", "ep"):
        parser.add_argument(f"--{axis}", type=int)
    parser.add_argument(
        "--phases", default="prefill,decode", help="prefill, decode, or prefill,decode"
    )
    parser.add_argument(
        "--context-lengths",
        type=_integers,
        help="prompt lengths for both phases; phase-specific options take precedence",
    )
    parser.add_argument(
        "--prefill-context-lengths",
        type=_integers,
        help="prefill prompt lengths (default: 4096,8192,16384,32768,65536)",
    )
    parser.add_argument(
        "--decode-context-lengths",
        type=_integers,
        help="initial decode context lengths (default: 8192)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_integers,
        help="requests per attention-DP replica for both phases; phase-specific options take precedence",
    )
    parser.add_argument(
        "--prefill-batch-sizes", type=_integers, help="prefill batches (default: 1)"
    )
    parser.add_argument(
        "--decode-batch-sizes",
        type=_integers,
        help="decode batches (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument("--shared-prefix-tokens", type=int, default=0)
    parser.add_argument(
        "--draft-tokens",
        type=int,
        help="draft positions; defaults to model configuration",
    )
    parser.add_argument(
        "--accepted-drafts",
        type=int,
        help="force this many accepted drafts per step; omit for natural acceptance",
    )
    parser.add_argument(
        "--balanced-experts",
        action="store_true",
        help="use synthetic balanced MoE routing (off by default)",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help="execution chunk size, separate from prompt length (default: model preset, at least 4096; subject to kernel limits)",
    )
    for phase, default in (("prefill", "breakable"), ("decode", "full")):
        parser.add_argument(
            f"--{phase}-graph",
            choices=("disabled", "breakable", "full"),
            default=default,
        )
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--expert-backend", choices=("flashinfer", "deepgemm"))
    parser.add_argument("--profile-graph-phases", action="store_true")
    parser.add_argument(
        "--overlap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--numa", choices=("auto", "off", "required"), default="auto")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="capture two untimed passes per case, then merge driver-rank Perfetto traces",
    )
    parser.add_argument("--output", type=Path, default=Path("bench/results.jsonl"))
    parser.add_argument(
        "--plot", action="store_true", help="plot measured results after writing JSONL"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--download-only",
        action="store_true",
        help="populate pinned checkpoints on Modal without allocating a GPU",
    )
    actions.add_argument(
        "--publish-only",
        action="store_true",
        help="publish completion markers for existing Modal checkpoint files",
    )
    return parser


def _configuration(args):
    from resin.bench import BenchCase

    spec = get_model(args.model)
    if args.backend == "modal":
        validate_gpu(spec, args.gpu)
    world = spec.world_size if args.world_size is None else args.world_size
    topology = dict(world_size=world, cp=1 if args.cp is None else args.cp)
    for axis in ("attn_dp", "attn_tp", "ep"):
        explicit = getattr(args, axis)
        default = getattr(spec, axis)
        topology[axis] = (
            explicit
            if explicit is not None
            else (
                world if spec.world_size > 1 and default == spec.world_size else default
            )
        )
    parallelism = Parallelism(**topology)
    phases = args.phases.split(",")
    if not phases or any(phase not in {"prefill", "decode"} for phase in phases):
        raise ValueError("phases must be prefill, decode, or prefill,decode")
    contexts = {
        "prefill": args.prefill_context_lengths
        or args.context_lengths
        or [4096, 8192, 16384, 32768, 65536],
        "decode": args.decode_context_lengths or args.context_lengths or [8192],
    }
    batches = {
        "prefill": args.prefill_batch_sizes or args.batch_sizes or [1],
        "decode": args.decode_batch_sizes
        or args.batch_sizes
        or [1, 2, 4, 8, 16, 32, 64],
    }
    cases = [
        asdict(
            BenchCase(
                phase=phase,
                context_length=context,
                batch_size=batch,
                shared_prefix_tokens=args.shared_prefix_tokens,
            )
        )
        for phase in dict.fromkeys(phases)
        for context in contexts[phase]
        for batch in batches[phase]
    ]
    chunk = (
        max(4096, spec.prefill_chunk_size)
        if args.prefill_chunk_size is None
        else args.prefill_chunk_size
    )
    if args.prefill_chunk_size is None and spec.family == "gpt_oss" and world == 1:
        from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

        chunk = min(
            chunk,
            max(
                1,
                FlashInferSM90MXFP4Kernel.max_tokens
                // max(case["batch_size"] for case in cases),
            ),
        )
    options = dict(
        draft_tokens=spec.draft_tokens
        if args.draft_tokens is None
        else args.draft_tokens,
        accepted_drafts=args.accepted_drafts,
        balanced_experts=args.balanced_experts,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        prefill_chunk_size=chunk,
        prefill_graph=args.prefill_graph,
        decode_graph=args.decode_graph,
        gpu_memory_utilization=spec.gpu_memory_utilization
        if args.gpu_memory_utilization is None
        else args.gpu_memory_utilization,
        expert_backend=spec.expert_backend
        if args.expert_backend is None
        else args.expert_backend,
        profile_graph_phases=args.profile_graph_phases,
        overlap=args.overlap,
        numa=args.numa,
    )
    _validate_configuration(spec.name, cases, parallelism, options)
    return spec, cases, parallelism, options


def main(argv=None) -> None:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.backend == "local" and (args.download_only or args.publish_only):
        parser.error("--download-only/--publish-only require --backend modal")
    if args.backend == "local" and args.gpu is not None:
        parser.error(
            "--gpu selects Modal hardware; local runs use visible CUDA devices"
        )
    if args.backend == "local" and args.checkpoint is None:
        parser.error("--checkpoint is required for local benchmarks")
    if args.backend == "modal" and (
        args.checkpoint is not None or args.assistant_checkpoint is not None
    ):
        parser.error(
            "Modal uses the pinned model registry; local checkpoint paths require --backend local"
        )
    if args.download_only or args.publish_only:
        import modal

        from deploy import modal_bench

        with modal.enable_output(), modal_bench.app.run():
            print(
                json.dumps(
                    modal_bench.download.remote(args.model, args.publish_only), indent=2
                )
            )
        return
    try:
        spec, cases, topology, options = _configuration(args)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    options["metadata"] = source_metadata()
    if args.backend == "modal":
        from uuid import uuid4

        import modal

        from deploy import modal_bench

        remote_trace_dir = f"/bench-traces/{uuid4().hex}" if args.trace_dir else None
        options["trace_dir"] = remote_trace_dir
        with modal.enable_output(), modal_bench.app.run():
            records = modal_bench.sweep.with_options(
                gpu=f"{args.gpu or spec.gpu}:{topology.world_size}"
            ).remote(
                spec.name,
                cases,
                {
                    axis: getattr(topology, axis)
                    for axis in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
                },
                options,
            )
            write_results(args.output, records)
            if remote_trace_dir:
                for path in modal_bench.download_traces(
                    remote_trace_dir, args.trace_dir
                ):
                    print(f"Perfetto trace: {path}")
    else:
        from deploy.models import require_checkpoint

        checkpoint = require_checkpoint(
            spec,
            checkpoint=args.checkpoint,
            assistant_checkpoint=args.assistant_checkpoint,
            draft_tokens=options["draft_tokens"],
        )
        local_trace_dir = None
        if args.trace_dir is not None:
            from uuid import uuid4

            local_trace_dir = args.trace_dir / uuid4().hex
        options.update(
            trace_dir=None if local_trace_dir is None else str(local_trace_dir),
            assistant_checkpoint=args.assistant_checkpoint,
        )
        records = run_benchmark(
            spec.name, checkpoint, cases, parallelism=topology, **options
        )
        write_results(args.output, records)
        if args.trace_dir:
            from scripts.merge_bench_traces import main as merge_main

            merge_main([str(local_trace_dir)])
    print(f"Appended {len(records)} raw points to {args.output}")
    if args.plot:
        from scripts.plot_bench import main as plot_main

        plot_main(
            [str(args.output), "--mode", "both" if "," in args.phases else args.phases]
        )


if __name__ == "__main__":
    main()
