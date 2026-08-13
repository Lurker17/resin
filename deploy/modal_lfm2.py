"""Run the tokenizer-free PyTorch LFM2 baseline on a Modal H100."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import modal

from deploy.jit_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
    KERNEL_CACHE_VOLUME_NAME,
    KERNEL_CACHE_VOLUME_VERSION,
    install_flashinfer_hash_cache,
    kernel_cache_manifest,
    run_cache_prewarm,
    run_cache_reuse_probe,
    run_cached_workload,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "LiquidAI/LFM2.5-230M"
REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"
MODEL_FILES = [
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
]
PROMPT = [
    1,
    6,
    24131,
    708,
    4083,
    938,
    768,
    11538,
    16701,
    12216,
    968,
    59929,
    14009,
    523,
    7,
    708,
    6,
    6423,
    708,
    3493,
    856,
    779,
    5706,
    803,
    4481,
    540,
    7,
    708,
    6,
    64015,
    708,
]
EXPECTED = [1098, 5706, 803, 4481, 856, 5242, 523, 7]
ASYNC_PROOF_PROTOCOL = "lfm2-h100-async-completion-v4-blake3-canary"
ASYNC_PROOF_BATCH_SIZE = 64
ASYNC_PROOF_PAGE_SIZE = 16
ASYNC_PROOF_OVERLAP_PROMPT_TOKENS = 4096
# Exactly 64 requests * (4096 tokens / 16 tokens per page).
ASYNC_PROOF_MAX_PAGES = 16_384
ASYNC_PROOF_MAX_NEW_TOKENS = 8
ASYNC_PROOF_CPU_BLAKE3_ROUNDS = 100_000
ASYNC_PROOF_CPU_DIGEST = (
    "9da38bf57ee2e526478f92c8132c2d441c9f1708a90b337f6b365cbcea197b21"
)
ASYNC_PROOF_DEFAULT_STREAM_SENTINEL_MS = 5000
NATIVE_CONTINUATION_PROOF_PROTOCOL = "lfm2-h100-native-scheduler-continuation-v2"
NATIVE_CONTINUATION_PAGE_SIZE = 16
NATIVE_CONTINUATION_PROMPT_TOKENS = 63
NATIVE_CONTINUATION_MAX_BATCH_TOKENS = 23
NATIVE_CONTINUATION_PREFILL_CHUNK_SIZE = 23
NATIVE_CONTINUATION_MAX_PAGES = 8
NATIVE_CONTINUATION_MAX_NEW_TOKENS = 8
NATIVE_CONTINUATION_EXPECTED = [560, 985, 510, 6567, 1090, 768, 56412, 810]


def _drive_native_continuation_engine(engine) -> tuple[list[object], dict[str, object]]:
    """Drain async native batches, including terminal batches with no events."""

    import time

    all_events = []
    batch_ids = []
    terminal_event_counts = []
    pending_polls = 0
    while engine.has_work():
        if engine.in_flight_batch_count:
            raise AssertionError("native continuation driver found an unowned batch")
        batch_id = engine.dispatch()
        if batch_id is None:
            raise AssertionError(
                "native continuation Engine had work but did not dispatch"
            )
        batch_ids.append(batch_id)
        terminal_events = None
        deadline = time.perf_counter() + 120
        while engine.in_flight_batch_count:
            polled_events = engine.poll()
            if engine.in_flight_batch_count:
                if polled_events:
                    raise AssertionError("pending native poll returned token events")
                pending_polls += 1
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        "native continuation CUDA batch did not complete"
                    )
                time.sleep(0)
            else:
                # A terminal partial-prefill poll intentionally lands here with
                # an empty list. Keep driving the requeued request rather than
                # treating zero sampled events as a missing completion.
                terminal_events = polled_events
        if terminal_events is None:
            raise AssertionError(
                "native continuation batch never reached terminal poll"
            )
        terminal_event_counts.append(len(terminal_events))
        all_events.extend(terminal_events)

    return all_events, {
        "batch_ids": batch_ids,
        "terminal_batches": len(batch_ids),
        "terminal_event_counts": terminal_event_counts,
        "zero_event_terminal_batches": sum(
            event_count == 0 for event_count in terminal_event_counts
        ),
        "pending_polls": pending_polls,
    }


def _native_plan_query_lengths(sequence_indptr) -> list[int]:
    """Convert an N+1 native row pointer into exactly N positive widths."""

    values = tuple(sequence_indptr)
    if len(values) < 2:
        raise ValueError("native sequence_indptr must contain at least one row")
    lengths = [end - start for start, end in zip(values, values[1:])]
    if any(length <= 0 for length in lengths):
        raise ValueError("native plan rows must contain at least one token")
    return lengths


def _submit_mapped_lfm2_request(
    engine, state_pool, token_ids, *, max_new_tokens: int
) -> int:
    """Transfer one fresh recurrent-state generation into an Engine request."""

    state_handle = state_pool.allocate()[0]
    try:
        request_id = engine.submit(
            token_ids,
            max_new_tokens=max_new_tokens,
            state_handles=[state_handle],
            state_access_modes=["write"],
        )
    except BaseException:
        state_pool.release([state_handle])
        raise
    # Engine.submit retained its own reference. Relinquish the caller reference
    # immediately so terminal retirement makes this generation reusable.
    state_pool.release([state_handle])
    return request_id


def _recurrent_state_pool_snapshot(state_pool) -> dict[str, object]:
    return {
        "pool_id": state_pool.pool_id,
        "capacity": state_pool.capacity,
        "active": state_pool.active_count,
        "free": state_pool.free_count,
        "retiring": state_pool.retiring_count,
        "physical_mapping": state_pool.physical_mapping,
    }


def cache_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION, allow_patterns=MODEL_FILES)


def _cpu_metadata_digest() -> str:
    """Perform deterministic, accelerator-independent work during a completion."""

    from blake3 import blake3

    payload = json.dumps(
        {
            "batch_size": ASYNC_PROOF_BATCH_SIZE,
            "expected": EXPECTED,
            "max_new_tokens": ASYNC_PROOF_MAX_NEW_TOKENS,
            "prompt": PROMPT,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = blake3(payload)
    for iteration in range(ASYNC_PROOF_CPU_BLAKE3_ROUNDS):
        digest.update(iteration.to_bytes(4, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _validated_h100(torch) -> tuple[str, tuple[int, int]]:
    """Reject a compatible-looking substitute for the requested physical GPU."""

    device_name = str(torch.cuda.get_device_name())
    capability = tuple(torch.cuda.get_device_capability())
    if "H100" not in device_name.upper() or capability != (9, 0):
        raise AssertionError(
            "the async completion proof requires an actual H100 (SM90), got "
            f"{device_name!r} with capability {capability!r}"
        )
    return device_name, capability


def _enqueue_default_stream_sentinel(torch):
    """Occupy the default stream without creating a dependency for the runner."""

    sleep = getattr(torch.cuda, "_sleep", None)
    if not callable(sleep):
        raise RuntimeError("the pinned PyTorch build does not expose torch.cuda._sleep")
    device = torch.device("cuda")
    clock_rate_khz = int(torch.cuda.get_device_properties(device).clock_rate)
    cycles = clock_rate_khz * ASYNC_PROOF_DEFAULT_STREAM_SENTINEL_MS
    if cycles <= 0:
        raise RuntimeError("CUDA reported a nonpositive device clock rate")
    default_stream = torch.cuda.default_stream(device)
    done = torch.cuda.Event()
    with torch.cuda.stream(default_stream):
        sleep(cycles)
        done.record(default_stream)
    return done, cycles


def _run_async_overlap_window(
    engine, torch, *, expected_token_id: int | None = None
) -> dict[str, object]:
    """Prove CPU work and private-stream GPU work are live concurrently."""

    import time

    default_sentinel, sentinel_cycles = _enqueue_default_stream_sentinel(torch)

    enqueue_started = time.perf_counter()
    batch_id = engine.dispatch()
    enqueue_wall_ms = (time.perf_counter() - enqueue_started) * 1000
    if batch_id is None:
        raise AssertionError(
            "the async proof had work but Engine.dispatch returned None"
        )
    if engine.in_flight_request_count != ASYNC_PROOF_BATCH_SIZE:
        raise AssertionError("Engine did not lease the complete async proof batch")

    initial_poll_started = time.perf_counter()
    initial_events = engine.poll()
    initial_poll_ms = (time.perf_counter() - initial_poll_started) * 1000
    if initial_events or engine.in_flight_batch_count != 1:
        raise AssertionError(
            "the real prefill completed before the first nonblocking event query"
        )

    cpu_started = time.perf_counter()
    cpu_digest = _cpu_metadata_digest()
    cpu_interval_ms = (time.perf_counter() - cpu_started) * 1000
    if cpu_digest != ASYNC_PROOF_CPU_DIGEST:
        raise AssertionError(f"deterministic CPU digest changed: {cpu_digest}")

    post_cpu_poll_started = time.perf_counter()
    post_cpu_events = engine.poll()
    post_cpu_poll_ms = (time.perf_counter() - post_cpu_poll_started) * 1000
    if post_cpu_events or engine.in_flight_batch_count != 1:
        raise AssertionError(
            "the completion did not remain pending across the CPU metadata interval"
        )

    terminal_wait_started = time.perf_counter()
    terminal_events = []
    terminal_poll_count = 0
    deadline = terminal_wait_started + 60
    while engine.in_flight_batch_count:
        terminal_poll_count += 1
        terminal_events = engine.poll()
        if time.perf_counter() >= deadline:
            raise TimeoutError("the real CUDA completion did not become terminal")
        if engine.in_flight_batch_count:
            time.sleep(0)
    terminal_wait_ms = (time.perf_counter() - terminal_wait_started) * 1000
    if len(terminal_events) != ASYNC_PROOF_BATCH_SIZE:
        raise AssertionError(
            "terminal async prefill did not emit one token per proof request"
        )

    sentinel_pending_at_terminal = not bool(default_sentinel.query())
    if not sentinel_pending_at_terminal:
        raise AssertionError(
            "the default-stream sentinel finished before the private-stream model; "
            "the run does not prove stream independence"
        )
    sentinel_wait_started = time.perf_counter()
    default_sentinel.synchronize()
    sentinel_wait_ms = (time.perf_counter() - sentinel_wait_started) * 1000

    terminal_token_ids = [event.token_id for event in terminal_events]
    if len(set(terminal_token_ids)) != 1:
        raise AssertionError(
            f"identical overlap prompts produced different tokens: {terminal_token_ids}"
        )
    if (
        expected_token_id is not None
        and terminal_token_ids != [expected_token_id] * ASYNC_PROOF_BATCH_SIZE
    ):
        raise AssertionError(
            f"unexpected terminal prefill tokens: {terminal_token_ids}"
        )
    return {
        "batch_id": batch_id,
        "enqueue_wall_ms": enqueue_wall_ms,
        "initial_pending_poll_ms": initial_poll_ms,
        "cpu_interval_ms": cpu_interval_ms,
        "cpu_digest": cpu_digest,
        "post_cpu_pending_poll_ms": post_cpu_poll_ms,
        "terminal_wait_ms": terminal_wait_ms,
        "terminal_poll_count": terminal_poll_count,
        "terminal_token_ids": terminal_token_ids,
        "cuda_event_gpu_ms": None,
        "cuda_event_gpu_time": (
            "unavailable: the runner completion is opaque and exposes no public "
            "event-timing metric"
        ),
        "default_stream_sentinel_cycles": sentinel_cycles,
        "default_stream_sentinel_target_ms": (ASYNC_PROOF_DEFAULT_STREAM_SENTINEL_MS),
        "default_stream_sentinel_pending_at_model_terminal": (
            sentinel_pending_at_terminal
        ),
        "default_stream_sentinel_event_wait_ms": sentinel_wait_ms,
        "global_cuda_synchronize_in_overlap_window": False,
    }


def _complete_async_batch(engine) -> tuple[list[object], float, int, int]:
    """Measure one logical batch from native dispatch through terminal poll."""

    import time

    if engine.in_flight_batch_count:
        raise AssertionError("cannot dispatch over an existing in-flight batch")
    started = time.perf_counter()
    batch_id = engine.dispatch()
    if batch_id is None:
        raise AssertionError("Engine had ready work but dispatch returned None")
    pending_polls = 0
    terminal_events = []
    deadline = started + 120
    while engine.in_flight_batch_count:
        terminal_events = engine.poll()
        if engine.in_flight_batch_count:
            if terminal_events:
                raise AssertionError("pending Engine.poll unexpectedly returned events")
            pending_polls += 1
            if time.perf_counter() >= deadline:
                raise TimeoutError("the async LFM2 batch did not complete")
            time.sleep(0)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not terminal_events:
        raise AssertionError("terminal Engine.poll returned no token events")
    return terminal_events, elapsed_ms, pending_polls, batch_id


def _drain_async_engine(engine) -> dict[str, int]:
    """Drive event-query completions without introducing a device-wide barrier."""

    dispatches = 0
    pending_polls = 0
    terminal_batches = 0
    while engine.has_work():
        _, _, batch_pending_polls, _ = _complete_async_batch(engine)
        dispatches += 1
        pending_polls += batch_pending_polls
        terminal_batches += 1
    return {
        "dispatches": dispatches,
        "pending_polls": pending_polls,
        "terminal_batches": terminal_batches,
    }


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal"
    )
    .env(
        {
            "PATH": (
                "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            )
        }
    )
    .uv_pip_install(
        "maturin>=1.9,<2",
        "torch==2.13.0",
        "numpy==2.5.1",
        "huggingface-hub==1.27.0",
        "safetensors==0.8.0",
        "flashinfer-python==0.6.15.post1",
        "flash-attn-4==4.0.0b22",
        "nvidia-cutlass-dsl==4.6.0.dev0",
        "apache-tvm-ffi==0.1.13.post2",
        "quack-kernels==0.5.3",
        "triton==3.7.1",
        "blake3==1.0.9",
    )
    .uv_pip_install(
        "flashinfer-cubin==0.6.15.post1",
        extra_index_url="https://flashinfer.ai/whl",
    )
    .add_local_file(ROOT / "deploy/__init__.py", "/root/deploy/__init__.py", copy=True)
    .add_local_file(
        ROOT / "deploy/jit_cache.py", "/root/deploy/jit_cache.py", copy=True
    )
    .run_function(cache_model)
    .add_local_file(ROOT / "Cargo.toml", "/opt/resin/Cargo.toml", copy=True)
    .add_local_file(ROOT / "Cargo.lock", "/opt/resin/Cargo.lock", copy=True)
    .add_local_file(ROOT / "pyproject.toml", "/opt/resin/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/opt/resin/README.md", copy=True)
    .add_local_file(ROOT / "NOTICE", "/opt/resin/NOTICE", copy=True)
    .add_local_dir(ROOT / "src", "/opt/resin/src", copy=True)
    .run_commands("python -m pip install --no-deps /opt/resin")
    .env(KERNEL_CACHE_ENV)
)

kernel_cache_volume = modal.Volume.from_name(
    KERNEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=KERNEL_CACHE_VOLUME_VERSION,
)
single_use_options = (
    {"single_use_containers": True}
    if "single_use_containers" in inspect.signature(modal.App.function).parameters
    else {"max_inputs": 1}
)
app = modal.App(
    "resin-lfm2-torch",
    image=image,
    volumes={KERNEL_CACHE_MOUNT: kernel_cache_volume},
)


def _benchmark_report() -> dict[str, object]:
    import sys
    import time
    from statistics import median

    import torch

    from resin import Engine
    from resin.backends.lfm2.reference import Lfm2TorchRunner

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    runner = Lfm2TorchRunner(MODEL, revision=REVISION, device="cuda")
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000

    engine = Engine(runner, eos_token_id=runner.eos_token_id)
    request = engine.submit(PROMPT, max_new_tokens=16)
    step_ms = []
    while engine.has_work():
        start = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - start) * 1000)
    output = request.take_result()
    if output != EXPECTED:
        raise AssertionError(f"unexpected greedy output: {output}")
    if runner.live_cache_count != 0:
        raise AssertionError("request cache was not released")

    return {
        "device": torch.cuda.get_device_name(),
        "device_capability": torch.cuda.get_device_capability(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "model_load_ms": load_ms,
        "prefill_ms": step_ms[0],
        "median_decode_ms": median(step_ms[1:]),
        "decode_tokens_per_second": 1000 / median(step_ms[1:]),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "forward_calls": runner.forward_calls,
        "output_ids": output,
        "tokenizer_modules_loaded": [
            name for name in ("transformers", "tokenizers") if name in sys.modules
        ],
    }


def _kernel_benchmark_report() -> dict[str, object]:
    import sys
    import time
    from statistics import median

    import torch

    install_flashinfer_hash_cache()
    from resin import Engine, StatePool
    from resin.backends.lfm2.cuda import Lfm2CudaRunner

    page_size = 16
    max_pages = 64

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    runner = Lfm2CudaRunner(
        MODEL,
        revision=REVISION,
        page_size=page_size,
        max_pages=max_pages,
    )
    state_pool = StatePool(1)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000
    engine = Engine(
        runner,
        eos_token_id=runner.eos_token_id,
        state_pool=state_pool,
    )

    def generate() -> list[float]:
        request = _submit_mapped_lfm2_request(
            engine, state_pool, PROMPT, max_new_tokens=16
        )
        latencies = []
        while engine.has_work():
            _, elapsed_ms, _, _ = _complete_async_batch(engine)
            latencies.append(elapsed_ms)
        output = request.take_result()
        if output != EXPECTED:
            raise AssertionError(f"unexpected greedy output: {output}")
        if runner.live_cache_count != 0:
            raise AssertionError("physical request cache was not released")
        if (
            engine.cache_occupied_block_count != 0
            or engine.cache_free_parent_count != max_pages
        ):
            raise AssertionError("Rust-owned cache blocks were not released")
        if state_pool.active_count != 0 or state_pool.free_count != 1:
            raise AssertionError("native recurrent state was not released")
        return latencies

    cold_step_ms = generate()
    warm_step_ms = generate()
    cold_post_jit_decode = cold_step_ms[2:]
    fully_warm_decode = warm_step_ms[1:]
    return {
        "device": torch.cuda.get_device_name(),
        "device_capability": torch.cuda.get_device_capability(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "model_load_ms": load_ms,
        "fa4_first_prefill_ms": cold_step_ms[0],
        "flashinfer_first_decode_ms": cold_step_ms[1],
        "cold_request_median_post_jit_decode_ms": median(cold_post_jit_decode),
        "fa4_warm_prefill_ms": warm_step_ms[0],
        "flashinfer_median_fully_warm_decode_ms": median(fully_warm_decode),
        "fully_warm_decode_tokens_per_second": 1000 / median(fully_warm_decode),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "forward_calls": runner.forward_calls,
        "cache_pool_id": engine.cache_pool_id,
        "cache_parent_capacity": engine.cache_parent_capacity,
        "cache_free_parent_count": engine.cache_free_parent_count,
        "recurrent_state_pool": _recurrent_state_pool_snapshot(state_pool),
        "output_ids": EXPECTED,
        "tokenizer_modules_loaded": [
            name for name in ("transformers", "tokenizers") if name in sys.modules
        ],
    }


def _async_completion_report() -> dict[str, object]:
    """Exercise real event-driven Engine completions on one physical H100."""

    import sys
    import time

    import torch
    from blake3 import blake3

    install_flashinfer_hash_cache()
    from resin import Engine, StatePool
    from resin.backends.lfm2.cuda import Lfm2CudaRunner

    device_name, capability = _validated_h100(torch)
    torch.cuda.reset_peak_memory_stats()
    construct_started = time.perf_counter()
    runner = Lfm2CudaRunner(
        MODEL,
        revision=REVISION,
        page_size=ASYNC_PROOF_PAGE_SIZE,
        max_pages=ASYNC_PROOF_MAX_PAGES,
    )
    state_pool = StatePool(ASYNC_PROOF_BATCH_SIZE)
    construct_enqueue_ms = (time.perf_counter() - construct_started) * 1000
    engine = Engine(
        runner,
        max_running_requests=ASYNC_PROOF_BATCH_SIZE,
        eos_token_id=runner.eos_token_id,
        state_pool=state_pool,
    )
    if not engine.asynchronous:
        raise AssertionError(
            "Engine did not select the runner's dispatch/poll contract"
        )

    # Establish the exact B64 model, kernel, staging, and caching-allocator
    # shapes before the measured window. Otherwise a first-shape cudaMalloc
    # could serialize behind the deliberately busy default stream and confound
    # the private-stream independence check.
    warmup_requests = [
        _submit_mapped_lfm2_request(
            engine,
            state_pool,
            PROMPT,
            max_new_tokens=ASYNC_PROOF_MAX_NEW_TOKENS,
        )
        for _ in range(ASYNC_PROOF_BATCH_SIZE)
    ]
    warmup_drive = _drain_async_engine(engine)
    warmup_outputs = [request.take_result() for request in warmup_requests]
    if any(output != EXPECTED for output in warmup_outputs):
        mismatch = next(
            index for index, output in enumerate(warmup_outputs) if output != EXPECTED
        )
        raise AssertionError(
            f"async warmup request {mismatch} produced {warmup_outputs[mismatch]}"
        )
    if runner.live_cache_count != 0:
        raise AssertionError("warmup physical request cache was not released")
    if (
        engine.cache_occupied_block_count != 0
        or engine.cache_free_parent_count != ASYNC_PROOF_MAX_PAGES
    ):
        raise AssertionError("warmup Rust-owned KV pages were not released")
    if state_pool.active_count != 0:
        raise AssertionError("warmup recurrent-state generations were not released")

    overlap_shape_prompt = (
        PROMPT * ((ASYNC_PROOF_OVERLAP_PROMPT_TOKENS + len(PROMPT) - 1) // len(PROMPT))
    )[:ASYNC_PROOF_OVERLAP_PROMPT_TOKENS]

    # Prime the exact long-prefill allocations before placing the unrelated
    # default-stream sentinel. Otherwise a first-shape cudaMalloc may wait for
    # unrelated device work and confound the stream-independence assertion.
    overlap_shape_warmup_ids = [
        _submit_mapped_lfm2_request(
            engine, state_pool, overlap_shape_prompt, max_new_tokens=1
        )
        for _ in range(ASYNC_PROOF_BATCH_SIZE)
    ]
    overlap_shape_warmup = _drain_async_engine(engine)
    overlap_shape_warmup_outputs = [
        request.take_result() for request in overlap_shape_warmup_ids
    ]
    overlap_shape_warmup_tokens = [
        output[0] for output in overlap_shape_warmup_outputs if len(output) == 1
    ]
    if (
        len(overlap_shape_warmup_tokens) != ASYNC_PROOF_BATCH_SIZE
        or len(set(overlap_shape_warmup_tokens)) != 1
    ):
        raise AssertionError(
            f"invalid long-prefill shape warmup outputs: {overlap_shape_warmup_outputs}"
        )
    if runner.live_cache_count != 0:
        raise AssertionError("long-prefill warmup physical caches were not released")
    if (
        engine.cache_occupied_block_count != 0
        or engine.cache_free_parent_count != ASYNC_PROOF_MAX_PAGES
    ):
        raise AssertionError("long-prefill warmup native pages were not released")
    if state_pool.active_count != 0:
        raise AssertionError(
            "long-prefill warmup recurrent-state generations were not released"
        )

    # The measured output must differ from the same-shape B64 allocation warmup.
    # Otherwise a kernel accidentally launched behind the default-stream sentinel
    # could leave stale output bytes that happen to satisfy an equality-only test.
    canary_prompt = None
    canary_reference_token = None
    canary_prefix_token = None
    for prefix_token in (0, 36309, 1):
        candidate = [prefix_token] * (
            ASYNC_PROOF_OVERLAP_PROMPT_TOKENS - len(PROMPT)
        ) + PROMPT
        reference_id = _submit_mapped_lfm2_request(
            engine, state_pool, candidate, max_new_tokens=1
        )
        _complete_async_batch(engine)
        reference_output = reference_id.take_result()
        if len(reference_output) != 1:
            raise AssertionError(f"invalid B1 canary reference: {reference_output}")
        if reference_output[0] != overlap_shape_warmup_tokens[0]:
            canary_prompt = candidate
            canary_reference_token = reference_output[0]
            canary_prefix_token = prefix_token
            break
    if canary_prompt is None:
        raise AssertionError(
            "could not find a long-prefill canary distinct from stale B64 warmup output"
        )
    if runner.live_cache_count != 0:
        raise AssertionError("canary reference physical cache was not released")
    if (
        engine.cache_occupied_block_count != 0
        or engine.cache_free_parent_count != ASYNC_PROOF_MAX_PAGES
    ):
        raise AssertionError("canary reference native pages were not released")
    if state_pool.active_count != 0:
        raise AssertionError("canary recurrent-state generation was not released")

    overlap_request_ids = [
        _submit_mapped_lfm2_request(engine, state_pool, canary_prompt, max_new_tokens=1)
        for _ in range(ASYNC_PROOF_BATCH_SIZE)
    ]
    calls_before_proof = runner.forward_calls
    overlap = _run_async_overlap_window(
        engine,
        torch,
        expected_token_id=canary_reference_token,
    )
    proof_dispatch_calls = runner.forward_calls - calls_before_proof
    overlap_outputs = [request.take_result() for request in overlap_request_ids]
    overlap_token_ids = [output[0] for output in overlap_outputs if len(output) == 1]
    if (
        len(overlap_token_ids) != ASYNC_PROOF_BATCH_SIZE
        or len(set(overlap_token_ids)) != 1
    ):
        raise AssertionError(f"invalid long-prefill overlap outputs: {overlap_outputs}")
    outputs = warmup_outputs
    output_digest = blake3(
        json.dumps(outputs, separators=(",", ":")).encode()
    ).hexdigest()
    tokenizer_modules = [
        name for name in ("transformers", "tokenizers") if name in sys.modules
    ]
    if tokenizer_modules:
        raise AssertionError(
            f"tokenizer packages entered the async engine path: {tokenizer_modules}"
        )
    if runner.live_cache_count != 0:
        raise AssertionError("physical request caches were not fully retired")
    if (
        engine.cache_occupied_block_count != 0
        or engine.cache_free_parent_count != ASYNC_PROOF_MAX_PAGES
    ):
        raise AssertionError("Rust-owned KV pages were not fully retired")
    if state_pool.active_count != 0 or state_pool.free_count != ASYNC_PROOF_BATCH_SIZE:
        raise AssertionError("native recurrent state was not fully retired")
    if engine.in_flight_batch_count != 0 or engine.ready_request_count != 0:
        raise AssertionError("Engine retained work after the async proof")
    if engine.poisoned:
        raise AssertionError(f"Engine was poisoned: {engine.poison_error}")
    if proof_dispatch_calls != 1:
        raise AssertionError(
            f"overlap proof used {proof_dispatch_calls} dispatches, expected 1"
        )
    engine.close()
    if not engine.closed:
        raise AssertionError("Engine.close did not enter the terminal closed state")

    return {
        "protocol": ASYNC_PROOF_PROTOCOL,
        "device": device_name,
        "device_capability": capability,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "model_construct_enqueue_ms": construct_enqueue_ms,
        "asynchronous_engine": engine.asynchronous,
        "engine_closed": engine.closed,
        "request_count": ASYNC_PROOF_BATCH_SIZE,
        "prompt_tokens_per_request": len(PROMPT),
        "expected_output_ids": EXPECTED,
        "all_outputs_match": True,
        "output_ids_blake3": output_digest,
        "warmup": {
            "request_count": ASYNC_PROOF_BATCH_SIZE,
            "expected_output_ids": EXPECTED,
            "all_outputs_match": True,
            **warmup_drive,
        },
        "overlap": overlap,
        "overlap_prompt_tokens_per_request": len(canary_prompt),
        "overlap_canary_prefix_token": canary_prefix_token,
        "overlap_canary_reference_token": canary_reference_token,
        "overlap_stale_shape_warmup_token": overlap_shape_warmup_tokens[0],
        "overlap_canary_distinct_from_stale_warmup": (
            canary_reference_token != overlap_shape_warmup_tokens[0]
        ),
        "overlap_shape_warmup": {
            "output_token_id": overlap_shape_warmup_tokens[0],
            **overlap_shape_warmup,
        },
        "overlap_output_token_id": overlap_token_ids[0],
        "proof_dispatch_calls": proof_dispatch_calls,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "cache_pool_id": engine.cache_pool_id,
        "cache_parent_capacity": engine.cache_parent_capacity,
        "cache_occupied_block_count": engine.cache_occupied_block_count,
        "cache_free_parent_count": engine.cache_free_parent_count,
        "physical_live_cache_count": runner.live_cache_count,
        "recurrent_state_pool": _recurrent_state_pool_snapshot(state_pool),
        "tokenizer_modules_loaded": tokenizer_modules,
    }


def _native_continuation_scheduler_report() -> dict[str, object]:
    """Prove native chunk scheduling through real paged FA4 and decode kernels."""

    import sys
    import time

    import torch

    install_flashinfer_hash_cache()
    from resin import Engine, SchedulerConfig, StatePool
    from resin.backends.lfm2.cuda import Lfm2CudaRunner

    class RecordingRunner:
        supports_continuation_prefill = True
        supports_recurrent_state_mapping = True
        recurrent_state_mapping = Lfm2CudaRunner.recurrent_state_mapping

        def __init__(self, inner) -> None:
            self.inner = inner
            self.cache_layout = inner.cache_layout
            self.plans = []
            self.release_calls = []

        def bind_cache_and_recurrent_state(
            self,
            cache_pool_id,
            layout,
            state_pool_id,
            capacity,
            protocol,
            required_arena_id,
        ):
            return self.inner.bind_cache_and_recurrent_state(
                cache_pool_id,
                layout,
                state_pool_id,
                capacity,
                protocol,
                required_arena_id,
            )

        def dispatch(self, plan):
            self.plans.append(plan)
            return self.inner.dispatch(plan)

        def poll(self, completion):
            return self.inner.poll(completion)

        def release(self, request_ids):
            self.release_calls.append(list(request_ids))
            self.inner.release(request_ids)

    device_name, capability = _validated_h100(torch)
    proof_prompt = (
        PROMPT * ((NATIVE_CONTINUATION_PROMPT_TOKENS + len(PROMPT) - 1) // len(PROMPT))
    )[:NATIVE_CONTINUATION_PROMPT_TOKENS]
    if len(proof_prompt) != 63:
        raise AssertionError("native continuation proof requires a 63-token prompt")

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    runner = Lfm2CudaRunner(
        MODEL,
        revision=REVISION,
        page_size=NATIVE_CONTINUATION_PAGE_SIZE,
        max_pages=NATIVE_CONTINUATION_MAX_PAGES,
    )
    recording_runner = RecordingRunner(runner)
    state_pool = StatePool(1)
    scheduler_config = SchedulerConfig(
        max_batch_tokens=NATIVE_CONTINUATION_MAX_BATCH_TOKENS,
        prefill_chunk_size=NATIVE_CONTINUATION_PREFILL_CHUNK_SIZE,
    )
    engine = Engine(
        recording_runner,
        max_running_requests=1,
        scheduler_config=scheduler_config,
        eos_token_id=runner.eos_token_id,
        state_pool=state_pool,
    )
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_started) * 1000
    if not engine.asynchronous:
        raise AssertionError("native continuation proof did not select async execution")

    request_id = _submit_mapped_lfm2_request(
        engine,
        state_pool,
        proof_prompt,
        max_new_tokens=NATIVE_CONTINUATION_MAX_NEW_TOKENS,
    )
    reserved_blocks = engine.cache_occupied_block_count
    maximum_cached_tokens = len(proof_prompt) + NATIVE_CONTINUATION_MAX_NEW_TOKENS - 1
    expected_reserved_blocks = (
        maximum_cached_tokens + NATIVE_CONTINUATION_PAGE_SIZE - 1
    ) // NATIVE_CONTINUATION_PAGE_SIZE
    if expected_reserved_blocks != 5:
        raise AssertionError("native continuation reservation oracle changed")
    if reserved_blocks != expected_reserved_blocks:
        raise AssertionError(
            f"native Engine reserved {reserved_blocks} blocks, expected 5"
        )

    events, drive = _drive_native_continuation_engine(engine)
    trajectory = [event.token_id for event in events]
    if trajectory != NATIVE_CONTINUATION_EXPECTED:
        raise AssertionError(f"native continuation trajectory changed: {trajectory}")
    if any(event.request_id != request_id for event in events):
        raise AssertionError("native continuation emitted an event for another request")
    if [event.finished for event in events] != [False] * 7 + [True]:
        raise AssertionError("native continuation finished flags are inconsistent")
    if request_id.take_result() != NATIVE_CONTINUATION_EXPECTED:
        raise AssertionError("native Engine output does not match emitted trajectory")

    plans = recording_runner.plans
    expected_contexts = [[0], [23], [46]] + [[value] for value in range(63, 70)]
    expected_query_lengths = [[23], [23], [17]] + [[1]] * 7
    expected_sample_rows = [[], [], [0]] + [[0]] * 7
    plan_trace = []
    for plan in plans:
        query_lengths = _native_plan_query_lengths(plan.sequence_indptr)
        group = plan.cache_groups[0]
        plan_trace.append(
            {
                "batch_id": plan.batch_id,
                "mode": plan.mode,
                "context_lengths": plan.context_lengths,
                "query_lengths": query_lengths,
                "prefill_sequence_indices": plan.prefill_sequence_indices,
                "decode_sequence_indices": plan.decode_sequence_indices,
                "sample_sequence_indices": plan.sample_sequence_indices,
                "block_indptr": group.block_indptr,
                "block_ids": group.block_ids,
                "first_logical_blocks": group.first_logical_blocks,
                "append_block_ids": group.append_block_ids,
                "append_block_offsets": group.append_block_offsets,
            }
        )
    if len(plans) != 10:
        raise AssertionError(
            f"native scheduler emitted {len(plans)} plans, expected 10"
        )
    if runner.forward_calls != 10:
        raise AssertionError(
            f"native scheduler used {runner.forward_calls} runner calls, expected 10"
        )
    if [item["context_lengths"] for item in plan_trace] != expected_contexts:
        raise AssertionError("native scheduler emitted unexpected row contexts")
    if [item["query_lengths"] for item in plan_trace] != expected_query_lengths:
        raise AssertionError("native scheduler emitted unexpected query lengths")
    if [item["sample_sequence_indices"] for item in plan_trace] != expected_sample_rows:
        raise AssertionError("native scheduler emitted unexpected sample masks")
    if [item["prefill_sequence_indices"] for item in plan_trace[:3]] != [[0]] * 3:
        raise AssertionError("native prompt chunks were not explicit prefill rows")
    if [item["decode_sequence_indices"] for item in plan_trace[3:]] != [[0]] * 7:
        raise AssertionError("native generated inputs were not explicit decode rows")
    if drive["terminal_event_counts"] != [0, 0] + [1] * 8:
        raise AssertionError(
            "native terminal event masks did not preserve unsampled prefill chunks"
        )
    try:
        plans[0].context_lengths = []
    except AttributeError:
        native_plans_frozen = True
    else:
        raise AssertionError("native Engine exposed a mutable BatchPlan")

    if (
        engine.has_work()
        or engine.ready_request_count != 0
        or engine.in_flight_batch_count != 0
        or engine.poisoned
    ):
        raise AssertionError("native Engine retained work or entered poison state")
    if (
        engine.cache_occupied_block_count != 0
        or engine.cache_free_parent_count != NATIVE_CONTINUATION_MAX_PAGES
    ):
        raise AssertionError("native continuation pages were not fully released")
    if runner.live_cache_count != 0 or runner._inflight is not None:
        raise AssertionError("native continuation runner retained request state")
    if state_pool.active_count != 0 or state_pool.free_count != 1:
        raise AssertionError("native recurrent-state generation was not released")
    if recording_runner.release_calls != [[request_id]]:
        raise AssertionError("native Engine did not release the completed request once")
    engine.close()
    if not engine.closed:
        raise AssertionError("native continuation Engine did not close")

    tokenizer_modules = [
        name for name in ("transformers", "tokenizers") if name in sys.modules
    ]
    if tokenizer_modules:
        raise AssertionError(
            "tokenizer packages entered the native continuation proof: "
            f"{tokenizer_modules}"
        )

    return {
        "protocol": NATIVE_CONTINUATION_PROOF_PROTOCOL,
        "device": device_name,
        "device_capability": capability,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "model_load_ms": model_load_ms,
        "model": {
            "id": MODEL,
            "revision": REVISION,
            "resolved_path": str(runner.model_path),
            "image_cached": True,
        },
        "native_engine": True,
        "native_batch_plans_frozen": native_plans_frozen,
        "asynchronous_engine": engine.asynchronous,
        "engine_closed": engine.closed,
        "scheduler_config": {
            "max_batch_tokens": engine.scheduler_config.max_batch_tokens,
            "prefill_chunk_size": engine.scheduler_config.prefill_chunk_size,
        },
        "cache_pool": {
            "pool_id": engine.cache_pool_id,
            "parent_capacity": engine.cache_parent_capacity,
            "reserved_blocks_after_submit": reserved_blocks,
            "occupied_blocks_after_completion": engine.cache_occupied_block_count,
            "free_parents_after_completion": engine.cache_free_parent_count,
        },
        "prompt_tokens": len(proof_prompt),
        "max_new_tokens": NATIVE_CONTINUATION_MAX_NEW_TOKENS,
        "expected_dense_reference": NATIVE_CONTINUATION_EXPECTED,
        "trajectory": trajectory,
        "exact_dense_reference_match": trajectory == NATIVE_CONTINUATION_EXPECTED,
        "plan_trace": plan_trace,
        "drive": drive,
        "runner_forward_calls": runner.forward_calls,
        "physical_live_cache_count": runner.live_cache_count,
        "recurrent_state_pool": _recurrent_state_pool_snapshot(state_pool),
        "inflight_completion": runner._inflight is not None,
        "tokenizer_modules_loaded": tokenizer_modules,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "cache_contract": {
            "volume": KERNEL_CACHE_VOLUME_NAME,
            "volume_version": KERNEL_CACHE_VOLUME_VERSION,
            "stable_artifacts_checked_by": "run_cache_reuse_probe",
            "mtime_validation": False,
        },
    }


def _assert_kernel_artifacts(report: dict[str, object]) -> dict[str, object]:
    paths = kernel_cache_manifest()
    matches = {
        "fa4_forward_object": [
            path
            for path in paths
            if path.startswith("flash-attention-4/")
            and "/fwd/" in path
            and path.endswith(".o")
        ],
        "flashinfer_page_binding": [
            path
            for path in paths
            if path.startswith("flashinfer/")
            and path.endswith("/cached_ops/page/page.so")
        ],
        "flashinfer_page_hash_manifest": [
            path
            for path in paths
            if path.startswith("flashinfer/")
            and path.endswith("/cached_ops/page/page.so.resin-cache.json")
        ],
        "flashinfer_d64_decode_binding": [
            path
            for path in paths
            if path.startswith("flashinfer/")
            and "/cached_ops/batch_decode" in path
            and "head_dim_qk_64" in path
            and path.endswith(".so")
        ],
        "flashinfer_d64_hash_manifest": [
            path
            for path in paths
            if path.startswith("flashinfer/")
            and "/cached_ops/batch_decode" in path
            and "head_dim_qk_64" in path
            and path.endswith(".so.resin-cache.json")
        ],
    }
    missing = [name for name, artifacts in matches.items() if not artifacts]
    if missing:
        raise AssertionError(f"kernel cache is missing required artifacts: {missing}")
    cache_report = report["kernel_cache"]
    if cache_report["mode"] == "reuse":
        expected_hits = len(matches["flashinfer_page_hash_manifest"]) + len(
            matches["flashinfer_d64_hash_manifest"]
        )
        if (
            cache_report["flashinfer_hash_cache_hits"] < expected_hits
            or cache_report["flashinfer_hash_cache_misses"]
        ):
            raise AssertionError(
                "fresh-container FlashInfer workload did not use only "
                "content-validated artifacts"
            )
    report["kernel_cache"]["required_artifacts"] = matches
    return report


@app.function(gpu="H100!", timeout=20 * 60)
def benchmark() -> dict[str, object]:
    return run_cached_workload(kernel_cache_volume, _benchmark_report)


@app.function(gpu="H100!", timeout=20 * 60)
def kernel_benchmark() -> dict[str, object]:
    return _assert_kernel_artifacts(
        run_cached_workload(kernel_cache_volume, _kernel_benchmark_report)
    )


@app.function(gpu="H100!", timeout=20 * 60, **single_use_options)
def prewarm_kernel_cache() -> dict[str, object]:
    return _assert_kernel_artifacts(
        run_cache_prewarm(kernel_cache_volume, "lfm2-kernel", _kernel_benchmark_report)
    )


@app.function(gpu="H100!", timeout=20 * 60, **single_use_options)
def verify_kernel_cache() -> dict[str, object]:
    return _assert_kernel_artifacts(
        run_cache_reuse_probe(
            kernel_cache_volume, "lfm2-kernel", _kernel_benchmark_report
        )
    )


@app.function(gpu="H100!", timeout=20 * 60, **single_use_options)
def async_completion_proof() -> dict[str, object]:
    return _assert_kernel_artifacts(
        run_cache_reuse_probe(
            kernel_cache_volume,
            "lfm2-kernel",
            _async_completion_report,
        )
    )


@app.function(gpu="H100!", timeout=20 * 60, **single_use_options)
def prewarm_native_continuation_scheduler_cache() -> dict[str, object]:
    return _assert_kernel_artifacts(
        run_cache_prewarm(
            kernel_cache_volume,
            "lfm2-native-continuation-scheduler",
            _native_continuation_scheduler_report,
        )
    )


@app.function(gpu="H100!", timeout=20 * 60, **single_use_options)
def native_continuation_scheduler_proof() -> dict[str, object]:
    return _assert_kernel_artifacts(
        run_cache_reuse_probe(
            kernel_cache_volume,
            "lfm2-native-continuation-scheduler",
            _native_continuation_scheduler_report,
        )
    )


@app.local_entrypoint()
def main(mode: str = "run") -> None:
    if mode == "run":
        report: object = kernel_benchmark.remote()
    elif mode == "baseline":
        report = benchmark.remote()
    elif mode == "prewarm":
        report = prewarm_kernel_cache.remote()
    elif mode == "verify":
        report = verify_kernel_cache.remote()
    elif mode == "probe":
        report = {
            "prewarm": prewarm_kernel_cache.remote(),
            "reuse": verify_kernel_cache.remote(),
        }
    elif mode == "async":
        report = async_completion_proof.remote()
    elif mode == "native-continuation-prewarm":
        report = prewarm_native_continuation_scheduler_cache.remote()
    elif mode == "native-continuation":
        report = native_continuation_scheduler_proof.remote()
    elif mode == "native-continuation-probe":
        report = {
            "prewarm": prewarm_native_continuation_scheduler_cache.remote(),
            "reuse": native_continuation_scheduler_proof.remote(),
        }
    else:
        raise ValueError(
            "mode must be one of: run, baseline, prewarm, verify, probe, async, "
            "native-continuation-prewarm, native-continuation, "
            "native-continuation-probe"
        )
    print(json.dumps(report, indent=2))
