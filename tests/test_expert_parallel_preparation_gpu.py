"""Independent old-Torch oracles for capturable expert transport preparation."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def reference_prepare(
    hidden, indices, weights, capacity, source, valid_rows, tp, per_rank
):
    def pad(value):
        out = value.new_zeros((capacity, *value.shape[1:]))
        if source:
            out[: value.shape[0]].copy_(value)
        return out

    hidden, indices, weights = (pad(value) for value in (hidden, indices, weights))
    active = torch.arange(capacity, device=hidden.device) < valid_rows
    active &= source
    owners = (
        indices.div(per_rank, rounding_mode="floor")[..., None] * tp
        + torch.arange(tp, device=hidden.device)
    ).flatten(1)
    return hidden, indices, weights, owners, active


def reference_routes(owners, active, indices, weights, world, rank, experts, limit=22):
    capacity = owners.shape[0]
    if owners.shape[1] <= limit:
        routes = [owners.sort(dim=1).values]
    else:
        destinations = torch.zeros(
            capacity, world, dtype=torch.bool, device=owners.device
        )
        destinations.scatter_(1, owners.long(), True)
        ranks = torch.arange(world, device=owners.device, dtype=torch.int32)
        routes = [
            torch.where(
                destinations[:, start : start + limit],
                ranks[start : start + limit],
                world,
            )
            for start in range(0, world, limit)
        ]
    outputs = []
    for route in routes:
        first = route.min(dim=1, keepdim=True).values
        selected = active & (first[:, 0] < world)
        route = torch.where(route < world, route, first)
        route = torch.where(selected[:, None], route, rank)
        k = next(k for k in (1, 2, 4, 6, 8, 10, 16, 22) if k >= route.shape[1])
        if k != route.shape[1]:
            route = torch.cat((route, route[:, :1].expand(-1, k - route.shape[1])), 1)
        outputs.append(
            (
                route.int().contiguous(),
                indices.masked_fill(~selected[:, None], experts).int(),
                weights.masked_fill(~selected[:, None], 0).float(),
            )
        )
    return outputs


def reference_receive(hidden, indices, weights, experts, dtype):
    valid = indices[..., 0] < experts
    return (
        hidden.masked_fill(~valid[..., None], 0).flatten(0, 1),
        weights.masked_fill(~valid[..., None], 0).to(dtype).flatten(0, 1),
    )


def same(actual, expected):
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            same(a, b)
    else:
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call()
    return graph, result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_padding_owners_active_matches_torch_for_strided_and_changed_rows(
    dtype, source, id_dtype
):
    from resin.backend.cuda.expert_parallel_kernels import prepare_dispatch

    for rows, capacity, tp in ((0, 8, 1), (1, 8, 4), (5, 8, 4), (17, 32, 2)):
        hidden = torch.randn(rows, 514, device="cuda", dtype=dtype)[:, ::2]
        indices = torch.randint(0, 32, (rows, 16), device="cuda", dtype=id_dtype)[
            :, ::2
        ]
        weights = torch.randn(rows, 16, device="cuda", dtype=dtype)[:, ::2]
        valid = torch.tensor(rows, device="cuda", dtype=torch.int32)

        def call():
            return prepare_dispatch(
                hidden,
                indices,
                weights,
                capacity,
                source=source,
                valid_rows=valid,
                expert_tp=tp,
                experts_per_rank=8,
            )

        same(
            call(),
            reference_prepare(hidden, indices, weights, capacity, source, valid, tp, 8),
        )
        graph, output = capture(call)
        hidden.neg_()
        weights.mul_(-0.5)
        indices.remainder_(8)
        valid.zero_()
        graph.replay()
        same(
            output,
            reference_prepare(hidden, indices, weights, capacity, source, valid, tp, 8),
        )
        valid.fill_(max(rows - 1, 0))
        graph.replay()
        same(
            output,
            reference_prepare(hidden, indices, weights, capacity, source, valid, tp, 8),
        )
        # Optional valid-row pointer must preserve ordinary source-row semantics.
        same(
            prepare_dispatch(
                hidden,
                indices,
                weights,
                capacity,
                source=source,
                valid_rows=None,
                expert_tp=tp,
                experts_per_rank=8,
            ),
            reference_prepare(hidden, indices, weights, capacity, source, rows, tp, 8),
        )


@pytest.mark.parametrize(
    "world,topk,tp",
    [(4, 1, 1), (4, 3, 1), (4, 8, 1), (4, 8, 4), (64, 8, 8), (64, 1, 64)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("limit", [1, 2, 22])
def test_routes_payloads_exact_duplicates_idle_and_multipass(
    world, topk, tp, dtype, limit
):
    from resin.backend.cuda.expert_parallel_kernels import prepare_routes

    rows, experts, rank = 9, world * 8 // tp, world - 1
    ids = torch.randint(0, experts, (rows, topk * 2), device="cuda")[:, ::2]
    weights = torch.randn(rows, topk * 2, device="cuda", dtype=dtype)[:, ::2]
    owners = (
        ids.div(8, rounding_mode="floor")[..., None] * tp
        + torch.arange(tp, device="cuda")
    ).flatten(1)
    # Include high destination63, repeated routes, and completely idle origins.
    owners[-1].fill_(world - 1)
    active = torch.arange(rows, device="cuda") % 2 == 0

    def call():
        return list(
            prepare_routes(
                owners,
                active,
                ids,
                weights,
                world_size=world,
                rank=rank,
                num_experts=experts,
                max_destinations_per_pass=limit,
            )
        )

    same(
        call(),
        reference_routes(owners, active, ids, weights, world, rank, experts, limit),
    )
    graph, output = capture(call)
    owners.fill_(0)
    active.logical_not_()
    weights.neg_()
    ids.zero_()
    graph.replay()
    same(
        output,
        reference_routes(owners, active, ids, weights, world, rank, experts, limit),
    )
    if topk * tp <= limit:
        # The original non-deduplicated path accepts no-destination sentinels.
        owners.fill_(world)
        active.fill_(True)
        graph.replay()
        same(
            output,
            reference_routes(owners, active, ids, weights, world, rank, experts, limit),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_receive_cleanup_preserves_callback_rows_strides_sentinels_and_dtype(
    dtype, weight_dtype
):
    from resin.backend.cuda.expert_parallel_kernels import prepare_receive

    hidden = torch.randn(4, 5, 514, device="cuda", dtype=dtype)[:, :, ::2]
    weights = torch.randn(4, 5, 16, device="cuda")[:, :, ::2]
    ids = torch.randint(0, 288, (4, 5, 16), device="cuda", dtype=torch.int32)[:, :, ::2]
    ids[:, 1, 0] = 288
    hidden[:, 1].fill_(float("nan"))
    weights[:, 1].fill_(float("nan"))

    def call():
        return prepare_receive(
            hidden, ids, weights, num_experts=288, weight_dtype=weight_dtype
        )

    same(call(), reference_receive(hidden, ids, weights, 288, weight_dtype))
    graph, output = capture(call)
    ids[:, :, 0].fill_(288)
    graph.replay()
    same(output, reference_receive(hidden, ids, weights, 288, weight_dtype))
    hidden.fill_(3.25)
    weights.fill_(-0.75)
    ids.zero_()
    graph.replay()
    same(output, reference_receive(hidden, ids, weights, 288, weight_dtype))
    assert all(value.is_contiguous() for value in output)


def test_preparation_reuses_compiled_variants_for_new_rows_capacities_strides_and_valid_values():
    from resin.backend.cuda import expert_parallel_kernels as kernels

    def variants():
        return {
            name: set(
                getattr(kernels, name).device_caches[torch.cuda.current_device()][0]
            )
            for name in ("_prepare", "_routes", "_receive")
        }

    def run(rows, capacity, stride, valid_count):
        hidden = torch.randn(rows, 256 * stride, device="cuda", dtype=torch.bfloat16)[
            :, ::stride
        ]
        indices = torch.randint(0, 288, (rows, 8 * stride), device="cuda")[:, ::stride]
        weights = torch.rand(rows, 8 * stride, device="cuda")[:, ::stride]
        valid = torch.tensor(valid_count, device="cuda", dtype=torch.int32)
        prepared = kernels.prepare_dispatch(
            hidden,
            indices,
            weights,
            capacity,
            source=True,
            valid_rows=valid,
            expert_tp=4,
            experts_per_rank=288,
        )
        list(
            kernels.prepare_routes(
                prepared[3],
                prepared[4],
                prepared[1],
                prepared[2],
                world_size=4,
                rank=0,
                num_experts=288,
            )
        )
        received_h = torch.randn(
            4, capacity, 256 * stride, device="cuda", dtype=torch.bfloat16
        )[:, :, ::stride]
        received_i = torch.zeros(
            4, capacity, 8 * stride, device="cuda", dtype=torch.int32
        )[:, :, ::stride]
        received_w = torch.rand(4, capacity, 8 * stride, device="cuda")[:, :, ::stride]
        kernels.prepare_receive(
            received_h,
            received_i,
            received_w,
            num_experts=288,
            weight_dtype=torch.float32,
        )
        torch.cuda.synchronize()

    run(1, 8, 1, 1)
    baseline = variants()
    assert all(baseline.values())
    for args in ((3, 8, 2, 0), (17, 32, 3, 13), (65, 128, 1, 64)):
        run(*args)
        assert variants() == baseline
