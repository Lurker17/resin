# DeepSeek-V4 kernel reuse

Resin delegates FP8 projection quantization to `tile-kernels==1.0.0` and keeps
local kernels where they preserve a useful fusion or a model-specific storage
contract. The decision includes output preparation and numerical behavior, not
just whether an upstream function has a similar name. Measurements below were
taken on an NVIDIA B300 SXM6 AC on 2026-09-11.

## Adopted: TileKernels FP8 quantization

`act_quant` in `models/cuda/resin_models_cuda/deepseek_v4/cutile.py` delegates
out-of-place projection quantization to
[TileKernels `per_token_cast`](https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/quant/per_token_cast_kernel.py).
It requests E4M3 output, groups of 128, power-of-two rounding, and TMA-aligned
column-major scales. The maximum-magnitude floor remains `1e-4`; the scale is
`2**ceil(log2(max(amax, 1e-4) / 448))`. See the upstream
[scale calculation and layout helpers](https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/quant/common.py).

The quantized payload is unchanged. Scale representations match their consumers:

- DeepGEMM decode receives four UE8M0 bytes packed into each `int32`, with
  logical shape `[rows, K/512]` and strides `[1, aligned_rows]`.
- TensorRT-LLM prefill receives the same power-of-two scale values as FP32,
  with logical shape `[rows, K/128]` and column-major storage. `gemm.py` accepts
  these directly; for aligned rows, its transpose is already contiguous and
  does not copy. The existing unaligned-row padding path remains.

This removes three local quantization kernels and the projection-scale
allocation/dispatch branches. A merged local kernel was also tested: it
preserved output but offered no consistent speed improvement.

`tilelang_kernels.py` owns the import and a Linux-only compatibility alias:
TileKernels 1.0.0 imports `tilelang.utils.target`, while the pinned
`tilelang==0.1.14` exposes `tilelang.backend.target`. The alias uses
`sys.modules.setdefault` immediately before the TileKernels import. Remove it
when a pinned TileKernels release corrects that import; it is not a general
dependency-loading mechanism.

All **38 shape/layout cases** passed exact quantized-byte and scale-value
comparisons, covering 19 dimensions with both packed and unpacked scales,
widths 1024/4096/7168, and rows through 2048. Packed scale strides also matched.

Median microseconds at width 4096:

| Rows | Old packed | TileKernels packed | Old FP32 scale preparation | TileKernels FP32 scales |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.730 | 1.281 | 2.733 | 1.282 |
| 6 | 1.792 | 1.287 | 5.890 | 1.299 |
| 32 | 2.883 | 1.388 | 6.592 | 1.408 |
| 512 | 31.770 | 2.570 | 36.854 | 2.565 |

The FP32 comparison includes the old scale cast and transpose/contiguous
preparation; TileKernels produces the required layout directly. These timings
exclude GEMM execution and its row-padding work.

## Retained and simplified: routing

Hash routing now gathers the six table-selected logits into a padded tile,
instead of evaluating every expert and reducing six masked copies. It retains
the original left-associated six-term normalization sum, applies `route_scale`
before BF16 conversion, and excludes padded lanes from normalization and stores.
Across **216 cases**, expert IDs and BF16 weights matched the original exactly,
including NaN masks. Coverage included FP32/BF16 logits, 256/384 experts,
duplicate experts, strided tables, and scales 1/2.5.

The shared `_sqrt_softplus` helper also fixes a numerical defect in both hash
and top-six routing. For sufficiently negative logits, `1 + exp(-abs(x))`
rounds to one in FP32. Its logarithm becomes zero, and all-negative rows such
as -20 or -80 could produce NaN normalized weights. The helper compensates for
that addition's rounding, preserving small positive scores without choosing an
arbitrary approximation threshold. It retains the existing softplus threshold
and BF16 output boundary.

The compensated implementation matched all **33,516 reference BF16 weights in
63 cases** exactly, with zero nonfinite results and unchanged expected expert
IDs. The uncompensated baseline differed on 11,905 weights, including 9,576
nonfinite values. Maximum candidate absolute error and BF16-ULP difference were
both zero on this test set.

Separate within-run comparisons, in microseconds, for 256 experts:

| Change | 1 row | 6 rows | 32 rows | 512 rows |
| --- | ---: | ---: | ---: | ---: |
| Original hash → gathered hash | 1.986 → 1.571 | 1.988 → 1.606 | 2.046 → 1.647 | 6.128 → 1.861 |
| Gathered hash → compensated hash | 1.506 → 1.507 | 1.536 → 1.538 | 1.647 → 1.665 | 1.857 → 1.889 |
| Top-six → compensated top-six | 2.944 → 2.979 | 2.912 → 2.976 | 3.040 → 3.088 | 9.981 → 10.212 |

The rows come from separate runs; their absolute baselines should not be mixed.

## Rejected: FlashInfer hash-routing adapter

Pinned FlashInfer 0.6.15.post1 already provides
[`hash_topk`](https://github.com/flashinfer-ai/flashinfer/blob/c05407ceffb7d9ce111a73553a8e37ad752adb62/flashinfer/fused_moe/hash_topk.py),
but its routed weights are unscaled FP32. Resin's fused expert interface
requires BF16 weights after route scaling. Moving scale application into the
expert computation would change the rounding boundary.

Both an ordinary Torch scale/cast adapter and a compiled fused adapter matched
FP32 and FP64 reference outputs exactly across **84 cases, 44,688 weights
each**. All 252 ID checks and 48 input-mutation graph replay checks passed.
However, with 256 experts and scale 2.5, the already-gathered local baseline
took **1.484 µs**, versus **3.588 µs** for the ordinary adapter and **2.347 µs**
for the compiled adapter at one row. At 512 rows the same comparison was
**1.839 / 3.763 / 2.755 µs**. These adapter timings are not comparisons against
the original full-expert-load router. The local compensated path was retained.

FlashInfer's similarly named
[`fused_topk_deepseek`](https://github.com/flashinfer-ai/flashinfer/blob/c05407ceffb7d9ce111a73553a8e37ad752adb62/flashinfer/fused_moe/fused_routing_dsv3.py)
uses DSV3 sigmoid routing, so it does not replace V4 sqrt-softplus top-six.

## Why other local kernels remain

- `moe_act_quant` emits each K128 UE8M0 scale byte four times in FlashInfer's
  K32 scale layout. Keeping this fused with quantization avoids a separate
  layout conversion; the tensor's byte interpretation is part of the contract.
- In-place FP8 and FP4 operations simulate quantization and dequantization in
  the original dtype. Their K64/K32 groups, magnitude floors, and rounding
  boundaries differ from the adopted out-of-place FP8 projection operation.
- mHC pre-normalization includes the final RMSNorm of the mixed layer input.
  FlashInfer's [`mhc_pre_big_fuse`](https://github.com/flashinfer-ai/flashinfer/blob/c05407ceffb7d9ce111a73553a8e37ad752adb62/flashinfer/mhc.py)
  does not include that downstream normalization. Splitting it would add work
  and change the normalization boundary. The existing FlashInfer mHC post
  operation remains in use.
- Compressor, sparse-index mapping, normalization/RoPE, and cache-store fusions
  retain Resin's packed-page addressing and speculative state semantics.
  Replacing only their arithmetic with separate upstream calls would not
  preserve those complete operations.

## Measurement and validation status

The [main kernel probe](https://modal.com/apps/ekzhang/main/ap-gPLvO17oGt08HE7pKwUyzN)
and [stable-softplus probe](https://modal.com/apps/ekzhang/main/ap-z8UONDw9y8FuxQfprYLk95)
used CUDA events around repeated captured work, after compilation and warmup.
Each comparison used seven rounds with candidate order alternated or rotated.
Quantization used 32 operations per graph at rows ≤128 and four above that,
with 100 replays per round. Routing used 64 operations per graph, with 20
replays for gather and 50 for adapter/softplus comparisons. The largest observed
seven-round spans among the selected measurements were 0.025 µs for
quantization, 0.019 µs for gather, 0.014 µs for adapters, and 0.0083 µs for
softplus. These are observed spreads, not confidence intervals or predictions
of end-to-end throughput.

The [production kernel validation](https://modal.com/apps/ekzhang/main/ap-Tq4N5EXgkM2yAz2afcRIjN)
passes all **36 GPU tests**. These cover quantization payloads and scale layouts
with BF16/FP32 and 3D inputs, unaligned prefill, fused normalization/RoPE and
packed cache writes, sparse-index mapping, speculative attention windows,
rejected compressor state, and negative-logit routing. The local suite passes
**342 tests**, with 38 GPU-only cases skipped; lint, format, and whitespace
checks pass. No Rust source changed in this pass.

Four full-model runs use DeepSeek-V4-Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, TP=4 on B300, the same 8,192-token
prompt corpus, 128 output tokens, temperature zero, seed zero, and five DSpark
draft tokens. All four pass tensor-parallel token agreement and the script's
cached-versus-fresh continuation check. That check compares **one continuation
token within each run**, not complete output across runs. All four capture one
decode graph with zero graph breaks/fallbacks and 120,867,840 bytes of graph
capture memory.

| Run | Decode tokens/s | First token, ms | Median device time per decode block, ms |
| --- | ---: | ---: | ---: |
| [Baseline `7a478f7`](https://modal.com/apps/ekzhang/main/ap-ythWuMXJqP1PkzCfYKMjOy) | 160.14 | 736.58 | 10.949 |
| [Identical baseline repeated](https://modal.com/apps/ekzhang/main/ap-6q9onpW2mGa9ayEX8mQfxV) | 212.62 | 710.77 | 10.782 |
| [Final implementation](https://modal.com/apps/ekzhang/main/ap-5qvFv7uvtutVkfLytm9G31) | 209.14 | 734.35 | 10.705 |
| [Final with only old softplus restored](https://modal.com/apps/ekzhang/main/ap-tP3SNOWpviXCW5wIVPlcEH) | 273.87 | 733.15 | 10.817 |

**Full-model output equivalence remains unestablished.** Generated text differs
between all four runs, including the two identical baselines. Reverting only
softplus did not reproduce the first baseline either. The unchanged baseline
therefore already has cross-run variability in this profile workload; its cause
has not been isolated. The script returns the timed request after warmup and
prefix-cache probes. Those runs take different speculative paths, so their
tokens-per-second values do not establish an end-to-end speedup or regression.
The kernel measurements above support the narrower performance claims.

The next repeatability diagnostic should compare fixed token IDs and logit
margins without profile/cache history or speculative decoding, then compare
layer activations at the first divergence. The routing correction is separately
validated against Torch, and the quantizer against exact payload/scale references;
neither test substitutes for that full-model investigation.

To repeat the persistent checks and model smoke:

```bash
uv run pytest -q
uv run modal run deploy/modal_deepseek_v4.py --mode validate-kernels
uv run modal run deploy/modal_deepseek_v4.py --mode profile \
  --input-tokens 8192 --max-new-tokens 128 --temperature 0 \
  --speculative-tokens 5 --batch-size 1
```
