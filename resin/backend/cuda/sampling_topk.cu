/*
 * Copyright (c) 2026 Resin contributors.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * Threshold-only adaptation of FlashInfer's multi-CTA radix top-k launcher
 * (Copyright 2026 FlashInfer team, Apache-2.0). The installed header supplies
 * the radix selector and last-CTA state reset. Unlike the masking kernel, this
 * wrapper reads arbitrary tensor strides and writes just one cutoff per row.
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/include/flashinfer/topk.cuh
 */

#include <algorithm>
#include <climits>
#include <math_constants.h>
#include <flashinfer/math.cuh>
#include <flashinfer/topk.cuh>
#include "tvm_ffi_utils.h"

namespace resin_sampling {

namespace topk = flashinfer::sampling;
constexpr uint32_t kTopKThreads = 1024;
constexpr size_t kTopKFixedBytes = ((sizeof(uint32_t) * (256 + 256 + 5) + 15) / 16) * 16;
static_assert(sizeof(topk::RadixRowState) <= 4096);

template <typename DType, bool SingleCTA>
__global__ void __launch_bounds__(kTopKThreads) TopKThresholdKernel(
    const DType* logits, const int32_t* ks, const bool* active, float* output,
    uint32_t rows, uint32_t vocab, int64_t row_stride, int64_t vocab_stride,
    int64_t k_stride, int64_t active_stride, int64_t output_stride,
    topk::RadixRowState* states, uint32_t chunk_size, uint32_t ctas_per_group) {
  using Traits = topk::RadixTopKTraits<DType>;
  using Ordered = typename Traits::OrderedType;
  const uint32_t group = blockIdx.x / ctas_per_group;
  const uint32_t cta = blockIdx.x % ctas_per_group;
  const uint32_t groups = gridDim.x / ctas_per_group;
  const uint32_t tx = threadIdx.x;
  extern __shared__ uint8_t shared[];
  auto* histogram = reinterpret_cast<uint32_t*>(shared);
  auto* suffix = histogram + 256;
  auto* scalars = suffix + 256;
  auto* ordered = reinterpret_cast<Ordered*>(shared + kTopKFixedBytes);
  auto* state = SingleCTA ? nullptr : states + group;
  int barrier_phase = 0;
  // Count actual selections, not skipped rows: the selector's triple-buffer
  // index must advance exactly when a radix pass clears the next histogram.
  uint32_t selection_iter = 0;

  for (uint32_t row = group; row < rows; row += groups) {
    const int32_t k = ks[static_cast<int64_t>(row) * k_stride];
    if (!active[static_cast<int64_t>(row) * active_stride] || k <= 0 ||
        static_cast<uint32_t>(k) >= vocab) {
      if (cta == 0 && tx == 0) {
        output[static_cast<int64_t>(row) * output_stride] = -CUDART_INF_F;
      }
      continue;
    }
    const uint32_t start = cta * chunk_size;
    const uint32_t size = min(chunk_size, vocab - start);
    const DType* input = logits + static_cast<int64_t>(row) * row_stride;
    // Scalar, thread-strided loads remain coalesced for contiguous vocabulary;
    // no pointer alignment or stride is encoded into a compilation variant.
    for (uint32_t i = tx; i < size; i += kTopKThreads) {
      ordered[i] = Traits::ToOrdered(input[static_cast<int64_t>(start + i) * vocab_stride]);
    }
    __syncthreads();
    uint32_t gt_count = 0;
    uint32_t eq_count = 0;
    const Ordered pivot = topk::RadixSelectFromSharedMemory<
        kTopKThreads, SingleCTA, Ordered, false>(
        ordered, size, static_cast<uint32_t>(k), histogram, suffix, scalars,
        state, barrier_phase, ctas_per_group, cta, tx, selection_iter,
        gt_count, eq_count);
    if (cta == 0 && tx == 0) {
      output[static_cast<int64_t>(row) * output_stride] =
          static_cast<float>(Traits::FromOrdered(pivot));
    }
    ++selection_iter;
  }
  if constexpr (!SingleCTA) {
    // All peer CTAs have exited their last spin before the last finisher clears
    // shared global state. Resetting from CTA zero can deadlock a graph replay.
    topk::RadixGroupResetStateLastCTA<kTopKThreads>(
        state, nullptr, barrier_phase, ctas_per_group, tx);
  }
}

inline void CheckTopKCuda(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "top-k cutoff failed: " << cudaGetErrorString(status);
}

template <typename DType, bool SingleCTA>
void LaunchTopKThresholds(TensorView logits, TensorView ks, TensorView active,
                         TensorView output, TensorView workspace, uint32_t chunk,
                         uint32_t ctas, int sms, cudaStream_t stream) {
  using Ordered = typename topk::RadixTopKTraits<DType>::OrderedType;
  auto kernel = TopKThresholdKernel<DType, SingleCTA>;
  const size_t shared_bytes = kTopKFixedBytes + static_cast<size_t>(chunk) * sizeof(Ordered);
  CheckTopKCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_bytes)));
  int blocks_per_sm;
  CheckTopKCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, kTopKThreads, shared_bytes));
  TVM_FFI_ICHECK_GT(blocks_per_sm, 0) << "top-k cutoff kernel has no resident blocks";
  const uint32_t resident = static_cast<uint32_t>(blocks_per_sm * sms);
  TVM_FFI_ICHECK_GE(resident, ctas)
      << "vocabulary exceeds the resident top-k group capacity";
  uint32_t rows = static_cast<uint32_t>(logits.size(0));
  uint32_t vocab = static_cast<uint32_t>(logits.size(1));
  const uint32_t groups = std::min(rows, resident / ctas);
  auto* states = static_cast<topk::RadixRowState*>(workspace.data_ptr());
  if constexpr (!SingleCTA) {
    CheckTopKCuda(cudaMemsetAsync(states, 0, groups * sizeof(topk::RadixRowState), stream));
  }
  const auto* logits_ptr = static_cast<const DType*>(logits.data_ptr());
  const auto* ks_ptr = static_cast<const int32_t*>(ks.data_ptr());
  const auto* active_ptr = static_cast<const bool*>(active.data_ptr());
  auto* output_ptr = static_cast<float*>(output.data_ptr());
  int64_t row_stride = logits.stride(0);
  int64_t vocab_stride = logits.stride(1);
  int64_t k_stride = ks.stride(0);
  int64_t active_stride = active.stride(0);
  int64_t output_stride = output.stride(0);
  void* arguments[] = {&logits_ptr, &ks_ptr, &active_ptr, &output_ptr, &rows, &vocab,
                       &row_stride, &vocab_stride, &k_stride, &active_stride,
                       &output_stride, &states, &chunk, &ctas};
  const dim3 grid(groups * ctas);
  if constexpr (SingleCTA) {
    CheckTopKCuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid,
                                  dim3(kTopKThreads), arguments, shared_bytes, stream));
  } else {
    // A software group barrier requires all blocks to be resident together.
    // Cooperative launch enforces that contract even with concurrent streams;
    // merely bounding an ordinary grid by SM count is not a residency promise.
    CheckTopKCuda(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), grid,
                                             dim3(kTopKThreads), arguments, shared_bytes, stream));
  }
}

void TopKCutoffs(TensorView logits, TensorView ks, TensorView active,
                TensorView output, TensorView workspace) {
  CHECK_CUDA(logits);
  CHECK_CUDA(ks);
  CHECK_INPUT_TYPE(ks, dl_int32);
  CHECK_CUDA(active);
  CHECK_INPUT_TYPE(active, dl_bool);
  CHECK_CUDA(output);
  CHECK_INPUT_TYPE(output, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(2, logits);
  CHECK_DIM(1, ks);
  CHECK_DIM(1, active);
  CHECK_DIM(1, output);
  CHECK_DIM(1, workspace);
  CHECK_DEVICE(logits, ks);
  CHECK_DEVICE(logits, active);
  CHECK_DEVICE(logits, output);
  CHECK_DEVICE(logits, workspace);
  const int64_t rows = logits.size(0);
  const int64_t vocab = logits.size(1);
  TVM_FFI_ICHECK(rows >= 0 && rows < INT32_MAX && vocab > 0 && vocab < INT32_MAX);
  TVM_FFI_ICHECK_EQ(ks.size(0), rows);
  TVM_FFI_ICHECK_EQ(active.size(0), rows);
  TVM_FFI_ICHECK_EQ(output.size(0), rows);
  TVM_FFI_ICHECK_GE(logits.stride(0), 0);
  TVM_FFI_ICHECK_GE(logits.stride(1), 0);
  TVM_FFI_ICHECK_GE(ks.stride(0), 0);
  TVM_FFI_ICHECK_GE(active.stride(0), 0);
  TVM_FFI_ICHECK_GT(output.stride(0), 0);
  TVM_FFI_ICHECK_GE(workspace.size(0), rows * 4096);
  if (rows == 0) return;
  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  cudaStream_t stream = get_stream(logits.device());
  int sms;
  int max_shared;
  CheckTopKCuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                      logits.device().device_id));
  CheckTopKCuda(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      logits.device().device_id));
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(logits.dtype(), DType, [&] {
    using Ordered = typename topk::RadixTopKTraits<DType>::OrderedType;
    // RadixGroupResetStateLastCTA owns one static shared integer; leave a full
    // allocation granule so dynamic+static storage never exceeds the opt-in cap.
    const size_t available = topk::GetRadixTopKAvailableOrderedSmemBytes<kTopKThreads>(
        max_shared - 256, kTopKFixedBytes, false);
    TVM_FFI_ICHECK_GT(available, sizeof(Ordered));
    const uint32_t capacity = static_cast<uint32_t>(available / sizeof(Ordered));
    const uint32_t ctas = static_cast<uint32_t>((vocab + capacity - 1) / capacity);
    const uint32_t chunk = static_cast<uint32_t>((vocab + ctas - 1) / ctas);
    if (ctas == 1) {
      LaunchTopKThresholds<DType, true>(logits, ks, active, output, workspace,
                                      chunk, ctas, sms, stream);
    } else {
      int cooperative;
      CheckTopKCuda(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                          logits.device().device_id));
      TVM_FFI_ICHECK(cooperative) << "multi-CTA top-k requires CUDA cooperative launch support";
      LaunchTopKThresholds<DType, false>(logits, ks, active, output, workspace,
                                       chunk, ctas, sms, stream);
    }
    return true;
  });
}

}  // namespace resin_sampling

TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_cutoffs, resin_sampling::TopKCutoffs);
