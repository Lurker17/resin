/*
 * Copyright (c) 2026 Resin contributors.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * This wrapper invokes FlashInfer's AIR Top-P implementation through its
 * installed headers. The initialization/workspace layout is adapted from
 * FlashInfer's air_top_p.cuh (Copyright 2026 FlashInfer team, Apache-2.0), whose
 * radix algorithm derives from TensorRT-LLM's samplingAirTopPKernels.cu.
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/include/flashinfer/air_top_p.cuh
 */

#include <flashinfer/air_top_p.cuh>
#include <climits>
#include "tvm_ffi_utils.h"

namespace resin_sampling {

namespace air = flashinfer::sampling::air_top_p;
using Counter = air::Counter<float>;
using Histogram = air::HisT<true, float>;
using tvm::ffi::Optional;

__global__ void InitAirThresholds(Counter* counters, const float* weights,
                                  const float* top_ps, const bool* active,
                                  Histogram* histograms, air::IdxT* count_histograms,
                                  int vocab, const double* guard_mass,
                                  const bool* guard_active, int guard_parts) {
  const int row = blockIdx.x;
  Counter* counter = counters + row;
  const float p = top_ps[row];
  bool enabled = active[row] && p > 0.0f && p < 1.0f;
  // Top-p is redundant when the full inclusive top-k set lies well before
  // its mass boundary. FP64 partials and a 0.001 probability margin keep
  // this decision far from AIR's FP32 histogram rounding. P-only rows and
  // inactive rows do not read scratch or execute these reductions.
  if (enabled && guard_mass && guard_active[row] && p > 0.001f) {
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage total_storage;
    __shared__ typename Reduce::TempStorage retained_storage;
    __shared__ bool redundant;
    double total = 0.0;
    double retained = 0.0;
    const int64_t base = static_cast<int64_t>(row) * guard_parts * 2;
    for (int part = threadIdx.x; part < guard_parts; part += blockDim.x) {
      total += guard_mass[base + part * 2];
      retained += guard_mass[base + part * 2 + 1];
    }
    total = Reduce(total_storage).Sum(total);
    retained = Reduce(retained_storage).Sum(retained);
    if (threadIdx.x == 0) {
      redundant = total > 0.0 && retained >= 0.0 &&
                  retained < (static_cast<double>(p) - 0.001) * total;
    }
    __syncthreads();
    enabled = !redundant;
  }
  if (threadIdx.x == 0) {
    counter->in = weights + static_cast<int64_t>(row) * vocab;
    counter->oriLen = vocab;
    counter->len = enabled ? vocab : 0;
    counter->previousLen = vocab;
    counter->p = p;
    counter->sum = 0;
    counter->kthValueBits = 0;
    counter->finishedBlockCnt = 0;
    counter->filterCnt = 0;
  }
  // All three AIR passes check len before touching histogram or vocabulary
  // memory. A later replay that enables this row initializes every bin here.
  if (!enabled) return;
  const int64_t base = static_cast<int64_t>(row) * air::NUM_BUCKETS;
  for (int i = threadIdx.x; i < air::NUM_BUCKETS; i += blockDim.x) {
    histograms[base + i] = 0;
    count_histograms[base + i] = 0;
  }
}

__global__ void EmitAirThresholds(const Counter* counters, float* output,
                                  int64_t output_stride, int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float cutoff = 0.0f;
  if (counters[row].len > 0) {
    cutoff = air::twiddleOut<float>(counters[row].kthValueBits, false);
  }
  output[static_cast<int64_t>(row) * output_stride] = cutoff;
}

inline size_t Align256(size_t size) { return (size + 255) / 256 * 256; }

inline void CheckCuda(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << "AIR cutoff failed: " << cudaGetErrorString(status);
}

void AirTopPCutoff(TensorView weights, TensorView top_ps, TensorView active,
                    TensorView output, TensorView workspace,
                    Optional<TensorView> guard_mass, Optional<TensorView> guard_active) {
  CHECK_INPUT_AND_TYPE(weights, dl_float32);
  CHECK_INPUT_AND_TYPE(top_ps, dl_float32);
  CHECK_INPUT_AND_TYPE(active, dl_bool);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_CUDA(output);
  CHECK_INPUT_TYPE(output, dl_float32);
  CHECK_DIM(2, weights);
  CHECK_DIM(1, top_ps);
  CHECK_DIM(1, active);
  CHECK_DIM(1, output);
  CHECK_DIM(1, workspace);
  CHECK_DEVICE(weights, top_ps);
  CHECK_DEVICE(weights, active);
  CHECK_DEVICE(weights, output);
  CHECK_DEVICE(weights, workspace);
  const int64_t rows64 = weights.size(0);
  const int64_t vocab64 = weights.size(1);
  TVM_FFI_ICHECK(rows64 >= 0 && rows64 < INT32_MAX);
  TVM_FFI_ICHECK(vocab64 >= air::NUM_BUCKETS && vocab64 < INT32_MAX);
  TVM_FFI_ICHECK_EQ(top_ps.size(0), rows64);
  TVM_FFI_ICHECK_EQ(active.size(0), rows64);
  TVM_FFI_ICHECK_EQ(output.size(0), rows64);
  TVM_FFI_ICHECK_GT(output.stride(0), 0);
  TVM_FFI_ICHECK_EQ(guard_mass.has_value(), guard_active.has_value());
  const double* guard_mass_ptr = nullptr;
  const bool* guard_active_ptr = nullptr;
  int guard_parts = 0;
  if (guard_mass.has_value()) {
    const auto mass = guard_mass.value();
    const auto enabled = guard_active.value();
    CHECK_INPUT_AND_TYPE(mass, dl_float64);
    CHECK_INPUT_AND_TYPE(enabled, dl_bool);
    CHECK_DIM(3, mass);
    CHECK_DIM(1, enabled);
    CHECK_DEVICE(weights, mass);
    CHECK_DEVICE(weights, enabled);
    TVM_FFI_ICHECK_EQ(mass.size(0), rows64);
    TVM_FFI_ICHECK_GT(mass.size(1), 0);
    TVM_FFI_ICHECK_LT(mass.size(1), INT32_MAX);
    TVM_FFI_ICHECK_EQ(mass.size(2), 2);
    TVM_FFI_ICHECK_EQ(enabled.size(0), rows64);
    guard_parts = static_cast<int>(mass.size(1));
    guard_mass_ptr = static_cast<const double*>(mass.data_ptr());
    guard_active_ptr = static_cast<const bool*>(enabled.data_ptr());
  }
  if (rows64 == 0) return;
  const int rows = static_cast<int>(rows64);
  const int vocab = static_cast<int>(vocab64);
  const int buffer_length = air::calcBufLen<float>(vocab);
  const size_t counters_size = Align256(sizeof(Counter) * rows);
  const size_t histogram_size = Align256(sizeof(Histogram) * air::NUM_BUCKETS * rows);
  const size_t count_size = Align256(sizeof(air::IdxT) * air::NUM_BUCKETS * rows);
  const size_t buffer_size = Align256(sizeof(float) * buffer_length * rows);
  const size_t workspace_size = counters_size + histogram_size + count_size + 2 * buffer_size;
  TVM_FFI_ICHECK_GE(static_cast<size_t>(workspace.size(0)), workspace_size);
  auto* base = static_cast<uint8_t*>(workspace.data_ptr());
  auto* counters = reinterpret_cast<Counter*>(base);
  auto* histograms = reinterpret_cast<Histogram*>(base + counters_size);
  auto* counts = reinterpret_cast<air::IdxT*>(base + counters_size + histogram_size);
  auto* buffer1 = reinterpret_cast<float*>(base + counters_size + histogram_size + count_size);
  auto* buffer2 = reinterpret_cast<float*>(base + counters_size + histogram_size + count_size + buffer_size);

  ffi::CUDADeviceGuard device_guard(weights.device().device_id);
  cudaStream_t stream = get_stream(weights.device());
  int sm_count;
  CheckCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                   weights.device().device_id));
  const uint32_t blocks = air::CalcAirTopPBlockNum<true, float>(rows, vocab, sm_count);
  const auto* weights_ptr = static_cast<const float*>(weights.data_ptr());
  const auto* ps_ptr = static_cast<const float*>(top_ps.data_ptr());
  const auto* active_ptr = static_cast<const bool*>(active.data_ptr());
  InitAirThresholds<<<rows, 256, 0, stream>>>(
      counters, weights_ptr, ps_ptr, active_ptr, histograms, counts, vocab,
      guard_mass_ptr, guard_active_ptr, guard_parts);
  CheckCuda(cudaGetLastError());
  const dim3 grid(blocks, rows);
  for (int pass = 0; pass < air::NUM_PASSES<float>; ++pass) {
    air::AirTopPRenormRadixKernel<true, float><<<grid, air::BLOCK_SIZE, 0, stream>>>(
        counters, histograms, counts, pass, buffer1, buffer2);
    CheckCuda(cudaGetLastError());
  }
  EmitAirThresholds<<<(rows + 255) / 256, 256, 0, stream>>>(
      counters, static_cast<float*>(output.data_ptr()), output.stride(0), rows);
  CheckCuda(cudaGetLastError());
}

}  // namespace resin_sampling

TVM_FFI_DLL_EXPORT_TYPED_FUNC(air_top_p_cutoff, resin_sampling::AirTopPCutoff);
