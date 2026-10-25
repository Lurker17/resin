"""Bounded-memory, rank-local checkpoint reads and NVFP4 scale preparation."""

import torch

from resin.backend.safetensors import SafetensorReader, indexed_tensor_files


class WeightReader:
    def __init__(self, root, device):
        self.files = indexed_tensor_files(root, require_index=True)
        self.reader = SafetensorReader()
        self.device = torch.device(device)
        self.loaded_bytes = 0

    def close(self):
        self.reader.close()

    def read(self, name, shape=None, dtype=None, *, cpu=False, slices=None):
        if name not in self.files:
            raise ValueError(f"checkpoint missing {name}")
        tensor = (
            self.reader.read(self.files[name], name)
            if slices is None
            else self.reader.read_slice(self.files[name], name, slices, shape=shape)
        )
        if slices is None and shape is not None and tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name}: expected {shape}, found {tuple(tensor.shape)}")
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name}: expected {dtype}, found {tensor.dtype}")
        self.loaded_bytes += tensor.numel() * tensor.element_size()
        return tensor if cpu else tensor.to(self.device)

    def prefix(self, prefix, *, exclude=()):
        return {
            key[len(prefix) :]: self.read(key)
            for key in sorted(self.files)
            if key.startswith(prefix) and not any(part in key for part in exclude)
        }


def nvfp4_expert_scales(gate, up, down, input_scale, output_input_scale, limit):
    """SGLang's ModelOpt TRT-LLM adapter, preserving raw checkpoint block scales.

    Source: SGLang baa9befa381511cd4d56f83980951e5a45fffc6d,
    modelopt_quant._compute_gemm1_alphas and flashinfer_trtllm._compute_g1_scale_c.
    The fused epilogue has separate gate and up alphas. Its clamp is in the
    unscaled accumulator domain, calibrated by the gate alpha. In this checkpoint
    gate/up global scales are equal; do not requantize E4M3 blocks to enforce it.
    """
    gate_alpha, up_alpha = gate.float() * input_scale, up.float() * input_scale
    return {
        "input_scale_quant": torch.as_tensor(
            input_scale, dtype=torch.float32, device=gate.device
        ).reciprocal(),
        "g1_alphas": gate_alpha,
        "g1_scale_c": up_alpha / output_input_scale,
        "g2_alphas": down.float() * output_input_scale,
        "clamp_limits": limit / gate_alpha,
    }
