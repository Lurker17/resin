"""Block32 FP8 projections with explicit attention-TP coordinates."""

import torch
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelContext
from resin_models_cuda.deepseek_v41.kernels import act_quant, fp8_gemm


class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        dtype=torch.float8_e4m3fn,
        tensor_parallel=None,
        axis=None,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel or TensorParallelContext()
        self.axis = axis
        if axis == 0:
            out_features //= self.tensor_parallel.size
        elif axis == 1:
            in_features //= self.tensor_parallel.size
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if dtype == torch.float8_e4m3fn:
            self.scale = nn.Parameter(
                torch.empty(
                    (out_features + 31) // 32,
                    (in_features + 31) // 32,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        else:
            self.register_parameter("scale", None)
        self.allreduce = self.tensor_parallel.sum

    def forward(self, value):
        if self.scale is None:
            output = F.linear(value.to(self.weight.dtype), self.weight)
        else:
            quantized, scales = act_quant(value, 32, "ue8m0", torch.float8_e8m0fnu)
            output = fp8_gemm(
                quantized,
                scales,
                self.weight,
                self.scale,
                torch.float8_e8m0fnu,
                block_size=32,
            )
        return self.allreduce(output) if self.axis == 1 else output


class SharedExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = Linear(args.dim, args.moe_inter_dim)
        self.w2 = Linear(args.moe_inter_dim, args.dim)
        self.w3 = Linear(args.dim, args.moe_inter_dim)
        self.limit = args.swiglu_limit

    def forward(self, value):
        gate, up = self.w1(value).float(), self.w3(value).float()
        if self.limit:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        return self.w2((F.silu(gate) * up).to(value.dtype))
