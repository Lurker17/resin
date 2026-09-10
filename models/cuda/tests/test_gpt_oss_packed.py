from __future__ import annotations

import torch

from resin.moe_kernels import MoEKernelConfig


class TestGptOssPackedBridge:
    def test_official_block_layout_is_flattened_and_installed(self) -> None:
        from resin_models_cuda.gpt_oss.mxfp4 import (
            _install_expert_layer,
            _make_kernel_expert_factory,
        )

        config = MoEKernelConfig(
            architecture="sm90",
            quantization="mxfp4",
            activation="swiglu",
            top_k=2,
            num_experts=4,
            hidden_size=32,
            intermediate_size=32,
            group_size=32,
            expert_bias=True,
            activation_alpha=1.702,
            activation_limit=7.0,
        )

        class Kernel:
            def __init__(self) -> None:
                self.prepared = []

            def prepare_weights(self, weights, actual_config):
                self.prepared.append((weights, actual_config))
                return weights

            def forward(self, hidden, routing, weights, actual_config):
                del routing, weights, actual_config
                return hidden

        class Args:
            num_local_experts = 4
            hidden_size = 32
            intermediate_size = 32
            num_experts_per_tok = 2

        kernel = Kernel()
        factory = _make_kernel_expert_factory(kernel, config)

        class MLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = factory(Args())

        class Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = MLP()

        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer()])

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Backbone()

        model = Model()
        tensors = {
            "gate_up_proj_blocks": torch.zeros(4, 64, 1, 16, dtype=torch.uint8),
            "gate_up_proj_scales": torch.zeros(4, 64, 1, dtype=torch.uint8),
            "gate_up_proj_bias": torch.zeros(4, 64),
            "down_proj_blocks": torch.zeros(4, 32, 1, 16, dtype=torch.uint8),
            "down_proj_scales": torch.zeros(4, 32, 1, dtype=torch.uint8),
            "down_proj_bias": torch.zeros(4, 32),
        }
        _install_expert_layer(model, kernel, config, 0, tensors)

        assert len(kernel.prepared) == 1
        weights, actual_config = kernel.prepared[0]
        assert actual_config is config
        assert weights.gate_up.shape == (4, 64, 16)
        assert weights.down.shape == (4, 32, 16)
        assert model.model.layers[0].mlp.experts.is_prepared
