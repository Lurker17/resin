`glm53_vision_hf.npz` is an offline oracle generated from Hugging Face Transformers
revision `50e35cc0ca18ec19b80653690de553e6decf74f5`,
`src/transformers/models/glm5_next/modeling_glm5_next.py` (Apache-2.0).
It stores the complete config, float32 state dict, patch input, image grids, and
`Glm5NextVisionModel(...).pooler_output` for a two-image batch. It contains no
downloaded model weights: weights are initialized using `torch.manual_seed(100)`.
The config uses hidden size 16, intermediate size 24, depth 2, heads 2, output
size 24, projection intermediate size 32, and patch size 2. Other defaults come
from `Glm5NextVisionConfig`; attention is `eager`.

Before generating the reference, the first block's gate-projection bias is filled
with 12 and its up-projection bias with `torch.linspace(-12, 12, 24)` to exercise
SwiGLU saturation. Pixels are the next `torch.randn(24, 24)` draw and grids are
`[[1, 4, 4], [1, 2, 4]]`. Generation uses PyTorch 2.13.0 and stores arrays with
`numpy.savez_compressed`. `test_glm53_vision.py` consumes the fixture without any
network or Transformers dependency.

`output_bf16` repeats the upstream eager forward after converting the stored
weights and inputs to BF16. It is stored as float32 for portable NumPy loading;
the test runs Resin with both FP32 and BF16 weights.
