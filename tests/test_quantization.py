import pytest

from resin.backend.quantization import (
    BF16,
    MXFP4,
    NVFP4,
    QuantizationConfig,
    WeightEncoding,
)


def test_quantization_config_resolves_exact_semantics_without_a_registry() -> None:
    config = QuantizationConfig(
        {
            "transformer.experts": MXFP4,
            "vision.projection": NVFP4,
        }
    )

    assert config.resolve("transformer.experts") == MXFP4
    assert config.resolve("vision.projection") == NVFP4
    assert config.resolve("transformer.attention") == BF16


def test_quantization_config_validates_checkpoint_encoding() -> None:
    config = QuantizationConfig({"transformer.experts": MXFP4})

    assert config.require("transformer.experts", MXFP4) == MXFP4
    with pytest.raises(ValueError, match="quantization config selected"):
        config.require("transformer.experts", BF16)


def test_quantization_declarations_are_frozen_and_validated() -> None:
    components = {"experts": MXFP4}
    config = QuantizationConfig(components)
    components.clear()
    assert config.resolve("experts") == MXFP4

    with pytest.raises(TypeError):
        config.components["attention"] = BF16  # type: ignore[index]  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="group_size"):
        WeightEncoding("packed", group_size=0)
    with pytest.raises(TypeError, match="WeightEncoding"):
        QuantizationConfig({"experts": "mxfp4"})  # type: ignore[dict-item]  # ty: ignore[invalid-argument-type]
