"""Deployment selection stays data-only and preserves the existing V4 default."""

import json

import pytest

from deploy import modal_deepseek_v4 as deployment


def test_checkpoint_specs_keep_old_identity_and_pin_1017():
    old = deployment._model_spec(deployment.DEFAULT_MODEL_KEY)
    assert old.key == "flash-0731"
    assert old.revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert deployment._model_spec("pro-0813").revision == (
        "72e1d3230f6c080a530b0a1d46f8eb4602340597"
    )
    new = deployment._model_spec("flash-1017")
    assert new.repo_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert new.revision == "1bae0e1088993b9b82de08f59de5be5a34e42170"
    assert deployment._model_path(new).parts[-2:] == (new.key, new.revision)


@pytest.mark.parametrize(
    "model_type,runner_name,recipe_name",
    [
        ("deepseek_v4", "DeepSeekV4CudaRunner", "DeepSeekV4Recipe"),
        ("deepseek_v41", "DeepSeekV41CudaRunner", "DeepSeekV41Recipe"),
    ],
)
def test_runner_and_recipe_follow_checkpoint_architecture(
    tmp_path, model_type, runner_name, recipe_name
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    runner, recipe = deployment._implementation(tmp_path)
    assert runner.__name__ == runner_name
    assert recipe.__name__ == recipe_name


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("model", ["flash-0731", "flash-1017"])
def test_legacy_batch_benchmark_rejects_query_only_tp_before_loading(
    monkeypatch, tp, model
):
    def cannot_load(_model):
        raise AssertionError("unsupported topology must fail before checkpoint loading")

    monkeypatch.setattr(deployment, "_require_checkpoint", cannot_load)
    benchmark = getattr(deployment, f"benchmark_tp{tp}")
    with pytest.raises(ValueError, match="one KV head and requires attn_tp=1"):
        benchmark.local(model)


@pytest.mark.parametrize("model", ["flash-0731", "flash-1017"])
def test_legacy_http_rejects_query_only_tp_before_loading(monkeypatch, model):
    monkeypatch.setenv("RESIN_DEEPSEEK_MODEL", model)

    def checkpoint(spec):
        raise AssertionError("unsupported topology must fail before checkpoint loading")

    monkeypatch.setattr(deployment, "_require_checkpoint", checkpoint)
    with pytest.raises(ValueError, match=f"{model} has one KV head"):
        deployment.serve_deepseek_v4.local()


@pytest.mark.parametrize("model", ["flash-0731", "flash-1017"])
def test_legacy_run_rejects_query_only_tp_before_loading(monkeypatch, model):
    def cannot_load(_model):
        raise AssertionError("unsupported topology must fail before checkpoint loading")

    monkeypatch.setattr(deployment, "_require_checkpoint", cannot_load)
    with pytest.raises(ValueError, match="one KV head and requires attn_tp=1"):
        deployment.run_model_tp4.local("test", model_key=model)


def test_legacy_parallel_oracle_rejects_v41_before_loading(monkeypatch):
    def cannot_load(_model):
        raise AssertionError("the old validation must not allocate a V4.1 model")

    monkeypatch.setattr(deployment, "_require_checkpoint", cannot_load)
    with pytest.raises(ValueError, match="V4-0731, not V4.1-1017"):
        deployment._validate_parallel("flash-1017")
