"""Observe retained rows around the HC tap; qualification copies are untimed."""

from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch

from deploy.v41_yoco_probe import (
    ValidationFailure,
    _state_hashes,
    run_probe,
    run_validation,
)


@dataclass
class _Row:
    positions: tuple[int, ...]
    input_rows: int
    states: dict[str, torch.Tensor]


class RetainedActivations:
    def __init__(self, runner, layers=(36, 37, 38)):
        self.runner, self.layers = runner, layers
        self.rows: dict[str, dict[tuple[int, int], _Row]] = {}
        self.current: _Row | None = None
        self._stack = ExitStack()

    def begin_row(self, mode, prompt, start, count, retained):
        first = min(max(prompt - retained - start, 0), count - 1)
        positions = tuple(range(start + first, start + count))
        key = (prompt, start + count)
        row = _Row(positions=positions, input_rows=count, states={})
        self.rows.setdefault(mode, {})[key] = row
        self.current = row

    def end_row(self):
        self.current = None

    def capture(self, name, value):
        if self.current is None:
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                self.capture(name + "." + field.name, getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.capture(f"{name}.{index}", item)
        elif isinstance(value, torch.Tensor):
            positions = self.current.positions
            count = len(positions)
            assert value.ndim >= 2 and value.shape[0] == 1, (name, value.shape)
            if value.shape[1] == self.current.input_rows:
                begin = value.shape[1] - count
            else:
                assert value.shape[1] == count, (name, value.shape, count)
                begin = 0
            self.current.states[name] = value[:, begin:].detach().cpu().clone()

    def __enter__(self):
        assert getattr(self.runner, "_yoco_diagnostic", None) is None
        self.runner._yoco_diagnostic = self
        self._stack.callback(setattr, self.runner, "_yoco_diagnostic", None)
        for index in self.layers:
            layer = self.runner._model.layers[index]

            def before(_module, inputs, *, index=index):
                self.capture(f"layer.{index}.input.hidden", inputs[0])
                self.capture(f"layer.{index}.input.pre", inputs[1])

            def after(_module, _inputs, output, *, index=index):
                self.capture(f"layer.{index}.output.hidden", output[0])
                self.capture(f"layer.{index}.output.pre", output[1])

            self._stack.callback(layer.register_forward_pre_hook(before).remove)
            self._stack.callback(layer.register_forward_hook(after).remove)
            original = layer.mixes

            def mixes(hidden, name, *, original=original, index=index):
                self.capture(f"layer.{index}.mixes.{name}.input", hidden)
                output = original(hidden, name)
                self.capture(f"layer.{index}.mixes.{name}.output", output)
                return output

            layer.mixes = mixes
            self._stack.callback(setattr, layer, "mixes", original)
        return self

    def __exit__(self, *_exception):
        self.current = None
        self._stack.close()

    def report(self):
        modes = list(self.rows)
        report: dict[str, Any] = dict(
            layers=list(self.layers), first_difference=None, comparisons={}
        )
        report["snapshots"] = {
            mode: [
                dict(
                    prompt=key[0],
                    endpoint=key[1],
                    positions=row.positions,
                    state_sha256=_state_hashes(row.states),
                )
                for key, row in rows.items()
            ]
            for mode, rows in self.rows.items()
        }
        if len(modes) != 2:
            return report
        left, right = (self.rows[mode] for mode in modes)
        report["missing_rows"] = {
            modes[0]: sorted(left.keys() - right.keys()),
            modes[1]: sorted(right.keys() - left.keys()),
        }
        for key, reference in left.items():
            if key not in right:
                continue
            candidate = right[key]
            differences = {}
            for name, value in reference.states.items():
                actual = candidate.states.get(name)
                if (
                    actual is None
                    or value.shape != actual.shape
                    or value.dtype != actual.dtype
                ):
                    differences[name] = dict(
                        error="missing field, changed shape or dtype"
                    )
                    continue
                delta = actual.double() - value.double()
                changed = (actual != value).sum().item()
                if changed:
                    differences[name] = dict(
                        max_abs=delta.abs().max().item(),
                        relative_l2=(
                            delta.norm() / value.double().norm().clamp_min(1e-20)
                        ).item(),
                        mismatched=changed,
                        values=value.numel(),
                    )
                    if report["first_difference"] is None:
                        report["first_difference"] = dict(
                            prompt=key[0],
                            endpoint=key[1],
                            field=name,
                            **differences[name],
                        )
            report["comparisons"][str(key)] = dict(
                fields_compared=len(reference.states), differences=differences
            )
        return report


def run_prefill_diagnostic(
    runner, engine, prompt_tokens, *, coordinate=None, modes=None
):
    """One fresh mask/compact prefill trajectory, without autoregressive decode."""
    with RetainedActivations(runner) as trace:
        try:
            result = run_probe(
                runner,
                engine,
                prompt_tokens,
                coordinate=coordinate,
                modes=modes,
                generated=1,
                prefill_only=True,
            )
        except Exception as error:
            setattr(error, "activation_diagnostics", trace.report())
            raise
        result["activation_diagnostics"] = trace.report()
        return result


def run_diagnostic_validation(raw, engine, checkpoint, metadata, groups):
    """Prefill tap diagnostic, then full qualification on the same Engine/load."""
    diagnostic = run_validation(
        raw,
        engine,
        checkpoint,
        metadata,
        groups,
        probe=run_prefill_diagnostic,
        validation_name="bounded_swa_replay_prefill_diagnostic",
    )
    try:
        return diagnostic + run_validation(raw, engine, checkpoint, metadata, groups)
    except ValidationFailure as error:
        error.validation_records = diagnostic + error.validation_records
        raise
