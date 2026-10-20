"""Ownership of submitted CUDA work until its sampled output is readable."""

from __future__ import annotations

import torch

from resin.backend.cuda.failures import CudaFailureGuard
from resin.backend.cuda.staging import TensorStager


class CudaStep:
    """Retain the plan, device output and staging through one completion event.

    Enqueue on the execution stream after sampling. The event covers model work
    and the nonblocking result copy. Graph outputs are borrowed until this step
    completes; its runner must prevent another replay from overwriting them.
    CPU tensors also work, allowing execution-contract tests without CUDA.
    """

    def __init__(self, plan, output, staging: TensorStager, guard: CudaFailureGuard):
        self.plan = plan
        self.output = output
        self.staging = staging
        self._guard = guard
        self._result = None
        self.completion = None
        self.host_output = output
        if output.is_cuda:
            self.host_output = staging.output("sampled_tokens", output)
            self.completion = torch.cuda.Event()
            self.completion.record(torch.cuda.current_stream(output.device))

    @property
    def done(self) -> bool:
        """Whether the caller has collected this step's host output."""

        return self._result is not None

    def ready(self) -> bool:
        with self._guard:
            return self.completion is None or self.completion.query()

    def result(self) -> list[int] | list[list[int]]:
        if not self.done:
            with self._guard:
                if self.completion is not None:
                    self.completion.synchronize()
                self._result = self.host_output.tolist()
        return self._result
