"""One-in-flight asynchronous CUDA execution and lifetime safety."""

from __future__ import annotations

from time import monotonic, sleep

from resin.backend.cuda.staging import TensorStager

CUDA_COMPLETION_QUARANTINE: list[dict[str, object]] = []
_ABANDON_QUERY_TIMEOUT_SECONDS = 0.010
_ABANDON_QUERY_INTERVAL_SECONDS = 0.001


class CudaCompletion:
    def __init__(
        self,
        owner,
        plan,
        resources,
        stream,
        quarantine,
        staging,
        start_event,
        done_event,
    ) -> None:
        self._owner = owner
        self._plan = plan
        self._resources = tuple(resources)
        self._stream = stream
        self._quarantine = quarantine
        self.staging = staging
        self.ephemeral = []
        self._output_host = None
        self._start_event = start_event
        self._done_event = done_event
        self._done_recorded = False
        self._dispatch_error = None
        self._accepted = False
        self._terminal = False
        self._retired = False

    def retain(self, *values: object) -> None:
        self.ephemeral.extend(values)

    def _release_lifetimes(self) -> None:
        flags = {"_owner", "_accepted", "_done_recorded", "_terminal", "_retired"}
        for name in self.__dict__.keys() - flags:
            setattr(self, name, None)
        self._retired = True

    def _retire(self) -> None:
        if not self._terminal:
            raise RuntimeError("a pending CUDA completion cannot be retired")
        self._release_lifetimes()

    def _quarantine_lifetimes(self, wait_error) -> None:
        quarantine = self._quarantine
        payload = self.__dict__.copy()
        payload.pop("_quarantine")
        payload["wait_error"] = wait_error
        quarantine.append(payload)
        self._terminal = True
        self._release_lifetimes()

    def _abandon(self) -> None:
        if self._retired:
            return
        if not self._accepted:
            self._terminal = True
            self._retire()
            return
        fence = self._done_event if self._done_recorded else self._stream
        fence_name = (
            "terminal CUDA event" if self._done_recorded else "private CUDA stream"
        )

        deadline = monotonic() + _ABANDON_QUERY_TIMEOUT_SECONDS
        while True:
            try:
                if fence.query():
                    break
            except BaseException as error:
                self._quarantine_lifetimes(error)
                return
            remaining = deadline - monotonic()
            if remaining <= 0:
                self._quarantine_lifetimes(
                    TimeoutError(
                        f"{fence_name} remained pending while abandoning completion"
                    )
                )
                return
            sleep(min(_ABANDON_QUERY_INTERVAL_SECONDS, remaining))
        self._terminal = True
        self._retire()

    def __del__(self) -> None:
        try:
            self._abandon()
        except BaseException as error:
            try:
                self._quarantine_lifetimes(error)
            except BaseException:
                pass


class CudaExecutor:
    def __init__(self, torch, device, stream, *, resources, quarantine=None) -> None:
        self.torch = torch
        self.device = device
        self.stream = stream
        self._resources = resources
        self._quarantine = (
            CUDA_COMPLETION_QUARANTINE if quarantine is None else quarantine
        )
        self._owner = object()
        self.inflight: CudaCompletion | None = None
        self.dispatch_count = 0

    def dispatch(self, plan, launch) -> CudaCompletion:
        if self.inflight is not None:
            raise RuntimeError("CUDA executor already has an in-flight batch")
        torch = self.torch
        completion = CudaCompletion(
            self._owner,
            plan,
            self._resources(),
            self.stream,
            self._quarantine,
            TensorStager(torch, self.device),
            torch.cuda.Event(),
            torch.cuda.Event(),
        )
        self.inflight = completion
        try:
            with torch.cuda.stream(self.stream), torch.inference_mode():
                completion._start_event.record(self.stream)
                completion._accepted = True
                self.dispatch_count += 1
                try:
                    output_device = launch(completion)
                    completion.retain(output_device)
                    completion._output_host = completion.staging.output(
                        "output", output_device
                    )
                except BaseException as error:
                    completion._dispatch_error = error
                try:
                    completion._done_event.record(self.stream)
                    completion._done_recorded = True
                except BaseException as error:
                    if completion._dispatch_error is None:
                        completion._dispatch_error = error
        except BaseException as error:
            if not completion._accepted:
                self.inflight = None
                raise
            if completion._dispatch_error is None:
                completion._dispatch_error = error
        return completion

    def poll(self, completion) -> list[int] | None:
        if not isinstance(completion, CudaCompletion):
            raise TypeError("completion was not created by CudaExecutor.dispatch")
        if completion._owner is not self._owner:
            raise ValueError("completion belongs to a different CUDA executor")
        if self.inflight is not completion or completion._retired:
            raise ValueError("completion is stale or is not the in-flight batch")
        if not completion._done_recorded:
            raise completion._dispatch_error
        if not completion._done_event.query():
            return None
        if completion._dispatch_error is not None:
            raise completion._dispatch_error

        output = completion._output_host.tolist()
        completion._terminal = True
        completion._retire()
        self.inflight = None
        return output
