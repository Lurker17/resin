"""Route HTTP ingress while every attention replica joins coordinated steps."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from resin.backend.distributed import DistributedExecutionError

if TYPE_CHECKING:
    from resin.backend.distributed import DistributedEngine
    from resin.recipe import Recipe


class DistributedHttpCoordinator:
    """One native HTTP session per driver; only the first driver owns clients.

    Construct collectively on replica drivers. The first driver calls round()
    from its native HTTP actor, while the remaining drivers call serve().
    Request IDs and events cross the control group; leases, KV, recipes, media,
    and parsers remain local to the selected replica.
    """

    def __init__(
        self,
        engine: DistributedEngine,
        recipe: Recipe,
        *,
        stream_interval: int = 1,
    ):
        self.engine = engine
        self.group = engine.worker._driver_group
        self.distributed = engine.worker.groups.distributed
        self.rank = self.group.local_rank
        self.routed_dp_size = self.group.size
        self.is_root = self.rank == 0
        self.closed = False
        self.running = False
        self._shutdown = False
        self._next_rank = 0
        self._owners: dict[str, int] = {}
        self.session = None
        error = None
        try:
            from resin._resin import _HttpSession

            self.session = _HttpSession(
                engine.local, recipe, stream_interval=stream_interval
            )
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("initialize", error)
        if failure:
            self._close(failure)

    @property
    def is_frontend(self):
        return self.is_root

    @property
    def size(self):
        return self.routed_dp_size

    def shutdown(self):
        """Request collective shutdown at the next round boundary."""
        self._shutdown = True

    def close(self):
        """Root startup/shutdown cleanup; peers must be in their serve loop."""
        if self.closed:
            return
        if not self.is_root:
            self.shutdown()
            return
        self.round("[]", [], True)

    def _vote(self, stage, error):
        errors = self.engine.worker._drivers(f"http {stage}", error)
        return next(
            (
                f"HTTP {stage} failed on rank {rank}: {error}"
                for rank, error in zip(self.group.ranks, errors, strict=True)
                if error
            ),
            None,
        )

    def _ingress(self, admissions_json, canceled, shutdown):
        """Validate before broadcast, carrying root errors to waiting peers."""
        try:
            requests = json.loads(admissions_json)
            if not isinstance(requests, list):
                raise ValueError("HTTP admissions must be a JSON list")
            for request in requests:
                request_id = request["id"]
                if not isinstance(request_id, str) or request_id in self._owners:
                    raise ValueError("HTTP request IDs must be unique strings")
                rank = request.get("routed_dp_rank")
                if rank is None:
                    rank = self._next_rank
                    self._next_rank = (rank + 1) % self.routed_dp_size
                if type(rank) is not int or not 0 <= rank < self.routed_dp_size:
                    raise ValueError("routed_dp_rank is outside the attention-DP mesh")
                request["routed_dp_rank"] = rank
                self._owners[request_id] = rank
            if not all(isinstance(request_id, str) for request_id in canceled):
                raise ValueError("HTTP cancellations must contain request IDs")
            return {"requests": requests, "canceled": canceled, "shutdown": shutdown}
        except BaseException as exc:
            return {"error": str(exc), "shutdown": True}

    def round(self, admissions_json: str, canceled: list[str], shutdown: bool) -> str:
        """Root actor boundary: dispatch ingress and collect one step's events."""
        if not self.is_root:
            raise ValueError("only the first replica driver owns HTTP ingress")
        if self.closed:
            if shutdown:
                return "[]"
            raise RuntimeError("distributed HTTP coordinator is closed")
        self.running = True
        command = self._ingress(admissions_json, canceled, shutdown)
        return self._round(command)

    def serve(self):
        """Block on a non-root driver until the frontend closes all replicas."""
        if self.is_root:
            raise ValueError("the first replica driver serves the HTTP frontend")
        self.running = True
        while not self.closed:
            self._round(None)

    def _round(self, command):
        messages = [command]
        if self.group.size > 1:
            self.distributed.broadcast_object_list(
                messages, src=self.group.source, group=self.group.handle
            )
        command = messages[0]
        requested = self.engine.worker._drivers(
            "http shutdown requested", self._shutdown
        )
        if any(requested):
            command["shutdown"] = True
        if command["shutdown"]:
            self._close(command.get("error"))
            return "[]"
        error = None
        try:
            assert self.session is not None
            local = [
                request
                for request in command["requests"]
                if request["routed_dp_rank"] == self.rank
            ]
            self.session.admit(json.dumps(local))
            self.session.cancel(command["canceled"])
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("ingress", error)
        if failure:
            self._close(failure)

        events = []
        error = None
        try:
            if self.engine.has_work():
                events = self.engine.step()
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("step", error)
        if failure:
            self._close(failure)

        encoded = "[]"
        error = None
        try:
            assert self.session is not None
            encoded = self.session.process(events)
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("events", error)
        if failure:
            self._close(failure)
        gathered = self.engine.worker._drivers("http output", encoded)
        if not self.is_root:
            return "[]"
        combined = [event for payload in gathered for event in json.loads(payload)]
        for request_id, event in combined:
            if event["type"] in ("finished", "error"):
                self._owners.pop(request_id, None)
        for request_id in command["canceled"]:
            self._owners.pop(request_id, None)
        return json.dumps(combined)

    def _close(self, failure=None):
        """Close collectively even when native admission or parsing has failed."""
        if self.closed:
            if failure:
                raise DistributedExecutionError(failure)
            return
        error = None
        try:
            if self.session is not None:
                self.session.close()
        except BaseException as exc:
            error = str(exc)
        failure = failure or self._vote("shutdown", error)
        try:
            self.engine.close()
        except BaseException as exc:
            failure = failure or str(exc)
        finally:
            self.closed = True
            self.running = False
            self._owners.clear()
        if failure:
            raise DistributedExecutionError(failure)
