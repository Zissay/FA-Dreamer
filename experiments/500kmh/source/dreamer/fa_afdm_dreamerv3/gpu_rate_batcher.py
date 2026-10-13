"""Thread-synchronised GPU batching for fixed-channel direct-position queries."""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass
from multiprocessing.connection import Client

import numpy as np

from .channel import FAAFDMChannel


@dataclass
class _Request:
    u: np.ndarray
    q: np.ndarray
    rate: float | None = None
    error: BaseException | None = None


class GPUAFDMRateBatcher:
    """Collect one query from each threaded environment, then evaluate together."""

    def __init__(
        self,
        channel: FAAFDMChannel,
        batch_size: int,
        device: str,
        backend: str = "local",
        socket_path: str = "",
    ) -> None:
        if batch_size < 2:
            raise ValueError("GPUAFDMRateBatcher needs at least two environments")
        self.channel = channel
        self.batch_size = int(batch_size)
        self.device = device
        self.backend = backend
        self.socket_path = socket_path
        self._condition = threading.Condition()
        self._pending: list[_Request] = []
        # Matched Dreamer environments have stable indices.  Preserve those
        # indices inside a CUDA batch so thread-arrival timing cannot change
        # the order of otherwise identical rate queries.
        self._pending_slots: list[_Request | None] = [None] * self.batch_size
        self._rpc_lock = threading.Lock()
        self._rpc_connection = None

    def _rate_batch(self, u: np.ndarray, q: np.ndarray) -> np.ndarray:
        if self.backend == "local":
            return self.channel.rate_batch_gpu(u, q, device=self.device)
        if self.backend != "rpc":
            raise ValueError(f"Unsupported GPU rate backend: {self.backend}")
        if not self.socket_path:
            raise ValueError("gpu_rate_backend=rpc needs a Unix socket path")
        with self._rpc_lock:
            try:
                if self._rpc_connection is None:
                    self._rpc_connection = Client(self.socket_path, family="AF_UNIX")
                self._rpc_connection.send(
                    {
                        "op": "rate_batch",
                        "config": dataclasses.asdict(self.channel.cfg),
                        "u": u,
                        "q": q,
                    }
                )
                reply = self._rpc_connection.recv()
            except BaseException:
                if self._rpc_connection is not None:
                    self._rpc_connection.close()
                    self._rpc_connection = None
                raise
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"GPU rate server failed: {reply['error']}")
        return np.asarray(reply, dtype=np.float64)

    def rate(self, u: np.ndarray, q: np.ndarray, slot: int | None = None) -> float:
        request = _Request(np.asarray(u, dtype=np.float64).copy(), np.asarray(q, dtype=np.float64).copy())
        with self._condition:
            if slot is None:
                self._pending.append(request)
                ready = len(self._pending) == self.batch_size
                requests = self._pending if ready else None
                if ready:
                    self._pending = []
            else:
                if not 0 <= int(slot) < self.batch_size:
                    raise ValueError(f"GPU rate slot {slot} is outside [0, {self.batch_size})")
                if self._pending_slots[int(slot)] is not None:
                    raise RuntimeError(f"GPU rate slot {slot} was submitted twice in one batch")
                self._pending_slots[int(slot)] = request
                ready = all(item is not None for item in self._pending_slots)
                requests = list(self._pending_slots) if ready else None
                if ready:
                    self._pending_slots = [None] * self.batch_size
            if ready:
                assert requests is not None
                try:
                    rates = self._rate_batch(
                        np.stack([item.u for item in requests]),
                        np.stack([item.q for item in requests]),
                    )
                    for item, value in zip(requests, rates, strict=True):
                        item.rate = float(value)
                except BaseException as exc:  # Propagate a GPU failure to every worker.
                    for item in requests:
                        item.error = exc
                self._condition.notify_all()
            else:
                self._condition.wait_for(lambda: request.rate is not None or request.error is not None)
            if request.error is not None:
                raise request.error
            assert request.rate is not None
            return request.rate


_LOCK = threading.Lock()
_BATCHERS: dict[tuple[tuple[tuple[str, object], ...], int, str, str, str], GPUAFDMRateBatcher] = {}


def shared_gpu_rate_batcher(
    channel: FAAFDMChannel,
    batch_size: int,
    device: str = "cuda",
    backend: str = "local",
    socket_path: str = "",
) -> GPUAFDMRateBatcher:
    """Return the one batcher shared by identical fixed-channel environments."""
    config_key = tuple(sorted(dataclasses.asdict(channel.cfg).items()))
    key = (config_key, int(batch_size), str(device), str(backend), str(socket_path))
    with _LOCK:
        batcher = _BATCHERS.get(key)
        if batcher is None:
            batcher = GPUAFDMRateBatcher(
                channel, int(batch_size), str(device), str(backend), str(socket_path))
            _BATCHERS[key] = batcher
        return batcher
