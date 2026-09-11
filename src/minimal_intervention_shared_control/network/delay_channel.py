from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class Packet(Generic[T]):
    payload: T
    sent_at: float
    source_stamp: float
    deliver_at: float
    sequence: int


@dataclass
class DelayModel:
    mean: float = 0.0
    jitter_std: float = 0.0
    rho: float = 0.0
    max_delay: float | None = None
    loss: float = 0.0

    def __post_init__(self) -> None:
        if self.mean < 0 or self.jitter_std < 0 or not 0 <= self.loss < 1:
            raise ValueError("invalid delay model parameters")
        if self.max_delay is not None and self.max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        self._last_delay = self.mean

    def sample(self, rng: np.random.Generator) -> float | None:
        if rng.random() < self.loss:
            return None
        innovation = float(rng.normal(0.0, self.jitter_std))
        delay = self.mean + self.rho * (self._last_delay - self.mean) + innovation
        delay = max(0.0, delay)
        if self.max_delay is not None:
            delay = min(delay, self.max_delay)
        self._last_delay = delay
        return delay


class DelayChannel(Generic[T]):
    """Timestamped FIFO channel with correlated delay and packet loss."""

    def __init__(self, model: DelayModel | None = None, seed: int | None = None):
        self.model = model or DelayModel()
        self.rng = np.random.default_rng(seed)
        self._queue: list[tuple[float, int, Packet[T]]] = []
        self._sequence = 0
        self._last_delivery = 0.0
        self.sent = 0
        self.dropped = 0

    def send(
        self, payload: T, now: float, source_stamp: float | None = None
    ) -> Packet[T] | None:
        if now < 0:
            raise ValueError("time must be non-negative")
        delay = self.model.sample(self.rng)
        self.sent += 1
        if delay is None:
            self.dropped += 1
            return None
        stamp = now if source_stamp is None else float(source_stamp)
        # A FIFO transport cannot deliver a newer packet before an older one.
        deliver_at = max(now + delay, self._last_delivery)
        packet = Packet(payload, float(now), stamp, deliver_at, self._sequence)
        self._sequence += 1
        self._last_delivery = deliver_at
        heapq.heappush(self._queue, (deliver_at, packet.sequence, packet))
        return packet

    def receive(self, now: float) -> list[Packet[T]]:
        ready: list[Packet[T]] = []
        while self._queue and self._queue[0][0] <= now + 1e-12:
            ready.append(heapq.heappop(self._queue)[2])
        return ready

    def pending(self) -> int:
        return len(self._queue)


class BidirectionalNetwork:
    """Separate downlink (state to human) and uplink (command to vehicle) channels."""

    def __init__(self, downlink: DelayModel, uplink: DelayModel, seed: int = 0):
        self.downlink = DelayChannel(downlink, seed=seed)
        self.uplink = DelayChannel(uplink, seed=seed + 1)

    def send_state(self, state: object, now: float) -> Packet[object] | None:
        return self.downlink.send(state, now, source_stamp=now)

    def receive_states(self, now: float) -> list[Packet[object]]:
        return self.downlink.receive(now)

    def send_command(
        self, command: object, now: float, source_stamp: float
    ) -> Packet[object] | None:
        return self.uplink.send(command, now, source_stamp=source_stamp)

    def receive_commands(self, now: float) -> list[Packet[object]]:
        return self.uplink.receive(now)
