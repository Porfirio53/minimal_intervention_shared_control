from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CommandEnvelope(Generic[T]):
    command: T
    sensor_stamp: float
    generated_stamp: float
    received_stamp: float

    @property
    def age(self) -> float:
        return max(0.0, self.received_stamp - self.sensor_stamp)

    @property
    def uplink_delay(self) -> float:
        """Return r_m - g_m from paper equation (2)."""
        return max(0.0, self.received_stamp - self.generated_stamp)

    @property
    def generation_interval(self) -> float:
        """Return g_m - s_m, including downlink and human response time."""
        return max(0.0, self.generated_stamp - self.sensor_stamp)

    def information_age(self, now: float) -> float:
        """Return tau_H=t_k-s_m at an arbitrary execution time."""
        if now < self.received_stamp - 1e-9:
            raise ValueError("information age is undefined before command reception")
        return max(0.0, now - self.sensor_stamp)


class TimestampBuffer(Generic[T]):
    """Keeps the newest causally valid command and exposes information age."""

    def __init__(self) -> None:
        self._latest: CommandEnvelope[T] | None = None
        self.rejected = 0

    def push(self, envelope: CommandEnvelope[T]) -> bool:
        if envelope.sensor_stamp > envelope.generated_stamp:
            self.rejected += 1
            return False
        if envelope.generated_stamp > envelope.received_stamp + 1e-9:
            self.rejected += 1
            return False
        if (
            self._latest is not None
            and envelope.generated_stamp < self._latest.generated_stamp
        ):
            self.rejected += 1
            return False
        self._latest = envelope
        return True

    def latest(self) -> CommandEnvelope[T] | None:
        return self._latest

    def get(self, now: float, default: T | None = None) -> tuple[T | None, float]:
        if self._latest is None:
            return default, float("inf")
        return self._latest.command, self._latest.information_age(now)
