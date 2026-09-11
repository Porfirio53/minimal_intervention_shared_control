import numpy as np
import pytest

from minimal_intervention_shared_control.network.delay_channel import (
    DelayChannel,
    DelayModel,
)
from minimal_intervention_shared_control.network.timestamp_buffer import (
    CommandEnvelope,
    TimestampBuffer,
)


def test_fifo_delivery_and_source_timestamp() -> None:
    channel = DelayChannel[str](DelayModel(mean=0.2), seed=3)
    packet = channel.send("command", now=1.0, source_stamp=0.4)
    assert packet is not None
    assert channel.receive(1.19) == []
    delivered = channel.receive(1.2)
    assert delivered[0].payload == "command"
    assert delivered[0].source_stamp == 0.4


def test_timestamp_buffer_rejects_causality_violation_and_stale_command() -> None:
    buffer = TimestampBuffer[np.ndarray]()
    assert not buffer.push(CommandEnvelope(np.zeros(2), 2.0, 1.0, 3.0))
    assert buffer.push(CommandEnvelope(np.ones(2), 1.0, 1.5, 2.0))
    assert not buffer.push(CommandEnvelope(np.zeros(2), 0.5, 1.4, 2.1))
    command, age = buffer.get(2.5)
    np.testing.assert_allclose(command, np.ones(2))
    assert age == 1.5


def test_human_information_age_decomposes_as_paper_equation_2() -> None:
    envelope = CommandEnvelope("command", 1.0, 1.35, 1.55)
    now = 1.8
    holding_age = now - envelope.received_stamp
    assert envelope.generation_interval == pytest.approx(0.35)
    assert envelope.uplink_delay == pytest.approx(0.2)
    assert envelope.information_age(now) == pytest.approx(0.8)
    assert envelope.information_age(now) == pytest.approx(
        holding_age + envelope.uplink_delay + envelope.generation_interval
    )


def test_loss_is_counted() -> None:
    channel = DelayChannel[str](DelayModel(loss=0.999999), seed=0)
    assert channel.send("lost", 0.0) is None
    assert channel.sent == channel.dropped == 1
