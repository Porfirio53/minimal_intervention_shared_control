import inspect

import numpy as np
import pytest

from minimal_intervention_shared_control.network.timestamp_buffer import CommandEnvelope
from minimal_intervention_shared_control.runtime import (
    RuntimeConfig,
    SharedControlRuntime,
    StepInput,
)
from minimal_intervention_shared_control.safety.robust_cbf_qp import RobustCBFFilter
from minimal_intervention_shared_control.types import Control


def test_runtime_keeps_nominal_control_when_execution_constraints_are_satisfied() -> (
    None
):
    runtime = SharedControlRuntime(RuntimeConfig(horizon=3, method="ours"))
    value = StepInput(
        now=0.0,
        state=np.zeros(3),
        human_commands=(),
        default_human=Control(0.2, 0.0),
        autonomous_controls=np.tile([0.6, 0.0], (3, 1)),
        obstacles=(),
        safety_observations=(),
    )
    output = runtime.step(value)
    assert output.allocation.minimum_budget == 0.0
    np.testing.assert_allclose(output.alpha, 0.0, atol=2e-6)
    np.testing.assert_allclose(
        output.filtered_control.as_array(), output.nominal_control.as_array()
    )
    assert not output.filter_result.infeasible


def test_runtime_safety_contract_has_no_human_intent_covariance() -> None:
    names = set(inspect.signature(RobustCBFFilter.filter).parameters)
    assert not {"sigma_h", "human_covariance", "human_intent"} & names


def test_runtime_rejects_a_backwards_platform_clock() -> None:
    runtime = SharedControlRuntime(RuntimeConfig(horizon=2, method="human_filter"))

    def value(now: float) -> StepInput:
        return StepInput(
            now=now,
            state=np.zeros(3),
            human_commands=(),
            default_human=Control(0.2, 0.0),
            autonomous_controls=np.tile([0.6, 0.0], (2, 1)),
            obstacles=(),
            safety_observations=(),
        )

    runtime.step(value(1.0))
    with pytest.raises(ValueError, match="monotonic"):
        runtime.step(value(0.9))


def test_delayed_callback_does_not_backfill_a_new_command_before_its_receipt() -> None:
    runtime = SharedControlRuntime(RuntimeConfig(horizon=2, method="human_filter"))
    default = Control(0.2, 0.0)
    common = {
        "state": np.zeros(3),
        "default_human": default,
        "autonomous_controls": np.tile([0.6, 0.0], (2, 1)),
        "obstacles": (),
        "safety_observations": (),
    }
    runtime.step(StepInput(now=0.0, human_commands=(), **common))
    new = Control(0.8, 0.1)
    runtime.step(
        StepInput(
            now=0.25,
            human_commands=(CommandEnvelope(new, 0.05, 0.1, 0.15),),
            **common,
        )
    )
    np.testing.assert_allclose(
        runtime.human_history,
        [default.as_array(), default.as_array(), new.as_array()],
    )


def test_first_runtime_step_accepts_an_already_received_command() -> None:
    runtime = SharedControlRuntime(RuntimeConfig(horizon=2, method="human_filter"))
    command = Control(0.8, 0.1)
    output = runtime.step(
        StepInput(
            now=1.0,
            state=np.zeros(3),
            human_commands=(CommandEnvelope(command, 0.4, 0.6, 0.8),),
            default_human=Control(0.2, 0.0),
            autonomous_controls=np.tile([0.6, 0.0], (2, 1)),
            obstacles=(),
            safety_observations=(),
        )
    )
    assert output.human_control == command
    assert output.human_information_age == pytest.approx(0.6)


def test_upper_runtime_updates_follow_the_shared_control_period() -> None:
    runtime = SharedControlRuntime(
        RuntimeConfig(horizon=2, method="human_filter", shared_control_period=0.05)
    )

    def run(now: float):
        return runtime.step(
            StepInput(
                now=now,
                state=np.zeros(3),
                human_commands=(),
                default_human=Control(0.2, 0.0),
                autonomous_controls=np.tile([0.6, 0.0], (2, 1)),
                obstacles=(),
                safety_observations=(),
            )
        )

    assert run(0.0).upper_updated
    assert not run(0.02).upper_updated
    assert run(0.05).upper_updated
