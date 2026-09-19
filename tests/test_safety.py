"""Unit tests for the enforceable safety guards and public control exports."""

import pytest

from arm5.controls import EStop, SafetyLimits  # EStop must be re-exported
from arm5.controls.safety import XARM5_DOF, check_joint_command_length


def test_dof_is_five():
    assert XARM5_DOF == 5


def test_joint_command_length_ok():
    check_joint_command_length([0.0, 0.0, 0.0, 0.0, 0.0])  # 5-DOF, no raise


@pytest.mark.parametrize("bad", [[0.0] * 4, [0.0] * 6, []])
def test_joint_command_length_wrong_raises(bad):
    with pytest.raises(ValueError):
        check_joint_command_length(bad)


def test_safety_limits_fail_closed_when_unset():
    limits = SafetyLimits()  # no datasheet limits configured
    # correct length, but unset limits -> refuse (fail closed), not approve
    with pytest.raises(NotImplementedError):
        limits.check_joint_positions([0.0] * XARM5_DOF)
    # wrong length is caught first, as a ValueError
    with pytest.raises(ValueError):
        limits.check_joint_positions([0.0] * 3)


def test_estop_triggers_callback():
    fired = []
    stop = EStop(on_trigger=lambda: fired.append(True))
    assert stop.is_triggered is False
    stop.trigger()
    assert stop.is_triggered is True
    assert fired == [True]
