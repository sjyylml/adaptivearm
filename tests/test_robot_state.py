"""Tests for RobotState."""

import numpy as np
import pytest

from adaptivearm.core.robot_state import RobotState


class TestRobotState:
    def test_n_joints(self) -> None:
        state = RobotState(
            q=np.zeros(6),
            qd=np.zeros(6),
            tau_motor=np.zeros(6),
        )
        assert state.n_joints == 6

    def test_validate_ok(self) -> None:
        state = RobotState(
            q=np.zeros(4),
            qd=np.zeros(4),
            tau_motor=np.zeros(4),
            jacobian=np.zeros((6, 4)),
        )
        state.validate()  # should not raise

    def test_validate_qd_mismatch(self) -> None:
        state = RobotState(
            q=np.zeros(6),
            qd=np.zeros(4),  # wrong size
            tau_motor=np.zeros(6),
        )
        with pytest.raises(ValueError, match="qd shape"):
            state.validate()

    def test_validate_tau_mismatch(self) -> None:
        state = RobotState(
            q=np.zeros(6),
            qd=np.zeros(6),
            tau_motor=np.zeros(3),
        )
        with pytest.raises(ValueError, match="tau_motor shape"):
            state.validate()

    def test_validate_jacobian_mismatch(self) -> None:
        state = RobotState(
            q=np.zeros(6),
            qd=np.zeros(6),
            tau_motor=np.zeros(6),
            jacobian=np.zeros((3, 6)),  # should be (6, 6)
        )
        with pytest.raises(ValueError, match="jacobian shape"):
            state.validate()
