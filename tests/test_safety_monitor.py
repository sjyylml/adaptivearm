"""Tests for safety monitor."""

from __future__ import annotations

import numpy as np

from openforce.control.safety_monitor import SafetyLimits, SafetyMonitor, SafetyState
from openforce.core.types import ObserverOutput
from openforce.estimation.collision_detector import CollisionDetector


class TestSafetyMonitor:
    def test_torque_clipping(self) -> None:
        """Torques exceeding limits should be clipped."""
        n = 6
        limits = SafetyLimits(
            q_min=np.full(n, -3.0),
            q_max=np.full(n, 3.0),
            qd_max=np.full(n, 5.0),
            tau_max=np.full(n, 10.0),
        )
        monitor = SafetyMonitor(n_joints=n, limits=limits)

        tau_cmd = np.full(n, 50.0)  # Way over limit
        q = np.zeros(n)
        qd = np.zeros(n)

        result = monitor.filter(tau_cmd, q, qd)
        assert np.all(np.abs(result.tau_cmd) <= 10.0)

    def test_passthrough_within_limits(self) -> None:
        """Torques within limits should pass through unchanged."""
        n = 6
        limits = SafetyLimits(
            q_min=np.full(n, -3.0),
            q_max=np.full(n, 3.0),
            qd_max=np.full(n, 5.0),
            tau_max=np.full(n, 100.0),
        )
        monitor = SafetyMonitor(n_joints=n, limits=limits)

        tau_cmd = np.array([1.0, -2.0, 3.0, -1.0, 0.5, -0.5])
        q = np.zeros(n)
        qd = np.zeros(n)

        result = monitor.filter(tau_cmd, q, qd)
        np.testing.assert_allclose(result.tau_cmd, tau_cmd, atol=1e-10)

    def test_collision_stops_robot(self) -> None:
        """Collision detection should zero the torque output."""
        n = 6
        det = CollisionDetector(n_joints=n, thresholds=np.full(n, 5.0))
        monitor = SafetyMonitor(n_joints=n, collision_detector=det)

        tau_cmd = np.full(n, 10.0)
        q = np.zeros(n)
        qd = np.zeros(n)

        # Pass an observer output that triggers collision
        obs = ObserverOutput(tau_ext=np.full(n, 20.0), timestamp=0.1)
        result = monitor.filter(tau_cmd, q, qd, observer_output=obs)

        assert np.all(result.tau_cmd == 0.0)
        assert monitor.state == SafetyState.COLLISION_DETECTED

    def test_position_limit_repulsion(self) -> None:
        """Near position limits, repulsive torque should be applied."""
        n = 6
        limits = SafetyLimits(
            q_min=np.full(n, -2.0),
            q_max=np.full(n, 2.0),
            qd_max=np.full(n, 5.0),
            tau_max=np.full(n, 100.0),
            q_margin=0.2,
        )
        monitor = SafetyMonitor(n_joints=n, limits=limits)

        # Joint 0 near upper limit
        q = np.zeros(n)
        q[0] = 1.95  # Within margin of q_max=2.0
        qd = np.zeros(n)
        tau_cmd = np.zeros(n)

        result = monitor.filter(tau_cmd, q, qd)
        # Should have negative (repulsive) torque on joint 0
        assert result.tau_cmd[0] < 0, f"Expected repulsive torque, got {result.tau_cmd[0]}"

    def test_reset(self) -> None:
        """Reset should restore normal state."""
        n = 6
        det = CollisionDetector(n_joints=n, thresholds=np.full(n, 5.0))
        monitor = SafetyMonitor(n_joints=n, collision_detector=det)

        # Trigger collision
        obs = ObserverOutput(tau_ext=np.full(n, 20.0), timestamp=0.1)
        monitor.filter(np.zeros(n), np.zeros(n), np.zeros(n), obs)
        assert monitor.state == SafetyState.COLLISION_DETECTED

        monitor.reset()
        assert monitor.state == SafetyState.NORMAL
