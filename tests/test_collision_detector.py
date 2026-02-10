"""Tests for collision detector."""

from __future__ import annotations

import numpy as np

from openforce.core.types import ObserverOutput
from openforce.estimation.collision_detector import CollisionDetector, CollisionReaction


class TestCollisionDetector:
    def test_no_collision_below_threshold(self) -> None:
        """No detection when residual is below threshold."""
        det = CollisionDetector(n_joints=6, thresholds=np.full(6, 5.0))
        obs = ObserverOutput(tau_ext=np.full(6, 2.0), timestamp=0.1)
        event = det.check(obs)
        assert not event.detected
        assert event.severity < 1.0

    def test_collision_above_threshold(self) -> None:
        """Detection when any joint exceeds threshold."""
        det = CollisionDetector(n_joints=6, thresholds=np.full(6, 5.0))
        tau = np.zeros(6)
        tau[2] = 10.0  # Joint 2 exceeds 5.0 threshold
        obs = ObserverOutput(tau_ext=tau, timestamp=0.1)
        event = det.check(obs)
        assert event.detected
        assert event.joint_mask[2]
        assert event.severity >= 2.0

    def test_holdoff_prevents_retrigger(self) -> None:
        """Holdoff time prevents immediate retriggering."""
        det = CollisionDetector(n_joints=6, thresholds=np.full(6, 5.0), holdoff_time=0.5)
        tau = np.full(6, 10.0)

        # First detection
        obs1 = ObserverOutput(tau_ext=tau, timestamp=0.1)
        event1 = det.check(obs1)
        assert event1.detected

        # Within holdoff — should not retrigger
        obs2 = ObserverOutput(tau_ext=tau, timestamp=0.3)
        event2 = det.check(obs2)
        assert not event2.detected

        # After holdoff — should retrigger
        obs3 = ObserverOutput(tau_ext=tau, timestamp=0.7)
        event3 = det.check(obs3)
        assert event3.detected

    def test_collision_clears_when_force_drops(self) -> None:
        """in_collision flag clears when residual drops well below threshold."""
        det = CollisionDetector(n_joints=6, thresholds=np.full(6, 5.0))

        # Trigger collision
        obs1 = ObserverOutput(tau_ext=np.full(6, 10.0), timestamp=0.1)
        det.check(obs1)
        assert det.in_collision

        # Drop below 50% of threshold
        obs2 = ObserverOutput(tau_ext=np.full(6, 1.0), timestamp=0.5)
        det.check(obs2)
        assert not det.in_collision

    def test_reset(self) -> None:
        """Reset clears collision state."""
        det = CollisionDetector(n_joints=6, thresholds=np.full(6, 5.0))
        obs = ObserverOutput(tau_ext=np.full(6, 10.0), timestamp=0.1)
        det.check(obs)
        assert det.in_collision
        det.reset()
        assert not det.in_collision
