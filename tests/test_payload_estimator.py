"""Tests for payload estimator."""

from __future__ import annotations

import numpy as np

from adaptivearm.identification.payload_estimator import PayloadEstimator


class TestPayloadEstimator:
    def test_estimates_mass(self) -> None:
        """Estimator should converge to correct mass from synthetic data."""
        n = 6
        true_mass = 2.0
        true_cog = np.array([0.0, 0.0, 0.05])
        g = np.array([0.0, 0.0, -9.81])

        estimator = PayloadEstimator(n_joints=n, forgetting_factor=0.99)
        rng = np.random.default_rng(42)

        for _ in range(200):
            # Random Jacobian (as if arm is in different configs)
            J = rng.standard_normal((6, n)) * 0.3

            # True payload torque: J^T · [m*g; m*(cog × g)]
            force = true_mass * g
            torque_cart = true_mass * np.cross(true_cog, g)
            wrench = np.concatenate([force, torque_cart])
            tau_ext = J.T @ wrench

            # Add small noise
            tau_ext += rng.normal(0, 0.05, size=n)

            est = estimator.update(tau_ext, J)

        assert abs(est.mass - true_mass) < 0.5, f"Mass estimate {est.mass} != {true_mass}"
        assert est.n_samples == 200

    def test_reset(self) -> None:
        """Reset should zero the estimate."""
        est = PayloadEstimator(n_joints=6)
        J = np.eye(6)
        est.update(np.ones(6), J)
        est.reset()
        result = est.get_estimate()
        assert result.mass == 0.0
        assert result.n_samples == 0
