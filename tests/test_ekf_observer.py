"""Tests for the EKF force observer."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from openforce.adapters.sim import SimAdapter
from openforce.dynamics.mujoco_dynamics import MuJoCoDynamics
from openforce.estimation.ekf_observer import EKFObserver, EKFParams


def _make_ekf_setup() -> tuple[SimAdapter, MuJoCoDynamics, EKFObserver]:
    """Create EKF observer test setup."""
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints

    observer = EKFObserver(
        dynamics=dynamics,
        n_joints=n,
        dt=adapter.dt,
        params=EKFParams(
            Q_q=np.full(n, 1e-6),
            Q_qd=np.full(n, 1e-4),
            Q_tau=np.full(n, 1.0),
            R=np.full(n, 1e-6),
        ),
    )
    return adapter, dynamics, observer


class TestEKFObserver:
    def test_zero_external_force(self) -> None:
        """With no external forces, EKF estimate should stay near zero."""
        adapter, dynamics, observer = _make_ekf_setup()
        n = adapter.n_joints

        q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        adapter.reset(q0)
        observer.reset()

        for _ in range(500):
            state = adapter.get_state()
            g = dynamics.gravity_vector(state.q)
            output = observer.update(state)
            adapter.send_torque(g)

        assert np.linalg.norm(output.tau_ext) < 5.0, (
            f"EKF residual too large without external force: {np.linalg.norm(output.tau_ext):.4f}"
        )

    def test_detects_external_force(self) -> None:
        """EKF should detect an applied external force."""
        adapter, dynamics, observer = _make_ekf_setup()

        q0 = np.array([0.0, 0.8, -0.5, 0.0, 0.4, 0.0])
        adapter.reset(q0)
        observer.reset()

        # Settle with gravity compensation
        for _ in range(300):
            state = adapter.get_state()
            g = dynamics.gravity_vector(state.q)
            observer.update(state)
            adapter.send_torque(g)

        # Apply external force
        adapter.env.apply_external_force("ee", np.array([5.0, 0.0, -5.0]))

        max_residual = 0.0
        for _ in range(200):
            state = adapter.get_state()
            g = dynamics.gravity_vector(state.q)
            output = observer.update(state)
            adapter.send_torque(g)

            norm = float(np.linalg.norm(output.tau_ext))
            if norm > max_residual:
                max_residual = norm

        assert max_residual > 0.1, (
            f"EKF never detected the force (max residual = {max_residual:.4f})"
        )

    def test_reset_clears_state(self) -> None:
        """Reset should clear EKF internal state."""
        adapter, dynamics, observer = _make_ekf_setup()
        n = adapter.n_joints

        q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        adapter.reset(q0)

        for _ in range(10):
            state = adapter.get_state()
            observer.update(state)
            adapter.send_torque(np.zeros(n))

        observer.reset()
        state = adapter.get_state()
        output = observer.update(state)
        assert_allclose(output.tau_ext, np.zeros(n))

    def test_default_params(self) -> None:
        """EKF should work with default parameters."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        observer = EKFObserver(dynamics=dynamics, n_joints=n, dt=adapter.dt)
        adapter.reset()
        state = adapter.get_state()
        output = observer.update(state)
        assert output.tau_ext.shape == (n,)
