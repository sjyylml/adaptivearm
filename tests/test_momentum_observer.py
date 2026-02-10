"""Tests for the Generalized Momentum Observer (GMO)."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from openforce.adapters.sim import SimAdapter
from openforce.dynamics.mujoco_dynamics import MuJoCoDynamics
from openforce.estimation.momentum_observer import MomentumObserver
from openforce.sim.virtual_sensor import VirtualForceSensor


def _make_observer_setup(
    gains: float = 50.0,
) -> tuple[SimAdapter, MuJoCoDynamics, MomentumObserver, VirtualForceSensor]:
    """Create a complete observer test setup with shared adapter."""
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints

    observer = MomentumObserver(
        dynamics=dynamics,
        n_joints=n,
        dt=adapter.dt,
        gains=np.full(n, gains),
    )
    sensor = VirtualForceSensor(adapter.env)
    return adapter, dynamics, observer, sensor


class TestMomentumObserver:
    def test_zero_external_force(self) -> None:
        """With no external forces, observer residual should stay near zero."""
        adapter, dynamics, observer, _ = _make_observer_setup()
        n = adapter.n_joints

        q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        adapter.reset(q0)
        observer.reset()

        for _ in range(500):
            state = adapter.get_state()
            g = dynamics.gravity_vector(state.q)
            output = observer.update(state)
            adapter.send_torque(g)

        assert_allclose(output.tau_ext, np.zeros(n), atol=1.5)

    def test_detects_external_force(self) -> None:
        """Observer should detect an applied external force during transient."""
        adapter, dynamics, observer, sensor = _make_observer_setup(gains=30.0)

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

        # Sample the observer during the transient response (not at steady state)
        # The momentum observer detects force *changes* — it picks up the
        # transient where the external force creates unmodeled acceleration.
        max_residual = 0.0
        for i in range(200):
            state = adapter.get_state()
            g = dynamics.gravity_vector(state.q)
            output = observer.update(state)
            adapter.send_torque(g)

            norm = float(np.linalg.norm(output.tau_ext))
            if norm > max_residual:
                max_residual = norm

        # During the transient, the observer should detect the external force
        assert max_residual > 0.1, (
            f"Observer never detected the force (max residual = {max_residual:.4f})"
        )

    def test_reset_clears_state(self) -> None:
        """Reset should clear observer internal state."""
        adapter, dynamics, observer, _ = _make_observer_setup()
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
