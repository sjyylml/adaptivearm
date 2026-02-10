"""Tests for admittance controller."""

from __future__ import annotations

import numpy as np

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.control.admittance import AdmittanceController
from adaptivearm.core.types import ObserverOutput
from adaptivearm.dynamics.mujoco_dynamics import MuJoCoDynamics


class TestAdmittanceController:
    def test_zero_force_holds_position(self) -> None:
        """With zero external force, arm stays at nominal position."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        from adaptivearm.control.admittance import AdmittanceParams

        ctrl = AdmittanceController(
            dynamics=dynamics,
            n_joints=n,
            dt=adapter.dt,
            params=AdmittanceParams(
                mass=np.full(n, 1.0),
                damping=np.full(n, 40.0),
                stiffness=np.full(n, 10.0),
                inner_kp=np.full(n, 500.0),
                inner_kd=np.full(n, 50.0),
            ),
        )
        ctrl.set_nominal(q0)

        adapter.reset(q0)
        ctrl.reset()

        for _ in range(1000):
            state = adapter.get_state()
            obs = ObserverOutput(tau_ext=np.zeros(n), timestamp=state.timestamp)
            output = ctrl.compute(state, obs)
            adapter.send_torque(output.tau_cmd)

        state = adapter.get_state()
        error = np.linalg.norm(state.q - q0)
        assert error < 0.6, f"Drifted from nominal: error = {error}"

    def test_external_force_causes_displacement(self) -> None:
        """A constant external torque should cause an admittance displacement."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        ctrl = AdmittanceController(dynamics=dynamics, n_joints=n, dt=adapter.dt)
        ctrl.set_nominal(q0)

        adapter.reset(q0)
        ctrl.reset()

        # Simulate constant external torque on joint 1
        fake_tau_ext = np.zeros(n)
        fake_tau_ext[1] = 5.0

        for _ in range(500):
            state = adapter.get_state()
            obs = ObserverOutput(tau_ext=fake_tau_ext, timestamp=state.timestamp)
            output = ctrl.compute(state, obs)
            adapter.send_torque(output.tau_cmd)

        # The admittance filter should have displaced delta_q
        assert np.linalg.norm(ctrl.delta_q) > 0.01, "Admittance filter did not respond to force"

    def test_reset_clears_displacement(self) -> None:
        """Reset should zero the admittance displacement."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        ctrl = AdmittanceController(dynamics=dynamics, n_joints=n, dt=adapter.dt)
        ctrl.set_nominal(np.zeros(n))

        # Fake some displacement
        state = adapter.reset()
        obs = ObserverOutput(tau_ext=np.ones(n) * 10.0, timestamp=0.0)
        for _ in range(100):
            ctrl.compute(state, obs)

        assert np.linalg.norm(ctrl.delta_q) > 0
        ctrl.reset()
        assert np.linalg.norm(ctrl.delta_q) == 0.0
