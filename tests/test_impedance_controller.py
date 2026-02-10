"""Tests for impedance controller."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.control.impedance import ImpedanceController, ImpedanceParams
from adaptivearm.dynamics.mujoco_dynamics import MuJoCoDynamics


class TestImpedanceController:
    def test_holds_position(self) -> None:
        """Impedance controller should hold the arm at the desired position."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        q_desired = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        ctrl = ImpedanceController(
            dynamics=dynamics,
            n_joints=n,
            params=ImpedanceParams(
                stiffness=np.full(n, 1000.0),
                damping=np.full(n, 100.0),
                q_desired=q_desired,
                use_coriolis_comp=True,
            ),
        )

        adapter.reset(q_desired)

        for _ in range(2000):
            state = adapter.get_state()
            output = ctrl.compute(state)
            adapter.send_torque(output.tau_cmd)

        state = adapter.get_state()
        error = np.linalg.norm(state.q - q_desired)
        assert error < 0.6, f"Position error too large: {error}"

    def test_converges_to_target(self) -> None:
        """Starting away from target, arm should converge."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        q_start = np.zeros(n)
        q_desired = np.array([0.0, 0.3, -0.2, 0.0, 0.1, 0.0])

        ctrl = ImpedanceController(
            dynamics=dynamics,
            n_joints=n,
            params=ImpedanceParams(
                stiffness=np.full(n, 500.0),
                damping=np.full(n, 50.0),
                q_desired=q_desired,
                use_coriolis_comp=True,
            ),
        )

        adapter.reset(q_start)

        for _ in range(3000):
            state = adapter.get_state()
            output = ctrl.compute(state)
            adapter.send_torque(output.tau_cmd)

        state = adapter.get_state()
        error = np.linalg.norm(state.q - q_desired)
        assert error < 0.4, f"Did not converge: error = {error}"

    def test_set_target(self) -> None:
        """set_target should update the desired position."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        ctrl = ImpedanceController(dynamics=dynamics, n_joints=n)
        new_target = np.ones(n) * 0.5
        ctrl.set_target(new_target)
        assert_allclose(ctrl.params.q_desired, new_target)
