"""Tests for the adaptive impedance controller."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.control.adaptive_impedance import (
    AdaptiveImpedanceController,
    AdaptiveImpedanceParams,
)
from adaptivearm.core.types import ObserverOutput
from adaptivearm.dynamics.mujoco_dynamics import MuJoCoDynamics


class TestAdaptiveImpedanceController:
    def test_holds_position(self) -> None:
        """With no external force, controller should hold position."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        q_desired = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        ctrl = AdaptiveImpedanceController(
            dynamics=dynamics,
            n_joints=n,
            dt=adapter.dt,
            params=AdaptiveImpedanceParams(
                stiffness_init=np.full(n, 500.0),
                stiffness_min=np.full(n, 50.0),
                stiffness_max=np.full(n, 2000.0),
                damping_init=np.full(n, 50.0),
                damping_min=np.full(n, 5.0),
                damping_max=np.full(n, 200.0),
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

    def test_stiffness_softens_with_force(self) -> None:
        """Stiffness should decrease when external force is applied."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        init_stiffness = np.full(n, 500.0)
        ctrl = AdaptiveImpedanceController(
            dynamics=dynamics,
            n_joints=n,
            dt=adapter.dt,
            params=AdaptiveImpedanceParams(
                stiffness_init=init_stiffness,
                stiffness_min=np.full(n, 50.0),
                stiffness_max=np.full(n, 2000.0),
                damping_init=np.full(n, 50.0),
                damping_min=np.full(n, 5.0),
                damping_max=np.full(n, 200.0),
                alpha_force=10.0,
                alpha_error=0.0,  # disable error stiffening for this test
                force_threshold=1.0,
            ),
        )

        adapter.reset()

        # Simulate with large external force observation
        for _ in range(100):
            state = adapter.get_state()
            obs = ObserverOutput(tau_ext=np.full(n, 10.0), timestamp=state.timestamp)
            output = ctrl.compute(state, obs)
            adapter.send_torque(output.tau_cmd)

        # Stiffness should have decreased
        assert np.all(ctrl.stiffness < init_stiffness), (
            f"Stiffness did not soften: {ctrl.stiffness}"
        )

    def test_stiffness_stiffens_with_error(self) -> None:
        """Stiffness should increase when position error grows."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        init_stiffness = np.full(n, 200.0)
        ctrl = AdaptiveImpedanceController(
            dynamics=dynamics,
            n_joints=n,
            dt=adapter.dt,
            params=AdaptiveImpedanceParams(
                stiffness_init=init_stiffness,
                stiffness_min=np.full(n, 50.0),
                stiffness_max=np.full(n, 2000.0),
                damping_init=np.full(n, 50.0),
                damping_min=np.full(n, 5.0),
                damping_max=np.full(n, 200.0),
                alpha_force=0.0,  # disable force softening
                alpha_error=5.0,
                q_desired=np.ones(n) * 0.5,  # far from zero
            ),
        )

        adapter.reset(np.zeros(n))

        # Simulate with zero external force (large position error)
        for _ in range(100):
            state = adapter.get_state()
            obs = ObserverOutput(tau_ext=np.zeros(n), timestamp=state.timestamp)
            output = ctrl.compute(state, obs)
            adapter.send_torque(output.tau_cmd)

        assert np.all(ctrl.stiffness > init_stiffness), (
            f"Stiffness did not increase: {ctrl.stiffness}"
        )

    def test_reset_restores_initial(self) -> None:
        """Reset should restore stiffness and damping to initial values."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        init_K = np.full(n, 300.0)
        init_D = np.full(n, 30.0)
        ctrl = AdaptiveImpedanceController(
            dynamics=dynamics,
            n_joints=n,
            dt=adapter.dt,
            params=AdaptiveImpedanceParams(
                stiffness_init=init_K,
                stiffness_min=np.full(n, 50.0),
                stiffness_max=np.full(n, 2000.0),
                damping_init=init_D,
                damping_min=np.full(n, 5.0),
                damping_max=np.full(n, 200.0),
            ),
        )

        adapter.reset()

        # Change stiffness via adaptation
        for _ in range(50):
            state = adapter.get_state()
            obs = ObserverOutput(tau_ext=np.full(n, 20.0), timestamp=state.timestamp)
            ctrl.compute(state, obs)

        ctrl.reset()
        assert_allclose(ctrl.stiffness, init_K)
        assert_allclose(ctrl.damping, init_D)

    def test_set_target(self) -> None:
        """set_target should update the desired position."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        ctrl = AdaptiveImpedanceController(
            dynamics=dynamics, n_joints=n, dt=adapter.dt
        )
        new_target = np.ones(n) * 0.5
        ctrl.set_target(new_target)
        assert_allclose(ctrl.params.q_desired, new_target)

    def test_default_params(self) -> None:
        """Controller should work with default parameters."""
        adapter = SimAdapter()
        dynamics = MuJoCoDynamics(adapter.env.model)
        n = adapter.n_joints

        ctrl = AdaptiveImpedanceController(
            dynamics=dynamics, n_joints=n, dt=adapter.dt
        )
        adapter.reset()
        state = adapter.get_state()
        output = ctrl.compute(state)
        assert output.tau_cmd.shape == (n,)
