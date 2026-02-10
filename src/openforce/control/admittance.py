"""Admittance controller.

Implements a virtual mass-spring-damper that converts estimated external
force into a reference trajectory modification:

    M_d · Δq̈ + D_d · Δq̇ + K_d · Δq = F_ext

The output Δq is added to the nominal trajectory, and an inner position
controller tracks the modified reference. This allows force-guided motion
without direct torque-level force control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from openforce.control.base_controller import BaseController
from openforce.core.robot_state import RobotState
from openforce.core.types import ControlOutput, ObserverOutput
from openforce.dynamics.mujoco_dynamics import MuJoCoDynamics


@dataclass
class AdmittanceParams:
    """Admittance controller parameters.

    The virtual dynamics: M_d · Δq̈ + D_d · Δq̇ + K_d · Δq = τ_ext

    Attributes:
        mass: Virtual inertia diagonal, shape (n,). Units: kg·m².
        damping: Virtual damping diagonal, shape (n,). Units: Nm·s/rad.
        stiffness: Virtual stiffness diagonal, shape (n,). Units: Nm/rad.
        inner_kp: Inner position loop proportional gain.
        inner_kd: Inner position loop derivative gain.
    """

    mass: NDArray[np.floating]
    damping: NDArray[np.floating]
    stiffness: NDArray[np.floating]
    inner_kp: NDArray[np.floating]
    inner_kd: NDArray[np.floating]


class AdmittanceController(BaseController):
    """Admittance controller with inner position loop.

    Outer loop: admittance filter converts τ_ext → Δq displacement.
    Inner loop: PD + gravity compensation tracks q_nominal + Δq.

    Args:
        dynamics: Dynamics model for gravity compensation.
        n_joints: Number of joints.
        dt: Control timestep.
        params: Admittance parameters.
    """

    def __init__(
        self,
        dynamics: MuJoCoDynamics,
        n_joints: int,
        dt: float,
        params: AdmittanceParams | None = None,
    ) -> None:
        self._dynamics = dynamics
        self._n = n_joints
        self._dt = dt

        if params is None:
            params = AdmittanceParams(
                mass=np.full(n_joints, 1.0),
                damping=np.full(n_joints, 40.0),
                stiffness=np.full(n_joints, 10.0),
                inner_kp=np.full(n_joints, 200.0),
                inner_kd=np.full(n_joints, 30.0),
            )
        self._params = params

        # Admittance filter state
        self._delta_q = np.zeros(n_joints)
        self._delta_qd = np.zeros(n_joints)

        # Nominal trajectory
        self._q_nominal = np.zeros(n_joints)
        self._qd_nominal = np.zeros(n_joints)

    @property
    def params(self) -> AdmittanceParams:
        return self._params

    @property
    def delta_q(self) -> NDArray[np.floating]:
        """Current admittance displacement."""
        return self._delta_q.copy()

    def set_nominal(
        self,
        q_nominal: NDArray[np.floating],
        qd_nominal: NDArray[np.floating] | None = None,
    ) -> None:
        """Set nominal (desired) trajectory."""
        self._q_nominal = np.asarray(q_nominal, dtype=np.float64)
        if qd_nominal is not None:
            self._qd_nominal = np.asarray(qd_nominal, dtype=np.float64)

    def reset(self) -> None:
        """Reset admittance filter state."""
        self._delta_q[:] = 0.0
        self._delta_qd[:] = 0.0

    def compute(
        self,
        state: RobotState,
        observer_output: ObserverOutput | None = None,
    ) -> ControlOutput:
        """Compute admittance control torque.

        Args:
            state: Current robot state.
            observer_output: Force estimate — required for admittance control.

        Returns:
            ControlOutput with commanded torques.
        """
        q, qd = state.q, state.qd
        p = self._params

        # Get estimated external torque
        tau_ext: NDArray[np.floating] = np.zeros(self._n)
        if observer_output is not None:
            tau_ext = observer_output.tau_ext

        # Admittance filter: M_d · Δq̈ + D_d · Δq̇ + K_d · Δq = τ_ext
        # Solve for Δq̈:
        delta_qdd = (tau_ext - p.damping * self._delta_qd - p.stiffness * self._delta_q) / p.mass

        # Semi-implicit Euler integration
        self._delta_qd += delta_qdd * self._dt
        self._delta_q += self._delta_qd * self._dt

        # Modified reference
        q_ref = self._q_nominal + self._delta_q
        qd_ref = self._qd_nominal + self._delta_qd

        # Inner PD loop + gravity compensation
        g = self._dynamics.gravity_vector(q)
        tau = g + p.inner_kp * (q_ref - q) + p.inner_kd * (qd_ref - qd)

        return ControlOutput(
            tau_cmd=tau,
            info={
                "delta_q": self._delta_q.copy(),
                "delta_qd": self._delta_qd.copy(),
                "q_ref": q_ref,
                "tau_ext": tau_ext,
            },
        )
