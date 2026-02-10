"""Joint-space impedance controller.

Implements: τ = g(q) + K_p·(q_d - q) + K_d·(q̇_d - q̇) [+ feedforward]

In joint-space impedance control the robot behaves as a virtual
mass-spring-damper system at each joint. The controller does NOT
require a force/torque sensor — it shapes the dynamic response.
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
class ImpedanceParams:
    """Impedance controller parameters.

    Attributes:
        stiffness: Joint stiffness diagonal, shape (n,). Units: Nm/rad.
        damping: Joint damping diagonal, shape (n,). Units: Nm·s/rad.
        q_desired: Desired joint positions, shape (n,).
        qd_desired: Desired joint velocities, shape (n,).
        use_coriolis_comp: Whether to compensate Coriolis/centrifugal terms.
        use_inertia_shaping: Whether to use full inertia shaping (M*qdd_d).
    """

    stiffness: NDArray[np.floating]
    damping: NDArray[np.floating]
    q_desired: NDArray[np.floating] | None = None
    qd_desired: NDArray[np.floating] | None = None
    use_coriolis_comp: bool = True
    use_inertia_shaping: bool = False


class ImpedanceController(BaseController):
    """Joint-space impedance controller.

    Computes: τ = g(q) [+ C(q,q̇)·q̇] + K_p·(q_d - q) + K_d·(q̇_d - q̇)

    Optionally compensates Coriolis/centrifugal and gravity to achieve
    the desired impedance behavior at each joint.

    Args:
        dynamics: Dynamics model for gravity/Coriolis computation.
        n_joints: Number of joints.
        params: Impedance parameters (stiffness, damping, targets).
    """

    def __init__(
        self,
        dynamics: MuJoCoDynamics,
        n_joints: int,
        params: ImpedanceParams | None = None,
    ) -> None:
        self._dynamics = dynamics
        self._n = n_joints

        if params is None:
            params = ImpedanceParams(
                stiffness=np.full(n_joints, 100.0),
                damping=np.full(n_joints, 20.0),
            )
        self._params = params

        if self._params.q_desired is None:
            self._params.q_desired = np.zeros(n_joints)
        if self._params.qd_desired is None:
            self._params.qd_desired = np.zeros(n_joints)

    @property
    def params(self) -> ImpedanceParams:
        return self._params

    def set_target(
        self,
        q_desired: NDArray[np.floating],
        qd_desired: NDArray[np.floating] | None = None,
    ) -> None:
        """Update desired position and velocity."""
        self._params.q_desired = np.asarray(q_desired, dtype=np.float64)
        if qd_desired is not None:
            self._params.qd_desired = np.asarray(qd_desired, dtype=np.float64)

    def reset(self) -> None:
        """Reset controller state (stateless controller, no-op)."""

    def compute(
        self,
        state: RobotState,
        observer_output: ObserverOutput | None = None,
    ) -> ControlOutput:
        """Compute impedance control torque.

        Args:
            state: Current robot state.
            observer_output: Not used by impedance controller directly.

        Returns:
            ControlOutput with commanded torques.
        """
        q, qd = state.q, state.qd
        p = self._params

        assert p.q_desired is not None
        assert p.qd_desired is not None

        # Position and velocity errors
        e_q = p.q_desired - q
        e_qd = p.qd_desired - qd

        # Gravity compensation (always on)
        g = self._dynamics.gravity_vector(q)
        tau = g.copy()

        # Optional Coriolis/centrifugal compensation
        if p.use_coriolis_comp:
            c = self._dynamics.coriolis_vector(q, qd)
            tau += c

        # Spring-damper impedance
        tau += p.stiffness * e_q + p.damping * e_qd

        return ControlOutput(
            tau_cmd=tau,
            info={
                "e_q": e_q,
                "e_qd": e_qd,
                "gravity": g,
            },
        )
