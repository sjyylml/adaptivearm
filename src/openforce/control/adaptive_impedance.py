"""Adaptive impedance controller with force-driven stiffness modulation.

The stiffness adapts online:
- Softens when external forces are detected (compliant behavior)
- Stiffens when position error grows (tracking recovery)
- Damping tracks stiffness to maintain critical damping ratio

Control law:
    tau = g(q) [+ C(q,qd)*qd] + K(t)*(q_d - q) + D(t)*(qd_d - qd)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from openforce.control.base_controller import BaseController
from openforce.core.interfaces import DynamicsModel
from openforce.core.robot_state import RobotState
from openforce.core.types import ControlOutput, ObserverOutput


@dataclass
class AdaptiveImpedanceParams:
    """Parameters for the adaptive impedance controller.

    Attributes:
        stiffness_init: Initial stiffness diagonal, shape (n,). Nm/rad.
        stiffness_min: Minimum stiffness diagonal, shape (n,). Nm/rad.
        stiffness_max: Maximum stiffness diagonal, shape (n,). Nm/rad.
        damping_init: Initial damping diagonal, shape (n,). Nm·s/rad.
        damping_min: Minimum damping diagonal, shape (n,). Nm·s/rad.
        damping_max: Maximum damping diagonal, shape (n,). Nm·s/rad.
        alpha_force: Force-driven softening rate. Higher = faster compliance.
        alpha_error: Error-driven stiffening rate. Higher = faster recovery.
        force_threshold: Dead zone threshold for force (Nm). Forces below this
            don't trigger softening.
        use_coriolis_comp: Whether to compensate Coriolis/centrifugal terms.
        q_desired: Desired joint positions, shape (n,).
        qd_desired: Desired joint velocities, shape (n,).
    """

    stiffness_init: NDArray[np.floating]
    stiffness_min: NDArray[np.floating]
    stiffness_max: NDArray[np.floating]
    damping_init: NDArray[np.floating]
    damping_min: NDArray[np.floating]
    damping_max: NDArray[np.floating]
    alpha_force: float = 5.0
    alpha_error: float = 2.0
    force_threshold: float = 1.0
    use_coriolis_comp: bool = True
    q_desired: NDArray[np.floating] | None = None
    qd_desired: NDArray[np.floating] | None = None


class AdaptiveImpedanceController(BaseController):
    """Adaptive impedance controller with online stiffness modulation.

    Adaptive law:
        dK_force = -alpha_f * (|F| - threshold)_+ * (K - K_min)   (softening)
        dK_error = alpha_e * |e_q| * (K_max - K)                  (stiffening)
        K += (dK_force + dK_error) * dt
        D = D_init * sqrt(K / K_init)   (preserve critical damping ratio)

    Args:
        dynamics: Dynamics model providing gravity/Coriolis computation.
            Must provide mass_matrix, coriolis_vector, gravity_vector methods.
        n_joints: Number of joints.
        dt: Control timestep in seconds.
        params: Adaptive impedance parameters.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        n_joints: int,
        dt: float,
        params: AdaptiveImpedanceParams | None = None,
    ) -> None:
        self._dynamics = dynamics
        self._n = n_joints
        self._dt = dt

        if params is None:
            params = AdaptiveImpedanceParams(
                stiffness_init=np.full(n_joints, 200.0),
                stiffness_min=np.full(n_joints, 20.0),
                stiffness_max=np.full(n_joints, 1000.0),
                damping_init=np.full(n_joints, 40.0),
                damping_min=np.full(n_joints, 5.0),
                damping_max=np.full(n_joints, 200.0),
            )
        self._params = params

        if self._params.q_desired is None:
            self._params.q_desired = np.zeros(n_joints)
        if self._params.qd_desired is None:
            self._params.qd_desired = np.zeros(n_joints)

        # Current adaptive stiffness and damping
        self._K = params.stiffness_init.copy()
        self._D = params.damping_init.copy()

    @property
    def params(self) -> AdaptiveImpedanceParams:
        return self._params

    @property
    def stiffness(self) -> NDArray[np.floating]:
        """Current adaptive stiffness values."""
        return self._K.copy()

    @property
    def damping(self) -> NDArray[np.floating]:
        """Current adaptive damping values."""
        return self._D.copy()

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
        """Reset stiffness and damping to initial values."""
        self._K = self._params.stiffness_init.copy()
        self._D = self._params.damping_init.copy()

    def compute(
        self,
        state: RobotState,
        observer_output: ObserverOutput | None = None,
    ) -> ControlOutput:
        """Compute adaptive impedance control torque.

        Args:
            state: Current robot state.
            observer_output: Force estimate from observer. If provided, drives
                the adaptive stiffness modulation.

        Returns:
            ControlOutput with commanded torques and adaptation info.
        """
        q, qd = state.q, state.qd
        p = self._params
        assert p.q_desired is not None
        assert p.qd_desired is not None

        e_q = p.q_desired - q
        e_qd = p.qd_desired - qd

        # --- Adaptive law ---
        if observer_output is not None:
            tau_ext = observer_output.tau_ext
            force_norm = float(np.linalg.norm(tau_ext))
            error_norm = float(np.linalg.norm(e_q))

            # Force-driven softening: reduce stiffness when forces detected
            force_excess = max(0.0, force_norm - p.force_threshold)
            dK_force = -p.alpha_force * force_excess * (self._K - p.stiffness_min)

            # Error-driven stiffening: increase stiffness when tracking error grows
            dK_error = p.alpha_error * error_norm * (p.stiffness_max - self._K)

            # Update stiffness
            self._K += (dK_force + dK_error) * self._dt
            self._K = np.clip(self._K, p.stiffness_min, p.stiffness_max)

            # Damping tracks stiffness to preserve critical damping ratio
            # D = D_init * sqrt(K / K_init)
            K_init = p.stiffness_init
            ratio = np.sqrt(np.clip(self._K / K_init, 0.0, None))
            self._D = p.damping_init * ratio
            self._D = np.clip(self._D, p.damping_min, p.damping_max)

        # --- Control law ---
        g = self._dynamics.gravity_vector(q)
        tau = g.copy()

        if p.use_coriolis_comp:
            c = self._dynamics.coriolis_vector(q, qd)
            tau += c

        tau += self._K * e_q + self._D * e_qd

        return ControlOutput(
            tau_cmd=tau,
            info={
                "e_q": e_q,
                "e_qd": e_qd,
                "stiffness": self._K.copy(),
                "damping": self._D.copy(),
                "gravity": g,
            },
        )
