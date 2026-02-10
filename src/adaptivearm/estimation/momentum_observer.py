"""Generalized Momentum Observer (GMO) for external force estimation.

Reference:
    De Luca, A., & Mattone, R. (2005). Sensorless robot collision detection
    and hybrid force/motion control.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from adaptivearm.core.robot_state import RobotState
from adaptivearm.core.types import ObserverOutput
from adaptivearm.dynamics.friction_models import CoulombViscousFriction
from adaptivearm.dynamics.mujoco_dynamics import MuJoCoDynamics
from adaptivearm.estimation.base_observer import BaseObserver
from adaptivearm.utils.math_utils import pseudoinverse
from adaptivearm.utils.signal_processing import LowPassFilter


class MomentumObserver(BaseObserver):
    """Generalized Momentum Observer (GMO).

    Estimates external joint torques from the generalized momentum residual:

        p(t) = M(q) * qd                               (generalized momentum)
        r(t) = K_O * ∫[τ_motor + τ_passive - C*qd - g(q) + r] dt - K_O * p(t)

    The residual r(t) converges to τ_ext with first-order dynamics governed
    by K_O. Uses trapezoidal integration.

    The equation of motion is: M*qdd + C*qd + g = τ_motor + τ_passive + τ_ext
    where τ_passive includes joint damping (from MuJoCo's dof_damping).

    Args:
        dynamics: MuJoCo dynamics model providing M(q), C(q,qd)*qd, g(q),
            and passive torques.
        n_joints: Number of robot joints.
        dt: Control timestep in seconds.
        gains: Observer diagonal gain vector, shape (n_joints,).
            Higher gains → faster convergence but more noise sensitivity.
        friction_model: Optional additional friction model for compensation
            (on top of MuJoCo's built-in passive damping).
        lowpass_cutoff: Low-pass filter cutoff frequency in Hz (0 = disabled).
    """

    def __init__(
        self,
        dynamics: MuJoCoDynamics,
        n_joints: int,
        dt: float,
        gains: NDArray[np.floating] | None = None,
        friction_model: CoulombViscousFriction | None = None,
        lowpass_cutoff: float = 0.0,
    ) -> None:
        self._dynamics = dynamics
        self._n = n_joints
        self._dt = dt
        self._K = np.diag(gains if gains is not None else np.full(n_joints, 10.0))
        self._friction = friction_model
        self._filter = LowPassFilter(lowpass_cutoff, dt, n_joints)

        # Internal state
        self._integral = np.zeros(n_joints)
        self._prev_integrand = np.zeros(n_joints)
        self._r = np.zeros(n_joints)
        self._initialized = False

    def reset(self) -> None:
        """Reset observer state."""
        self._integral[:] = 0.0
        self._prev_integrand[:] = 0.0
        self._r[:] = 0.0
        self._initialized = False
        self._filter.reset()

    def update(self, state: RobotState) -> ObserverOutput:
        """Run one observer step.

        Args:
            state: Current robot state.

        Returns:
            ObserverOutput with estimated external joint torques (and Cartesian
            wrench if Jacobian is available).
        """
        q, qd, tau_motor = state.q, state.qd, state.tau_motor

        # Dynamics terms
        M = self._dynamics.mass_matrix(q)
        c = self._dynamics.coriolis_vector(q, qd)
        g = self._dynamics.gravity_vector(q)
        tau_passive = self._dynamics.passive_torque(qd, q)

        # Generalized momentum
        p = M @ qd

        # Additional friction compensation (beyond MuJoCo's built-in damping)
        tau_friction: NDArray[np.floating] = np.zeros(self._n)
        if self._friction is not None:
            tau_friction = self._friction.compute(qd)

        # Integrand: τ_motor + τ_passive - τ_friction_extra - C*qd - g(q) + r
        # From EoM: M*qdd + C*qd + g = τ_motor + τ_passive + τ_ext
        # So: p_dot = (τ_motor + τ_passive + τ_ext) - C*qd - g
        # Integrand without τ_ext: τ_motor + τ_passive - C*qd - g + r
        integrand = tau_motor + tau_passive - tau_friction - c - g + self._r

        if not self._initialized:
            self._integral = p.copy()
            self._prev_integrand = integrand.copy()
            self._initialized = True
            return ObserverOutput(
                tau_ext=np.zeros(self._n),
                timestamp=state.timestamp,
                residual_raw=np.zeros(self._n),
            )

        # Trapezoidal integration
        self._integral += 0.5 * self._dt * (integrand + self._prev_integrand)
        self._prev_integrand = integrand.copy()

        # Residual: r = K_O * (p - β), where β is the integral
        r_raw = self._K @ (p - self._integral)
        self._r = r_raw.copy()

        # Optional low-pass filtering
        r_filtered = self._filter(r_raw)

        # Cartesian wrench via Jacobian pseudoinverse
        wrench = None
        if state.jacobian is not None:
            J_pinv = pseudoinverse(state.jacobian)
            wrench = J_pinv.T @ r_filtered

        return ObserverOutput(
            tau_ext=r_filtered,
            wrench_ext=wrench,
            timestamp=state.timestamp,
            residual_raw=r_raw,
        )
