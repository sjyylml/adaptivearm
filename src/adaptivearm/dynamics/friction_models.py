"""Joint friction models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CoulombViscousFriction:
    """Coulomb + viscous friction model.

    τ_friction = fc * sign(qd) + fv * qd

    Args:
        fc: Coulomb friction coefficients, shape (n_joints,).
        fv: Viscous friction coefficients, shape (n_joints,).
        velocity_threshold: Below this speed, use tanh approximation to avoid
            discontinuity. Default 0.01 rad/s.
    """

    def __init__(
        self,
        fc: NDArray[np.floating],
        fv: NDArray[np.floating],
        velocity_threshold: float = 0.01,
    ) -> None:
        self._fc = np.asarray(fc, dtype=np.float64)
        self._fv = np.asarray(fv, dtype=np.float64)
        self._vel_thresh = velocity_threshold

    def compute(self, qd: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute friction torque.

        Args:
            qd: Joint velocities, shape (n_joints,).

        Returns:
            Friction torques, shape (n_joints,).
        """
        qd = np.asarray(qd, dtype=np.float64)
        # Smooth sign approximation near zero velocity
        sign_qd = np.tanh(qd / self._vel_thresh)
        return self._fc * sign_qd + self._fv * qd
