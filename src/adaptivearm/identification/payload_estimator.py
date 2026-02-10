"""Online payload estimation via recursive least squares (RLS).

Estimates the mass, center-of-gravity, and inertia of an unknown
payload attached to the end-effector, using the observer residual
and the robot's Jacobian.

The payload wrench at the end-effector in gravity is:
    F_payload = [m·g_world; m·(p_cog × g_world)]

Projected to joint space: τ_payload = J^T · F_payload

By observing τ_payload (from GMO) at multiple configurations, we
can estimate [m, m·cx, m·cy, m·cz] via RLS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PayloadEstimate:
    """Estimated payload parameters.

    Attributes:
        mass: Estimated mass in kg.
        cog: Center of gravity relative to end-effector, shape (3,), in meters.
        confidence: Estimation confidence (inverse of parameter covariance trace).
        n_samples: Number of samples used.
    """

    mass: float = 0.0
    cog: NDArray[np.floating] = None  # type: ignore[assignment]
    confidence: float = 0.0
    n_samples: int = 0

    def __post_init__(self) -> None:
        if self.cog is None:
            self.cog = np.zeros(3)


class PayloadEstimator:
    """Online RLS payload estimator.

    Estimates payload parameters θ = [m, m·cx, m·cy, m·cz] from
    the relationship: τ_observed = Φ(q, J) · θ

    where Φ is the observation matrix constructed from gravity and Jacobian.

    Args:
        n_joints: Number of joints.
        gravity: Gravity vector in world frame, default [0, 0, -9.81].
        forgetting_factor: RLS forgetting factor (0.95-1.0). Lower = adapts faster.
        initial_covariance: Initial P matrix diagonal value.
    """

    def __init__(
        self,
        n_joints: int,
        gravity: NDArray[np.floating] | None = None,
        forgetting_factor: float = 0.998,
        initial_covariance: float = 1000.0,
    ) -> None:
        self._n = n_joints
        self._g = np.array([0.0, 0.0, -9.81]) if gravity is None else np.asarray(gravity)
        self._lambda = forgetting_factor

        # RLS state: 4 parameters [m, m*cx, m*cy, m*cz]
        self._n_params = 4
        self._theta = np.zeros(self._n_params)
        self._P = np.eye(self._n_params) * initial_covariance
        self._n_samples = 0

    def reset(self, initial_covariance: float = 1000.0) -> None:
        """Reset estimator state."""
        self._theta[:] = 0.0
        self._P = np.eye(self._n_params) * initial_covariance
        self._n_samples = 0

    def update(
        self,
        tau_ext: NDArray[np.floating],
        jacobian: NDArray[np.floating],
    ) -> PayloadEstimate:
        """Process one observation and update payload estimate.

        Args:
            tau_ext: Observed external joint torques from GMO, shape (n_joints,).
            jacobian: End-effector Jacobian, shape (6, n_joints).

        Returns:
            Current PayloadEstimate.
        """
        # Build observation matrix Φ such that τ_ext ≈ Φ · θ
        # τ_ext = J^T · [m·g; m·(p_cog × g)] = J^T · [g, skew(g)] · [m; m·p_cog]
        J_trans = jacobian[:3, :]  # (3, n_joints) — translational
        J_rot = jacobian[3:, :]    # (3, n_joints) — rotational

        g = self._g  # (3,)

        # For force part: τ_f = J_trans^T · (m · g) → Φ_f · m
        # For moment part: τ_m = J_rot^T · (m · p_cog × g) = J_rot^T · skew(g) · (m·p_cog)
        # skew(g) such that g × p = skew(g) · p
        S_g = np.array([
            [0, -g[2], g[1]],
            [g[2], 0, -g[0]],
            [-g[1], g[0], 0],
        ])

        # Φ: (n_joints, 4)
        # Column 0: J_trans^T · g  (mass contribution)
        # Columns 1-3: -J_rot^T · S_g  (CoG contribution, note sign: p × g = -g × p)
        phi = np.zeros((self._n, self._n_params))
        phi[:, 0] = J_trans.T @ g
        phi[:, 1:4] = -J_rot.T @ S_g

        # RLS update: process each joint as a separate measurement
        for i in range(self._n):
            phi_i = phi[i, :]  # (4,)
            y_i = tau_ext[i]   # scalar

            # Prediction error
            e = y_i - phi_i @ self._theta

            # Kalman-like gain
            denom = self._lambda + phi_i @ self._P @ phi_i
            if abs(denom) < 1e-12:
                continue
            K = (self._P @ phi_i) / denom

            # Update
            self._theta += K * e
            self._P = (self._P - np.outer(K, phi_i @ self._P)) / self._lambda

        self._n_samples += 1
        return self.get_estimate()

    def get_estimate(self) -> PayloadEstimate:
        """Get current payload estimate without updating."""
        mass = max(self._theta[0], 0.0)  # mass must be non-negative
        cog = self._theta[1:4] / mass if mass > 1e-6 else np.zeros(3)

        confidence = 1.0 / (np.trace(self._P) + 1e-10)

        return PayloadEstimate(
            mass=mass,
            cog=cog,
            confidence=confidence,
            n_samples=self._n_samples,
        )
