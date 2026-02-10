"""Friction parameter identification from slow-motion data.

Identifies Coulomb (fc) and viscous (fv) friction coefficients per joint:
    τ_friction = fc · sign(q̇) + fv · q̇

Method: collect (τ_residual, q̇) pairs during slow constant-velocity
motions, then fit via least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FrictionParams:
    """Identified friction parameters.

    Attributes:
        coulomb: Coulomb friction per joint, shape (n,). Units: Nm.
        viscous: Viscous friction per joint, shape (n,). Units: Nm·s/rad.
        r_squared: R² fit quality per joint, shape (n,).
        n_samples: Number of data points used.
    """

    coulomb: NDArray[np.floating]
    viscous: NDArray[np.floating]
    r_squared: NDArray[np.floating]
    n_samples: int = 0


class FrictionIdentifier:
    """Identifies Coulomb + viscous friction per joint.

    Collects data during operation, then solves a least-squares problem
    per joint: τ_friction[j] = fc[j]·sign(q̇[j]) + fv[j]·q̇[j].

    Args:
        n_joints: Number of joints.
        velocity_threshold: Minimum |q̇| to include a sample (avoid stiction region).
        max_samples: Maximum stored samples (circular buffer).
    """

    def __init__(
        self,
        n_joints: int,
        velocity_threshold: float = 0.05,
        max_samples: int = 10000,
    ) -> None:
        self._n = n_joints
        self._vel_thresh = velocity_threshold
        self._max_samples = max_samples

        self._velocities: list[NDArray[np.floating]] = []
        self._torques: list[NDArray[np.floating]] = []

    def reset(self) -> None:
        """Clear collected data."""
        self._velocities.clear()
        self._torques.clear()

    def add_sample(
        self,
        qd: NDArray[np.floating],
        tau_friction: NDArray[np.floating],
    ) -> None:
        """Add a data point.

        Args:
            qd: Joint velocities, shape (n,).
            tau_friction: Observed friction torques (from GMO residual
                during free-space motion), shape (n,).
        """
        if len(self._velocities) >= self._max_samples:
            self._velocities.pop(0)
            self._torques.pop(0)
        self._velocities.append(np.asarray(qd, dtype=np.float64).copy())
        self._torques.append(np.asarray(tau_friction, dtype=np.float64).copy())

    def identify(self) -> FrictionParams:
        """Run least-squares identification on collected data.

        Returns:
            FrictionParams with identified Coulomb and viscous coefficients.
        """
        if len(self._velocities) < 10:
            return FrictionParams(
                coulomb=np.zeros(self._n),
                viscous=np.zeros(self._n),
                r_squared=np.zeros(self._n),
                n_samples=len(self._velocities),
            )

        qd_all = np.array(self._velocities)   # (N, n_joints)
        tau_all = np.array(self._torques)      # (N, n_joints)

        fc = np.zeros(self._n)
        fv = np.zeros(self._n)
        r2 = np.zeros(self._n)

        for j in range(self._n):
            # Filter by velocity threshold
            mask = np.abs(qd_all[:, j]) > self._vel_thresh
            if np.sum(mask) < 5:
                continue

            qd_j = qd_all[mask, j]
            tau_j = tau_all[mask, j]

            # Build regressor: [sign(qd), qd]
            A = np.column_stack([np.sign(qd_j), qd_j])
            # Least squares: tau = A · [fc, fv]^T
            result, residuals, _, _ = np.linalg.lstsq(A, tau_j, rcond=None)
            fc[j] = abs(result[0])  # Coulomb is always positive magnitude
            fv[j] = result[1]

            # R² score
            ss_res = np.sum((tau_j - A @ result) ** 2)
            ss_tot = np.sum((tau_j - np.mean(tau_j)) ** 2)
            r2[j] = 1.0 - ss_res / (ss_tot + 1e-10)

        return FrictionParams(
            coulomb=fc,
            viscous=fv,
            r_squared=r2,
            n_samples=len(self._velocities),
        )
