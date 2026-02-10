"""Mathematical utility functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def skew_symmetric(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Build 3x3 skew-symmetric matrix from a 3-vector."""
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


hat = skew_symmetric  # alias


def pseudoinverse(
    J: NDArray[np.floating], damping: float = 1e-6
) -> NDArray[np.floating]:
    """Damped Moore-Penrose pseudoinverse of a matrix.

    Uses the damped least-squares formula: J^T (J J^T + λ² I)^{-1}
    which is numerically more stable near singularities than np.linalg.pinv.

    Args:
        J: Matrix of shape (m, n).
        damping: Damping factor λ.

    Returns:
        Pseudoinverse of shape (n, m).
    """
    J = np.asarray(J, dtype=np.float64)
    m = J.shape[0]
    JJT = J @ J.T + damping**2 * np.eye(m)
    return J.T @ np.linalg.inv(JJT)


def wrap_angle(theta: NDArray[np.floating] | float) -> NDArray[np.floating]:
    """Wrap angles to [-π, π]."""
    return np.arctan2(np.sin(theta), np.cos(theta))  # type: ignore[return-value]
