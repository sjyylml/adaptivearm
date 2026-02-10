"""MuJoCo-based rigid-body dynamics computation."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray


class MuJoCoDynamics:
    """Compute M(q), C(q, qd)*qd, and g(q) using MuJoCo.

    Uses ``qfrc_bias`` (Coriolis + gravity) and ``qfrc_passive`` (damping)
    from MuJoCo's forward dynamics to extract the standard rigid-body terms.

    Args:
        model: A MuJoCo model (mjModel).
    """

    def __init__(self, model: mujoco.MjModel) -> None:  # type: ignore[no-any-unimported]
        self._model = model
        self._data = mujoco.MjData(model)
        self._nv: int = int(model.nv)

    @property
    def n_joints(self) -> int:
        return self._nv

    def mass_matrix(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute joint-space inertia matrix M(q).

        Includes armature inertia.
        """
        self._set_state(q, np.zeros(self._nv))
        M = np.zeros((self._nv, self._nv), dtype=np.float64)
        mujoco.mj_fullM(self._model, M, self._data.qM)
        return M

    def coriolis_vector(
        self, q: NDArray[np.floating], qd: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute Coriolis/centrifugal vector C(q, qd)*qd.

        This does NOT include passive damping or gravity — only the
        velocity-dependent Coriolis and centrifugal terms.
        """
        self._set_state(q, qd)
        bias = np.array(self._data.qfrc_bias[: self._nv], dtype=np.float64)
        passive = np.array(self._data.qfrc_passive[: self._nv], dtype=np.float64)
        g = self.gravity_vector(q)
        return bias - g - passive

    def gravity_vector(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute gravity torque vector g(q)."""
        self._set_state(q, np.zeros(self._nv))
        return np.array(self._data.qfrc_bias[: self._nv], dtype=np.float64)

    def passive_torque(
        self, qd: NDArray[np.floating], q: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute passive joint torques (damping, etc.).

        Args:
            qd: Joint velocities.
            q: Joint positions (needed to set state).

        Returns:
            Passive torques, shape (n,).
        """
        self._set_state(q, qd)
        return np.array(self._data.qfrc_passive[: self._nv], dtype=np.float64)

    def _set_state(
        self, q: NDArray[np.floating], qd: NDArray[np.floating]
    ) -> None:
        """Set model state and run forward kinematics/dynamics."""
        self._data.qpos[:] = q
        self._data.qvel[:] = qd
        self._data.qacc[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
