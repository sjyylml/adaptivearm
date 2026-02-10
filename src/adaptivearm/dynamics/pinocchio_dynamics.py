"""Pinocchio-based rigid-body dynamics computation from URDF.

Provides an alternative dynamics backend using the Pinocchio library,
which loads robot models from URDF files. Satisfies the ExtendedDynamicsModel protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import pinocchio as pin

    _HAS_PINOCCHIO = True
except ImportError:
    _HAS_PINOCCHIO = False


def _require_pinocchio() -> None:
    if not _HAS_PINOCCHIO:
        raise ImportError(
            "pinocchio is required for PinocchioDynamics. "
            "Install it with: pip install pin"
        )


class PinocchioDynamics:
    """Compute M(q), C(q, qd)*qd, g(q) using Pinocchio from a URDF model.

    Satisfies the ``ExtendedDynamicsModel`` protocol.

    Args:
        urdf_path: Path to the URDF file.
        package_dirs: Optional list of package directories for mesh lookup.
        floating_base: Whether the robot has a floating base.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        package_dirs: list[str] | None = None,
        floating_base: bool = False,
    ) -> None:
        _require_pinocchio()
        assert _HAS_PINOCCHIO  # for type narrowing

        urdf_str = str(urdf_path)
        if floating_base:
            self._model = pin.buildModelFromUrdf(urdf_str, pin.JointModelFreeFlyer())
        elif package_dirs:
            self._model = pin.buildModelFromUrdf(urdf_str, package_dirs)
        else:
            self._model = pin.buildModelFromUrdf(urdf_str)

        self._data = self._model.createData()
        self._nv: int = int(self._model.nv)
        self._nq: int = int(self._model.nq)

        # Extract joint damping from URDF (stored in model.damping)
        self._damping = np.array(self._model.damping, dtype=np.float64)

    @property
    def n_joints(self) -> int:
        """Number of velocity degrees of freedom."""
        return self._nv

    @property
    def nq(self) -> int:
        """Configuration space dimension."""
        return self._nq

    def mass_matrix(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute joint-space inertia matrix M(q), shape (nv, nv).

        Uses CRBA (Composite Rigid Body Algorithm) and symmetrizes the result.
        """
        assert _HAS_PINOCCHIO
        q_ = np.asarray(q, dtype=np.float64)
        pin.crba(self._model, self._data, q_)
        M = np.array(self._data.M, dtype=np.float64)
        # CRBA only fills upper triangle
        M = M + M.T - np.diag(np.diag(M))
        return M

    def coriolis_vector(
        self, q: NDArray[np.floating], qd: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute Coriolis/centrifugal vector C(q, qd)*qd, shape (nv,).

        Computed as nle(q, qd) - g(q), where nle includes both Coriolis and gravity.
        """
        assert _HAS_PINOCCHIO
        q_ = np.asarray(q, dtype=np.float64)
        qd_ = np.asarray(qd, dtype=np.float64)
        nle = pin.nonLinearEffects(self._model, self._data, q_, qd_)
        g = self.gravity_vector(q)
        return np.array(nle - g, dtype=np.float64)

    def gravity_vector(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute gravity torque vector g(q), shape (nv,)."""
        assert _HAS_PINOCCHIO
        q_ = np.asarray(q, dtype=np.float64)
        pin.computeGeneralizedGravity(self._model, self._data, q_)
        return np.array(self._data.g, dtype=np.float64)

    def passive_torque(
        self, qd: NDArray[np.floating], q: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute passive joint torques (viscous damping from URDF).

        Args:
            qd: Joint velocities, shape (nv,).
            q: Joint positions (unused, kept for protocol compatibility).

        Returns:
            Passive torques = -damping * qd, shape (nv,).
        """
        qd_ = np.asarray(qd, dtype=np.float64)
        return -self._damping[: self._nv] * qd_

    def forward_kinematics(
        self, q: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute end-effector pose via forward kinematics.

        Returns the SE(3) homogeneous transform of the last frame as a 4x4 matrix.
        """
        assert _HAS_PINOCCHIO
        q_ = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self._model, self._data, q_)
        pin.updateFramePlacements(self._model, self._data)
        # Last frame (typically end-effector)
        frame_id = self._model.nframes - 1
        oMf = self._data.oMf[frame_id]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(oMf.rotation)
        T[:3, 3] = np.array(oMf.translation)
        return T

    def jacobian(
        self,
        q: NDArray[np.floating],
        frame_id: int | None = None,
    ) -> NDArray[np.floating]:
        """Compute the geometric Jacobian at the given frame.

        Args:
            q: Joint configuration, shape (nq,).
            frame_id: Pinocchio frame ID. Defaults to last frame.

        Returns:
            Jacobian of shape (6, nv).
        """
        assert _HAS_PINOCCHIO
        q_ = np.asarray(q, dtype=np.float64)
        if frame_id is None:
            frame_id = self._model.nframes - 1
        pin.computeFrameJacobian(
            self._model, self._data, q_, frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        J = pin.getFrameJacobian(
            self._model, self._data, frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        return np.array(J, dtype=np.float64)
