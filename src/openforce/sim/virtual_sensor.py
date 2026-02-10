"""Virtual force sensor for simulation ground-truth comparison."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from openforce.sim.mujoco_env import MuJoCoArmEnv


class VirtualForceSensor:
    """Extracts ground-truth external torques from MuJoCo simulation.

    Computes the joint-space projection of ``xfrc_applied`` using the body
    Jacobians, providing the "true" external torques for validating observers.

    Args:
        env: MuJoCo simulation environment.
    """

    def __init__(self, env: MuJoCoArmEnv) -> None:
        self._env = env

    def get_external_torques(self) -> NDArray[np.floating]:
        """Get ground-truth external joint torques from all applied forces.

        Computes τ_ext = Σ J_i^T @ F_i for all bodies with non-zero xfrc_applied.

        Returns:
            External joint torques, shape (nv,).
        """
        model = self._env.model
        data = self._env.data
        nv = model.nv

        tau_ext = np.zeros(nv, dtype=np.float64)

        for body_id in range(model.nbody):
            wrench = data.xfrc_applied[body_id]
            if np.any(wrench != 0):
                # Compute body Jacobian
                jacp = np.zeros((3, nv))
                jacr = np.zeros((3, nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                # τ = J_trans^T @ F + J_rot^T @ T
                tau_ext += jacp.T @ wrench[:3] + jacr.T @ wrench[3:]

        return tau_ext

    def get_external_wrench(self, body_name: str) -> NDArray[np.floating]:
        """Get the applied wrench on a specific body.

        Args:
            body_name: Name of the body.

        Returns:
            Wrench [fx, fy, fz, tx, ty, tz] in world frame.
        """
        body_id = mujoco.mj_name2id(
            self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        return np.array(self._env.data.xfrc_applied[body_id], dtype=np.float64)
