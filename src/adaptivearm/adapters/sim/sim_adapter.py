"""SimAdapter: RobotInterface implementation backed by MuJoCo simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adaptivearm.core.robot_state import RobotState
from adaptivearm.sim.mujoco_env import MuJoCoArmEnv


class SimAdapter:
    """Robot interface adapter for MuJoCo simulation.

    Implements the RobotInterface protocol using a MuJoCo environment.

    Args:
        xml_path: Path to MJCF model. None = built-in 6-DOF arm.
        dt: Override simulation timestep.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        dt: float | None = None,
    ) -> None:
        self._env = MuJoCoArmEnv(xml_path=xml_path, dt=dt)

    @property
    def env(self) -> MuJoCoArmEnv:
        """Access the underlying MuJoCo environment."""
        return self._env

    @property
    def n_joints(self) -> int:
        return self._env.n_joints

    @property
    def dt(self) -> float:
        return self._env.dt

    def get_state(self) -> RobotState:
        """Read current simulation state."""
        data = self._env.data
        nv = self._env.n_joints

        return RobotState(
            q=data.qpos[:nv].copy(),
            qd=data.qvel[:nv].copy(),
            tau_motor=data.ctrl[:nv].copy(),
            timestamp=self._env.time,
            jacobian=self._env.get_jacobian(),
        )

    def send_torque(self, tau: NDArray[np.floating]) -> None:
        """Apply torque and advance simulation one step."""
        self._env.step(tau)

    def reset(self, q0: NDArray[np.floating] | None = None) -> RobotState:
        """Reset simulation and return initial state."""
        self._env.reset(q0)
        return self.get_state()
