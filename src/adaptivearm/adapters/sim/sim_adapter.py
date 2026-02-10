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

    Can be initialized in three ways:

    1. ``SimAdapter()`` — uses the built-in 6-DOF arm model.
    2. ``SimAdapter(xml_path="path/to/model.xml")`` — loads from a file.
    3. ``SimAdapter(model_name="ur5e")`` — looks up a registered model
       from ``adaptivearm.models``.

    Args:
        xml_path: Path to MJCF/URDF model file. None = use default or registry.
        dt: Override simulation timestep.
        model_name: Name of a registered model to load. When provided,
            the model path and ee_site_name are resolved from the registry.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        dt: float | None = None,
        model_name: str | None = None,
    ) -> None:
        ee_site_name = "ee_site"

        if model_name is not None:
            from adaptivearm.models import get_model

            info = get_model(model_name)
            xml_path = info.model_path
            ee_site_name = info.ee_site_name

        self._env = MuJoCoArmEnv(
            xml_path=xml_path, dt=dt, ee_site_name=ee_site_name
        )

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
