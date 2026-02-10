"""IsaacGymAdapter: Batched RobotInterface backed by Isaac Gym GPU simulation.

Unlike the single-environment SimAdapter, this adapter operates on batches:
- get_state() returns a BatchRobotState with shape (num_envs, n_joints)
- send_torque() accepts shape (num_envs, n_joints)
- reset() can reset all or specific environments
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adaptivearm.core.robot_state import RobotState
from adaptivearm.sim.isaacgym_env import IsaacGymArmEnv


@dataclass
class BatchRobotState:
    """Batched robot state for parallel environments.

    All arrays have shape (num_envs, n_joints) or (num_envs, 6, n_joints).

    Attributes:
        q: Joint positions, shape (num_envs, n_joints).
        qd: Joint velocities, shape (num_envs, n_joints).
        tau_motor: Motor torques, shape (num_envs, n_joints).
        timestamp: Simulation time.
        num_envs: Number of parallel environments.
    """

    q: NDArray[np.floating]
    qd: NDArray[np.floating]
    tau_motor: NDArray[np.floating]
    timestamp: float = 0.0
    num_envs: int = 1

    def get_single(self, env_idx: int) -> RobotState:
        """Extract a single-environment RobotState."""
        return RobotState(
            q=self.q[env_idx].copy(),
            qd=self.qd[env_idx].copy(),
            tau_motor=self.tau_motor[env_idx].copy(),
            timestamp=self.timestamp,
        )


class IsaacGymAdapter:
    """Batched robot interface adapter for Isaac Gym GPU simulation.

    Provides the same conceptual interface as SimAdapter but operates on
    batches of environments for GPU-parallel simulation.

    Args:
        asset_file: Path to URDF or MJCF robot asset.
        num_envs: Number of parallel simulation environments.
        dt: Simulation timestep.
        device: Torch device (e.g. "cuda:0").
        spacing: Distance between environments.
    """

    def __init__(
        self,
        asset_file: str | Path,
        num_envs: int = 256,
        dt: float = 0.002,
        device: str = "cuda:0",
        spacing: float = 1.5,
    ) -> None:
        self._env = IsaacGymArmEnv(
            asset_file=asset_file,
            num_envs=num_envs,
            dt=dt,
            device=device,
            spacing=spacing,
        )
        self._last_tau = np.zeros((num_envs, self._env.n_joints))

    @property
    def env(self) -> IsaacGymArmEnv:
        """Access the underlying Isaac Gym environment."""
        return self._env

    @property
    def n_joints(self) -> int:
        return self._env.n_joints

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def dt(self) -> float:
        return self._env.dt

    def get_state(self) -> BatchRobotState:
        """Read current state of all parallel environments.

        Returns:
            BatchRobotState with arrays of shape (num_envs, n_joints).
        """
        return BatchRobotState(
            q=self._env.get_dof_positions(),
            qd=self._env.get_dof_velocities(),
            tau_motor=self._last_tau.copy(),
            timestamp=self._env.time,
            num_envs=self._env.num_envs,
        )

    def send_torque(self, tau: NDArray[np.floating]) -> None:
        """Apply torques and advance all environments one step.

        Args:
            tau: Joint torques, shape (num_envs, n_joints).
        """
        self._last_tau = np.asarray(tau, dtype=np.float64).copy()
        self._env.step(tau)

    def reset(
        self,
        q0: NDArray[np.floating] | None = None,
        env_ids: NDArray[np.integer] | None = None,
    ) -> BatchRobotState:
        """Reset environments and return initial state.

        Args:
            q0: Initial joint positions. Shape (n_joints,) for uniform reset,
                or (num_envs, n_joints) for per-environment reset.
            env_ids: Specific environments to reset. None = all.

        Returns:
            BatchRobotState after reset.
        """
        self._env.reset(q0=q0, env_ids=env_ids)
        self._last_tau[:] = 0.0
        return self.get_state()

    def destroy(self) -> None:
        """Release Isaac Gym resources."""
        self._env.destroy()
