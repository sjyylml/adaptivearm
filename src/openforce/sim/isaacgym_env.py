"""Isaac Gym GPU-parallel simulation environment for robot arms.

Requires Isaac Gym Preview (isaacgym package) to be installed separately.
See: https://developer.nvidia.com/isaac-gym
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from isaacgym import gymapi, gymtorch  # type: ignore[import-untyped]

    ISAACGYM_AVAILABLE = True
except ImportError:
    ISAACGYM_AVAILABLE = False


def _require_isaacgym() -> None:
    if not ISAACGYM_AVAILABLE:
        raise ImportError(
            "Isaac Gym is not installed. Install it from "
            "https://developer.nvidia.com/isaac-gym and then "
            "pip install -e '.[isaacgym]'"
        )


class IsaacGymArmEnv:
    """Isaac Gym parallel simulation environment for robot arms.

    Runs multiple robot instances on GPU for high-throughput simulation.
    All state tensors are torch tensors on the GPU device.

    Args:
        asset_file: Path to URDF or MJCF asset file for the robot arm.
        num_envs: Number of parallel environments.
        spacing: Distance between environments in the grid.
        dt: Simulation timestep in seconds.
        device: Torch device string (e.g. "cuda:0").
        use_gpu_pipeline: Whether to use GPU pipeline for state transfer.
        physics_engine: "physx" or "flex".
    """

    def __init__(
        self,
        asset_file: str | Path,
        num_envs: int = 256,
        spacing: float = 1.5,
        dt: float = 0.002,
        device: str = "cuda:0",
        use_gpu_pipeline: bool = True,
        physics_engine: str = "physx",
    ) -> None:
        _require_isaacgym()

        self._num_envs = num_envs
        self._device = device
        self._dt = dt
        self._asset_file = str(asset_file)

        # Initialize Isaac Gym
        self._gym = gymapi.acquire_gym()

        # Simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = dt
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = use_gpu_pipeline

        if physics_engine == "physx":
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.contact_offset = 0.002
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.use_gpu = True
            engine = gymapi.SIM_PHYSX
        else:
            engine = gymapi.SIM_FLEX

        self._sim = self._gym.create_sim(0, 0, engine, sim_params)

        # Load robot asset
        asset_root = str(Path(self._asset_file).parent)
        asset_name = Path(self._asset_file).name
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.01

        self._asset = self._gym.load_asset(
            self._sim, asset_root, asset_name, asset_options
        )
        self._n_joints = self._gym.get_asset_dof_count(self._asset)

        # Configure DOF properties (position + effort control)
        dof_props = self._gym.get_asset_dof_properties(self._asset)
        for i in range(self._n_joints):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            dof_props["stiffness"][i] = 0.0
            dof_props["damping"][i] = 0.0

        # Create environments
        self._envs: list[Any] = []
        self._actor_handles: list[Any] = []

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(num_envs):
            env = self._gym.create_env(self._sim, lower, upper, int(np.sqrt(num_envs)))
            actor = self._gym.create_actor(env, self._asset, gymapi.Transform(), f"arm_{i}", i, 1)
            self._gym.set_actor_dof_properties(env, actor, dof_props)
            self._envs.append(env)
            self._actor_handles.append(actor)

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self._gym.add_ground(self._sim, plane_params)

        # Prepare simulation
        self._gym.prepare_sim(self._sim)

        # Acquire GPU tensor references
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)

        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # Shape: (num_envs * n_joints, 2) — [pos, vel] per DOF

        dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
        self._dof_force = gymtorch.wrap_tensor(dof_force_tensor)

        self._time = 0.0

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def n_joints(self) -> int:
        return self._n_joints

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return self._time

    @property
    def device(self) -> str:
        return self._device

    def reset(
        self, q0: NDArray[np.floating] | None = None, env_ids: NDArray[np.integer] | None = None
    ) -> None:
        """Reset environments to initial or given configuration.

        Args:
            q0: Joint positions, shape (n_joints,) for all envs, or
                (num_envs, n_joints) for per-env configuration.
            env_ids: Specific environment indices to reset. None = all.
        """
        if env_ids is None:
            env_ids_np: NDArray[np.integer] = np.arange(self._num_envs)
        else:
            env_ids_np = np.asarray(env_ids)

        for idx in env_ids_np:
            i = int(idx)
            dof_states = self._gym.get_actor_dof_states(
                self._envs[i], self._actor_handles[i], gymapi.STATE_ALL
            )
            for j in range(self._n_joints):
                if q0 is not None:
                    if q0.ndim == 1:
                        dof_states["pos"][j] = q0[j]
                    else:
                        dof_states["pos"][j] = q0[i, j]
                else:
                    dof_states["pos"][j] = 0.0
                dof_states["vel"][j] = 0.0
            self._gym.set_actor_dof_states(
                self._envs[i], self._actor_handles[i], dof_states, gymapi.STATE_ALL
            )

        self._time = 0.0

    def step(self, tau: NDArray[np.floating] | None = None) -> None:
        """Advance all environments by one timestep.

        Args:
            tau: Joint torques, shape (num_envs, n_joints). If None, zero torque.
        """
        if tau is not None:
            tau_tensor = torch.tensor(tau, dtype=torch.float32, device=self._device)
            self._gym.set_dof_actuation_force_tensor(
                self._sim, gymtorch.unwrap_tensor(tau_tensor.flatten())
            )

        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._time += self._dt

    def get_dof_positions(self) -> NDArray[np.floating]:
        """Get joint positions for all environments.

        Returns:
            Positions, shape (num_envs, n_joints).
        """
        pos = self._dof_state[:, 0].reshape(self._num_envs, self._n_joints)
        return pos.cpu().numpy()

    def get_dof_velocities(self) -> NDArray[np.floating]:
        """Get joint velocities for all environments.

        Returns:
            Velocities, shape (num_envs, n_joints).
        """
        vel = self._dof_state[:, 1].reshape(self._num_envs, self._n_joints)
        return vel.cpu().numpy()

    def get_dof_forces(self) -> NDArray[np.floating]:
        """Get measured DOF forces for all environments.

        Returns:
            Forces, shape (num_envs, n_joints).
        """
        forces = self._dof_force.reshape(self._num_envs, self._n_joints)
        return forces.cpu().numpy()

    def apply_external_force(
        self,
        body_index: int,
        forces: NDArray[np.floating],
        torques: NDArray[np.floating] | None = None,
        env_ids: NDArray[np.integer] | None = None,
    ) -> None:
        """Apply external force/torque to a body across environments.

        Args:
            body_index: Index of the rigid body in the actor.
            forces: Forces, shape (num_envs, 3) or (3,) for broadcast.
            torques: Torques, shape (num_envs, 3) or (3,). Default zero.
            env_ids: Environments to apply to. None = all.
        """
        n = self._num_envs if env_ids is None else len(env_ids)
        forces = np.broadcast_to(forces, (n, 3))
        torques = np.zeros((n, 3)) if torques is None else np.broadcast_to(torques, (n, 3))

        # Construct full force tensor: (num_envs, num_bodies, 6)
        num_bodies = self._gym.get_asset_rigid_body_count(self._asset)
        force_tensor = torch.zeros(
            (self._num_envs, num_bodies, 3), dtype=torch.float32, device=self._device
        )
        torque_tensor = torch.zeros(
            (self._num_envs, num_bodies, 3), dtype=torch.float32, device=self._device
        )

        target_envs = range(self._num_envs) if env_ids is None else env_ids
        for j, env_idx in enumerate(target_envs):
            force_tensor[int(env_idx), body_index] = torch.tensor(
                forces[j], dtype=torch.float32, device=self._device
            )
            torque_tensor[int(env_idx), body_index] = torch.tensor(
                torques[j], dtype=torch.float32, device=self._device
            )

        self._gym.apply_rigid_body_force_tensors(
            self._sim,
            gymtorch.unwrap_tensor(force_tensor.reshape(-1, 3)),
            gymtorch.unwrap_tensor(torque_tensor.reshape(-1, 3)),
            gymapi.ENV_SPACE,
        )

    def destroy(self) -> None:
        """Clean up Isaac Gym resources."""
        self._gym.destroy_sim(self._sim)
