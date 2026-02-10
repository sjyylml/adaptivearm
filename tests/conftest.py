"""Shared test fixtures for OpenForce."""

from __future__ import annotations

import numpy as np
import pytest

from openforce.adapters.sim import SimAdapter
from openforce.dynamics.mujoco_dynamics import MuJoCoDynamics
from openforce.sim.mujoco_env import MuJoCoArmEnv


@pytest.fixture
def mujoco_env() -> MuJoCoArmEnv:
    """Create a MuJoCo environment with the default 6-DOF arm."""
    env = MuJoCoArmEnv()
    env.reset()
    return env


@pytest.fixture
def sim_adapter() -> SimAdapter:
    """Create a SimAdapter with the default arm."""
    adapter = SimAdapter()
    adapter.reset()
    return adapter


@pytest.fixture
def dynamics(mujoco_env: MuJoCoArmEnv) -> MuJoCoDynamics:
    """Create a MuJoCoDynamics model."""
    return MuJoCoDynamics(mujoco_env.model)


@pytest.fixture
def n_joints(mujoco_env: MuJoCoArmEnv) -> int:
    return mujoco_env.n_joints


@pytest.fixture
def random_q(n_joints: int) -> np.ndarray:
    """Random joint configuration within typical range."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, size=n_joints)
