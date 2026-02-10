"""Simulation environments: MuJoCo and Isaac Gym."""

from openforce.sim.mujoco_env import MuJoCoArmEnv
from openforce.sim.virtual_sensor import VirtualForceSensor

__all__ = [
    "MuJoCoArmEnv",
    "VirtualForceSensor",
]

# Isaac Gym is optional — only export if available
try:
    from openforce.sim.isaacgym_env import IsaacGymArmEnv as IsaacGymArmEnv

    __all__.append("IsaacGymArmEnv")
except ImportError:
    pass
