"""Simulation environments: MuJoCo and Isaac Gym."""

from adaptivearm.sim.mujoco_env import MuJoCoArmEnv
from adaptivearm.sim.virtual_sensor import VirtualForceSensor

__all__ = [
    "MuJoCoArmEnv",
    "VirtualForceSensor",
]

# Isaac Gym is optional — only export if available
try:
    from adaptivearm.sim.isaacgym_env import IsaacGymArmEnv as IsaacGymArmEnv

    __all__.append("IsaacGymArmEnv")
except ImportError:
    pass
