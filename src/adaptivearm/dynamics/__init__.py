"""Dynamics computation backends."""

from adaptivearm.dynamics.friction_models import CoulombViscousFriction
from adaptivearm.dynamics.mujoco_dynamics import MuJoCoDynamics

__all__ = [
    "CoulombViscousFriction",
    "MuJoCoDynamics",
]

import importlib.util as _importlib_util

if _importlib_util.find_spec("pinocchio") is not None:
    from adaptivearm.dynamics.pinocchio_dynamics import PinocchioDynamics  # noqa: F401

    __all__.append("PinocchioDynamics")
