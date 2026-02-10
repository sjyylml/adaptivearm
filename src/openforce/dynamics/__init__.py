"""Dynamics computation backends."""

from openforce.dynamics.friction_models import CoulombViscousFriction
from openforce.dynamics.mujoco_dynamics import MuJoCoDynamics

__all__ = [
    "CoulombViscousFriction",
    "MuJoCoDynamics",
]

import importlib.util as _importlib_util

if _importlib_util.find_spec("pinocchio") is not None:
    from openforce.dynamics.pinocchio_dynamics import PinocchioDynamics  # noqa: F401

    __all__.append("PinocchioDynamics")
