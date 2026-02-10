"""Core data types for AdaptiveArm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ObserverOutput:
    """Output from a force/torque observer.

    Attributes:
        tau_ext: Estimated external joint torques, shape (n_joints,).
        wrench_ext: Estimated external Cartesian wrench [fx,fy,fz,tx,ty,tz], shape (6,).
            None if Jacobian is not available.
        timestamp: Time at which the estimate was produced.
        residual_raw: Raw residual before filtering, shape (n_joints,).
    """

    tau_ext: NDArray[np.floating]
    wrench_ext: NDArray[np.floating] | None = None
    timestamp: float = 0.0
    residual_raw: NDArray[np.floating] | None = None


@dataclass
class ControlOutput:
    """Output from a controller.

    Attributes:
        tau_cmd: Commanded joint torques, shape (n_joints,).
        info: Optional dictionary with controller-specific debug info.
    """

    tau_cmd: NDArray[np.floating]
    info: dict[str, object] = field(default_factory=dict)
