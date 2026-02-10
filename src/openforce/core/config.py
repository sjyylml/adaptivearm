"""Configuration dataclass for OpenForce."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class OpenForceConfig:
    """Global configuration for an OpenForce session.

    Attributes:
        n_joints: Number of robot joints.
        dt: Control loop timestep in seconds.
        observer_gains: Diagonal gains for momentum observer, shape (n_joints,).
        friction_coulomb: Coulomb friction coefficients per joint.
        friction_viscous: Viscous friction coefficients per joint.
        lowpass_cutoff: Low-pass filter cutoff frequency in Hz (0 = disabled).
    """

    n_joints: int = 6
    dt: float = 0.002
    observer_gains: NDArray[np.floating] | None = field(default=None)
    friction_coulomb: NDArray[np.floating] | None = field(default=None)
    friction_viscous: NDArray[np.floating] | None = field(default=None)
    lowpass_cutoff: float = 0.0

    def __post_init__(self) -> None:
        n = self.n_joints
        if self.observer_gains is None:
            self.observer_gains = np.full(n, 10.0)
        if self.friction_coulomb is None:
            self.friction_coulomb = np.zeros(n)
        if self.friction_viscous is None:
            self.friction_viscous = np.zeros(n)
