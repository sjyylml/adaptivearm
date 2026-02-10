"""Robot state container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RobotState:
    """Snapshot of the robot's kinematic and dynamic state.

    Attributes:
        q: Joint positions, shape (n_joints,).
        qd: Joint velocities, shape (n_joints,).
        tau_motor: Motor torques (commanded or measured), shape (n_joints,).
        timestamp: Acquisition time in seconds.
        qdd: Joint accelerations if available, shape (n_joints,).
        jacobian: End-effector Jacobian if available, shape (6, n_joints).
    """

    q: NDArray[np.floating]
    qd: NDArray[np.floating]
    tau_motor: NDArray[np.floating]
    timestamp: float = 0.0
    qdd: NDArray[np.floating] | None = None
    jacobian: NDArray[np.floating] | None = None

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return len(self.q)

    def validate(self) -> None:
        """Check dimensional consistency."""
        n = self.n_joints
        if self.qd.shape != (n,):
            raise ValueError(f"qd shape {self.qd.shape} != expected ({n},)")
        if self.tau_motor.shape != (n,):
            raise ValueError(f"tau_motor shape {self.tau_motor.shape} != expected ({n},)")
        if self.qdd is not None and self.qdd.shape != (n,):
            raise ValueError(f"qdd shape {self.qdd.shape} != expected ({n},)")
        if self.jacobian is not None and self.jacobian.shape != (6, n):
            raise ValueError(f"jacobian shape {self.jacobian.shape} != expected (6, {n})")
