"""Protocol definitions for robot interface and dynamics model."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from adaptivearm.core.robot_state import RobotState


class RobotInterface(Protocol):
    """Adapter protocol for connecting to any robot arm."""

    @property
    def n_joints(self) -> int:
        """Number of controllable joints."""
        ...

    @property
    def dt(self) -> float:
        """Control timestep in seconds."""
        ...

    def get_state(self) -> RobotState:
        """Read current robot state."""
        ...

    def send_torque(self, tau: NDArray[np.floating]) -> None:
        """Send joint torque command to the robot."""
        ...

    def reset(self, q0: NDArray[np.floating] | None = None) -> RobotState:
        """Reset robot to initial or given configuration. Returns resulting state."""
        ...


class DynamicsModel(Protocol):
    """Protocol for rigid-body dynamics computation."""

    def mass_matrix(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute joint-space inertia matrix M(q), shape (n, n)."""
        ...

    def coriolis_vector(
        self, q: NDArray[np.floating], qd: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute Coriolis/centrifugal vector C(q, qd)*qd, shape (n,)."""
        ...

    def gravity_vector(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute gravity torque vector g(q), shape (n,)."""
        ...


class ExtendedDynamicsModel(Protocol):
    """Extended dynamics protocol including passive torques and joint count.

    MuJoCoDynamics already satisfies this protocol structurally.
    """

    @property
    def n_joints(self) -> int:
        """Number of degrees of freedom."""
        ...

    def mass_matrix(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute joint-space inertia matrix M(q), shape (n, n)."""
        ...

    def coriolis_vector(
        self, q: NDArray[np.floating], qd: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute Coriolis/centrifugal vector C(q, qd)*qd, shape (n,)."""
        ...

    def gravity_vector(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute gravity torque vector g(q), shape (n,)."""
        ...

    def passive_torque(
        self, qd: NDArray[np.floating], q: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute passive joint torques (damping, etc.), shape (n,)."""
        ...
