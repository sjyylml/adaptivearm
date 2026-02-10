"""Base class for force/torque observers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptivearm.core.robot_state import RobotState
from adaptivearm.core.types import ObserverOutput


class BaseObserver(ABC):
    """Abstract base for all force/torque observers."""

    @abstractmethod
    def reset(self) -> None:
        """Reset observer internal state."""

    @abstractmethod
    def update(self, state: RobotState) -> ObserverOutput:
        """Process one timestep and return estimated external forces.

        Args:
            state: Current robot state snapshot.

        Returns:
            Observer output with estimated external torques/wrench.
        """
