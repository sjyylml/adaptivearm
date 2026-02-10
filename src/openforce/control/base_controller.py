"""Base class for controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openforce.core.robot_state import RobotState
from openforce.core.types import ControlOutput, ObserverOutput


class BaseController(ABC):
    """Abstract base for all compliant controllers."""

    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state."""

    @abstractmethod
    def compute(
        self,
        state: RobotState,
        observer_output: ObserverOutput | None = None,
    ) -> ControlOutput:
        """Compute control torque.

        Args:
            state: Current robot state.
            observer_output: Optional force estimate from observer.

        Returns:
            ControlOutput with commanded joint torques.
        """
