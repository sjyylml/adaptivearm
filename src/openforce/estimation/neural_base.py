"""Base class for neural network-based force/torque observers.

Provides common infrastructure for PINN and Transformer observers including
training data collection, mode switching, and model persistence.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from openforce.core.robot_state import RobotState
from openforce.estimation.base_observer import BaseObserver

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _require_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for neural observers. "
            "Install it with: pip install torch"
        )


class ObserverMode(Enum):
    """Operating mode for neural observers."""

    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class TrainingData:
    """Container for training data collected from simulation.

    Attributes:
        states_q: List of joint position arrays.
        states_qd: List of joint velocity arrays.
        states_tau_motor: List of motor torque arrays.
        targets_tau_ext: List of ground-truth external torque arrays.
        timestamps: List of timestamps.
    """

    states_q: list[NDArray[np.floating]] = field(default_factory=list)
    states_qd: list[NDArray[np.floating]] = field(default_factory=list)
    states_tau_motor: list[NDArray[np.floating]] = field(default_factory=list)
    targets_tau_ext: list[NDArray[np.floating]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states_q)

    def clear(self) -> None:
        """Clear all collected data."""
        self.states_q.clear()
        self.states_qd.clear()
        self.states_tau_motor.clear()
        self.targets_tau_ext.clear()
        self.timestamps.clear()


class NeuralObserver(BaseObserver):
    """Abstract base for neural network-based force/torque observers.

    Extends BaseObserver with training data collection, mode switching,
    and model persistence capabilities.

    Args:
        n_joints: Number of robot joints.
    """

    def __init__(self, n_joints: int) -> None:
        _require_torch()
        self._n_joints = n_joints
        self._mode = ObserverMode.INFERENCE
        self._trained = False
        self._training_data = TrainingData()

    @property
    def n_joints(self) -> int:
        """Number of robot joints."""
        return self._n_joints

    @property
    def mode(self) -> ObserverMode:
        """Current operating mode."""
        return self._mode

    @property
    def trained(self) -> bool:
        """Whether the model has been trained."""
        return self._trained

    @property
    def training_data(self) -> TrainingData:
        """Access collected training data."""
        return self._training_data

    def set_mode(self, mode: ObserverMode) -> None:
        """Switch between training and inference modes.

        Args:
            mode: Target operating mode.
        """
        self._mode = mode

    def collect(
        self,
        state: RobotState,
        tau_ext_ground_truth: NDArray[np.floating],
    ) -> None:
        """Collect a training sample.

        Args:
            state: Current robot state.
            tau_ext_ground_truth: Ground-truth external torques for this state.
        """
        self._training_data.states_q.append(state.q.copy())
        self._training_data.states_qd.append(state.qd.copy())
        self._training_data.states_tau_motor.append(state.tau_motor.copy())
        self._training_data.targets_tau_ext.append(tau_ext_ground_truth.copy())
        self._training_data.timestamps.append(state.timestamp)

    @abstractmethod
    def train(self, epochs: int = 100, verbose: bool = False) -> dict[str, float]:
        """Train the neural network model.

        Args:
            epochs: Number of training epochs.
            verbose: Whether to print progress.

        Returns:
            Dictionary of final loss values.
        """

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk.

        Args:
            path: File path for the saved model.
        """
        assert _HAS_TORCH
        model = self._get_network()
        torch.save(model.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            path: File path of the saved model.
        """
        assert _HAS_TORCH
        model = self._get_network()
        model.load_state_dict(torch.load(str(path), weights_only=True))
        model.eval()
        self._trained = True

    @abstractmethod
    def _get_network(self) -> Any:
        """Return the underlying PyTorch network module."""

    def reset(self) -> None:
        """Reset observer state. In training mode, clears collected data."""
        if self._mode == ObserverMode.TRAINING:
            self._training_data.clear()
