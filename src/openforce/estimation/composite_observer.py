"""Composite observer for multi-observer fusion.

Combines outputs from multiple force/torque observers using configurable
fusion strategies (weighted average, max-norm selection, min-norm selection,
or custom callable).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from openforce.core.robot_state import RobotState
from openforce.core.types import ObserverOutput
from openforce.estimation.base_observer import BaseObserver


class FusionStrategy(Enum):
    """Built-in fusion strategies for combining observer outputs."""

    WEIGHTED_AVERAGE = "weighted_average"
    MAX_NORM = "max_norm"
    MIN_NORM = "min_norm"


def _fuse_weighted_average(
    outputs: list[ObserverOutput],
    weights: NDArray[np.floating],
) -> ObserverOutput:
    """Weighted average of observer outputs."""
    w = weights / weights.sum()
    tau_fused = sum(w[i] * outputs[i].tau_ext for i in range(len(outputs)))
    tau_fused = np.asarray(tau_fused, dtype=np.float64)

    # Fuse wrench if all observers provide it
    wrench_fused = None
    if all(o.wrench_ext is not None for o in outputs):
        wrench_fused = sum(
            w[i] * outputs[i].wrench_ext for i in range(len(outputs))  # type: ignore[operator]
        )
        wrench_fused = np.asarray(wrench_fused, dtype=np.float64)

    return ObserverOutput(
        tau_ext=tau_fused,
        wrench_ext=wrench_fused,
        timestamp=outputs[0].timestamp,
    )


def _fuse_max_norm(outputs: list[ObserverOutput]) -> ObserverOutput:
    """Select the observer output with the largest tau_ext norm."""
    norms = [float(np.linalg.norm(o.tau_ext)) for o in outputs]
    idx = int(np.argmax(norms))
    wrench = outputs[idx].wrench_ext
    return ObserverOutput(
        tau_ext=outputs[idx].tau_ext.copy(),
        wrench_ext=wrench.copy() if wrench is not None else None,
        timestamp=outputs[idx].timestamp,
    )


def _fuse_min_norm(outputs: list[ObserverOutput]) -> ObserverOutput:
    """Select the observer output with the smallest tau_ext norm."""
    norms = [float(np.linalg.norm(o.tau_ext)) for o in outputs]
    idx = int(np.argmin(norms))
    wrench = outputs[idx].wrench_ext
    return ObserverOutput(
        tau_ext=outputs[idx].tau_ext.copy(),
        wrench_ext=wrench.copy() if wrench is not None else None,
        timestamp=outputs[idx].timestamp,
    )


class CompositeObserver(BaseObserver):
    """Observer that fuses outputs from multiple sub-observers.

    Supports built-in fusion strategies (weighted average, max/min norm selection)
    as well as custom callable fusion functions.

    Supports nesting: a CompositeObserver can contain other CompositeObservers.

    Args:
        observers: Sequence of BaseObserver instances to fuse.
        strategy: Fusion strategy enum or callable.
            If callable, signature: (list[ObserverOutput]) -> ObserverOutput
        weights: Per-observer weights for WEIGHTED_AVERAGE strategy.
            Defaults to equal weights.
    """

    def __init__(
        self,
        observers: Sequence[BaseObserver],
        strategy: (
            FusionStrategy | Callable[[list[ObserverOutput]], ObserverOutput]
        ) = FusionStrategy.WEIGHTED_AVERAGE,
        weights: NDArray[np.floating] | None = None,
    ) -> None:
        if len(observers) < 1:
            raise ValueError("CompositeObserver requires at least one sub-observer")

        self._observers = list(observers)
        self._strategy = strategy
        self._n_obs = len(observers)

        if weights is not None:
            self._weights = np.asarray(weights, dtype=np.float64)
        else:
            self._weights = np.ones(self._n_obs, dtype=np.float64)

    @property
    def observers(self) -> list[BaseObserver]:
        """Access the list of sub-observers."""
        return self._observers

    def reset(self) -> None:
        """Reset all sub-observers."""
        for obs in self._observers:
            obs.reset()

    def update(self, state: RobotState) -> ObserverOutput:
        """Run all sub-observers and fuse their outputs.

        Args:
            state: Current robot state.

        Returns:
            Fused ObserverOutput.
        """
        outputs = [obs.update(state) for obs in self._observers]
        return self._fuse(outputs)

    def _fuse(self, outputs: list[ObserverOutput]) -> ObserverOutput:
        """Apply the fusion strategy to observer outputs."""
        if callable(self._strategy) and not isinstance(self._strategy, FusionStrategy):
            return self._strategy(outputs)

        if self._strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return _fuse_weighted_average(outputs, self._weights)
        elif self._strategy == FusionStrategy.MAX_NORM:
            return _fuse_max_norm(outputs)
        elif self._strategy == FusionStrategy.MIN_NORM:
            return _fuse_min_norm(outputs)
        else:
            raise ValueError(f"Unknown fusion strategy: {self._strategy}")
