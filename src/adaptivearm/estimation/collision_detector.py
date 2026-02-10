"""Collision detection based on momentum observer residual.

Uses per-joint thresholds on the GMO residual to detect unexpected
contacts. Supports configurable reaction strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from adaptivearm.core.types import ObserverOutput


class CollisionReaction(Enum):
    """What to do when a collision is detected."""

    NONE = auto()       # Just report, don't act
    STOP = auto()       # Zero torque (let the arm go limp)
    RETRACT = auto()    # Reverse motion along collision direction


@dataclass
class CollisionEvent:
    """Describes a detected collision.

    Attributes:
        detected: Whether a collision was detected.
        joint_mask: Boolean mask of which joints exceeded threshold.
        residual: Observer residual at detection time.
        severity: Max ratio of residual to threshold across joints.
        timestamp: Time of detection.
    """

    detected: bool = False
    joint_mask: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=bool))
    residual: NDArray[np.floating] = field(default_factory=lambda: np.array([], dtype=np.float64))
    severity: float = 0.0
    timestamp: float = 0.0


class CollisionDetector:
    """Threshold-based collision detector using observer residual.

    Compares the absolute value of each joint's observer residual against
    a per-joint threshold. When any joint exceeds its threshold, a
    collision is reported.

    Includes a configurable hold-off time to avoid retriggering.

    Args:
        n_joints: Number of joints.
        thresholds: Per-joint torque thresholds in Nm, shape (n,).
            Lower = more sensitive, but more false positives.
        holdoff_time: Minimum time between collision events (seconds).
        reaction: What to do when collision is detected.
    """

    def __init__(
        self,
        n_joints: int,
        thresholds: NDArray[np.floating] | None = None,
        holdoff_time: float = 0.1,
        reaction: CollisionReaction = CollisionReaction.STOP,
    ) -> None:
        self._n = n_joints
        self._thresholds = (
            np.asarray(thresholds, dtype=np.float64)
            if thresholds is not None
            else np.full(n_joints, 5.0)
        )
        self._holdoff = holdoff_time
        self._reaction = reaction
        self._last_collision_time = -np.inf
        self._in_collision = False

    @property
    def reaction(self) -> CollisionReaction:
        return self._reaction

    @reaction.setter
    def reaction(self, value: CollisionReaction) -> None:
        self._reaction = value

    @property
    def in_collision(self) -> bool:
        return self._in_collision

    def reset(self) -> None:
        """Reset detector state."""
        self._last_collision_time = -np.inf
        self._in_collision = False

    def check(self, observer_output: ObserverOutput) -> CollisionEvent:
        """Check for collision based on observer output.

        Args:
            observer_output: Latest observer estimate.

        Returns:
            CollisionEvent with detection result.
        """
        residual = np.abs(observer_output.tau_ext)
        ratios = residual / self._thresholds
        joint_mask = ratios > 1.0
        severity = float(np.max(ratios))
        t = observer_output.timestamp

        # Check hold-off
        if severity > 1.0 and (t - self._last_collision_time) >= self._holdoff:
            self._in_collision = True
            self._last_collision_time = t
            return CollisionEvent(
                detected=True,
                joint_mask=joint_mask,
                residual=observer_output.tau_ext.copy(),
                severity=severity,
                timestamp=t,
            )

        # Clear collision state if residual drops below threshold
        if severity < 0.5:
            self._in_collision = False

        return CollisionEvent(
            detected=False,
            joint_mask=joint_mask,
            residual=observer_output.tau_ext.copy(),
            severity=severity,
            timestamp=t,
        )
