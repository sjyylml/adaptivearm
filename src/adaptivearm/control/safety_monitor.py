"""Safety monitor — wraps any controller with runtime safety checks.

Enforces joint position limits, velocity limits, torque limits,
and collision-triggered reactions. Acts as a transparent wrapper:
normal operation passes torques through; violation triggers a safe response.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from adaptivearm.core.types import ControlOutput, ObserverOutput
from adaptivearm.estimation.collision_detector import CollisionDetector, CollisionEvent


class SafetyState(Enum):
    """Current safety system state."""

    NORMAL = auto()
    COLLISION_DETECTED = auto()
    LIMIT_VIOLATED = auto()
    STOPPED = auto()


@dataclass
class SafetyLimits:
    """Joint-level safety limits.

    Attributes:
        q_min: Lower joint position limits (rad), shape (n,).
        q_max: Upper joint position limits (rad), shape (n,).
        qd_max: Maximum joint velocity (rad/s), shape (n,).
        tau_max: Maximum torque command (Nm), shape (n,).
        q_margin: Position margin before limit (rad). Controller softens near limits.
    """

    q_min: NDArray[np.floating]
    q_max: NDArray[np.floating]
    qd_max: NDArray[np.floating]
    tau_max: NDArray[np.floating]
    q_margin: float = 0.1


class SafetyMonitor:
    """Runtime safety wrapper for any controller output.

    Checks:
    1. Torque clipping — clamps |τ| to tau_max
    2. Velocity limiting — reduces torque if near velocity limit
    3. Position limiting — applies repulsive torque near joint limits
    4. Collision reaction — zeros torque or retracts on collision

    Args:
        n_joints: Number of joints.
        limits: Joint safety limits.
        collision_detector: Optional collision detector instance.
    """

    def __init__(
        self,
        n_joints: int,
        limits: SafetyLimits | None = None,
        collision_detector: CollisionDetector | None = None,
    ) -> None:
        self._n = n_joints
        self._collision_detector = collision_detector
        self._state = SafetyState.NORMAL
        self._last_collision: CollisionEvent | None = None

        if limits is None:
            limits = SafetyLimits(
                q_min=np.full(n_joints, -3.14),
                q_max=np.full(n_joints, 3.14),
                qd_max=np.full(n_joints, 3.0),
                tau_max=np.full(n_joints, 50.0),
            )
        self._limits = limits

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def last_collision(self) -> CollisionEvent | None:
        return self._last_collision

    def reset(self) -> None:
        """Reset safety monitor to normal state."""
        self._state = SafetyState.NORMAL
        self._last_collision = None
        if self._collision_detector is not None:
            self._collision_detector.reset()

    def filter(
        self,
        tau_cmd: NDArray[np.floating],
        q: NDArray[np.floating],
        qd: NDArray[np.floating],
        observer_output: ObserverOutput | None = None,
    ) -> ControlOutput:
        """Apply safety filtering to a torque command.

        Args:
            tau_cmd: Raw torque command from controller, shape (n,).
            q: Current joint positions, shape (n,).
            qd: Current joint velocities, shape (n,).
            observer_output: Optional observer output for collision detection.

        Returns:
            ControlOutput with safe (filtered) torques.
        """
        tau = np.asarray(tau_cmd, dtype=np.float64).copy()
        lim = self._limits
        info: dict[str, object] = {"safety_state": self._state.name}

        # 1. Collision detection
        if self._collision_detector is not None and observer_output is not None:
            event = self._collision_detector.check(observer_output)
            if event.detected:
                self._state = SafetyState.COLLISION_DETECTED
                self._last_collision = event
                info["collision"] = True
                info["collision_severity"] = event.severity
                # Collision reaction: zero torque (go limp)
                return ControlOutput(tau_cmd=np.zeros(self._n), info=info)

        # 2. Position limit avoidance (virtual wall)
        for j in range(self._n):
            margin = lim.q_margin
            if q[j] < lim.q_min[j] + margin:
                # Push away from lower limit
                penetration = (lim.q_min[j] + margin) - q[j]
                tau[j] += 200.0 * penetration  # repulsive spring
                tau[j] = max(tau[j], 0.0)      # only push outward
            elif q[j] > lim.q_max[j] - margin:
                penetration = q[j] - (lim.q_max[j] - margin)
                tau[j] -= 200.0 * penetration
                tau[j] = min(tau[j], 0.0)

        # 3. Velocity limiting — scale down torque in velocity direction
        for j in range(self._n):
            if abs(qd[j]) > lim.qd_max[j] * 0.9:
                scale = max(0.0, 1.0 - (abs(qd[j]) - lim.qd_max[j] * 0.9) / (lim.qd_max[j] * 0.1))
                # Only limit torque that accelerates in the same direction
                if qd[j] * tau[j] > 0:
                    tau[j] *= scale

        # 4. Torque clipping
        tau = np.clip(tau, -lim.tau_max, lim.tau_max)

        # Update state
        if self._state == SafetyState.COLLISION_DETECTED:
            if self._collision_detector is not None and not self._collision_detector.in_collision:
                self._state = SafetyState.NORMAL
        else:
            self._state = SafetyState.NORMAL

        info["safety_state"] = self._state.name
        return ControlOutput(tau_cmd=tau, info=info)
