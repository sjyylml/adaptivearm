"""Universal Robots adapter using RTDE (Real-Time Data Exchange).

Implements the RobotInterface protocol for UR e-Series robots (UR3e/5e/10e/16e).
Since UR does not support direct torque control, torques are converted to
position increments via an internal admittance model and sent via servoJ.

Requires: pip install ur_rtde
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from adaptivearm.core.robot_state import RobotState

try:
    import rtde_control  # type: ignore[import-untyped]
    import rtde_receive  # type: ignore[import-untyped]

    _HAS_RTDE = True
except ImportError:
    _HAS_RTDE = False


def _require_rtde() -> None:
    if not _HAS_RTDE:
        raise ImportError(
            "ur_rtde is required for URAdapter. "
            "Install it with: pip install ur_rtde"
        )


@dataclass
class URConfig:
    """Configuration for UR robot connection.

    Attributes:
        ip: Robot IP address.
        dt: Control loop timestep in seconds.
        servo_gain: servoJ gain parameter (0-2000). Higher = stiffer tracking.
        servo_lookahead: servoJ lookahead time in seconds (0.03-0.2).
        torque_stiffness: Virtual stiffness for torque-to-position conversion (Nm/rad).
            Used when send_torque() converts torques to position increments.
        n_joints: Number of joints (always 6 for UR).
    """

    ip: str = "192.168.1.1"
    dt: float = 0.002  # 500 Hz
    servo_gain: float = 300.0
    servo_lookahead: float = 0.1
    torque_stiffness: float = 500.0
    n_joints: int = 6


class URAdapter:
    """Robot interface adapter for Universal Robots via RTDE.

    Implements the RobotInterface protocol. Since UR robots don't support
    direct joint torque control, ``send_torque()`` uses an internal admittance
    model to convert desired torques into position deltas sent via servoJ.

    Usage:
        with URAdapter(URConfig(ip="192.168.1.1")) as ur:
            state = ur.get_state()
            ur.send_torque(tau)

    Args:
        config: UR connection and control parameters.
    """

    def __init__(self, config: URConfig | None = None) -> None:
        _require_rtde()

        if config is None:
            config = URConfig()
        self._config = config
        self._n = config.n_joints
        self._dt = config.dt

        self._rtde_c: rtde_control.RTDEControlInterface | None = None  # type: ignore[name-defined]
        self._rtde_r: rtde_receive.RTDEReceiveInterface | None = None  # type: ignore[name-defined]
        self._connected = False

        # Internal state for admittance-based torque control
        self._q_target: NDArray[np.floating] = np.zeros(self._n)

    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Establish RTDE connection to the robot."""
        _require_rtde()
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self._config.ip)
        self._rtde_c = rtde_control.RTDEControlInterface(self._config.ip)
        self._connected = True

        # Initialize target to current position
        q = self._rtde_r.getActualQ()
        self._q_target = np.array(q, dtype=np.float64)

    def disconnect(self) -> None:
        """Close RTDE connection."""
        if self._rtde_c is not None:
            self._rtde_c.stopScript()
            self._rtde_c = None
        self._rtde_r = None
        self._connected = False

    def __enter__(self) -> URAdapter:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.disconnect()

    def get_state(self) -> RobotState:
        """Read current robot state via RTDE.

        Returns joint positions, velocities, and motor currents (converted to torques).
        """
        if not self._connected or self._rtde_r is None:
            raise RuntimeError("URAdapter is not connected. Call connect() first.")

        q = np.array(self._rtde_r.getActualQ(), dtype=np.float64)
        qd = np.array(self._rtde_r.getActualQd(), dtype=np.float64)
        # UR provides motor currents; approximate torques
        current = np.array(self._rtde_r.getActualCurrent(), dtype=np.float64)
        timestamp = float(self._rtde_r.getTimestamp())

        return RobotState(
            q=q,
            qd=qd,
            tau_motor=current,  # In practice, multiply by torque constant
            timestamp=timestamp,
        )

    def send_torque(self, tau: NDArray[np.floating]) -> None:
        """Send torque command via admittance conversion + servoJ.

        Since UR doesn't support direct torque control, we convert the desired
        torque into a position increment: delta_q = tau / K_virtual, then
        update the servoJ target.

        Args:
            tau: Desired joint torques, shape (n_joints,).
        """
        if not self._connected or self._rtde_c is None:
            raise RuntimeError("URAdapter is not connected. Call connect() first.")

        # Admittance conversion: torque → position delta
        delta_q = np.asarray(tau, dtype=np.float64) / self._config.torque_stiffness
        self._q_target += delta_q * self._dt

        self._rtde_c.servoJ(
            self._q_target.tolist(),
            0.0,  # velocity (not used in servoJ mode)
            0.0,  # acceleration (not used in servoJ mode)
            self._dt,
            self._config.servo_lookahead,
            self._config.servo_gain,
        )

    def send_position(self, q: NDArray[np.floating]) -> None:
        """Send direct position command via servoJ.

        Args:
            q: Desired joint positions, shape (n_joints,).
        """
        if not self._connected or self._rtde_c is None:
            raise RuntimeError("URAdapter is not connected. Call connect() first.")

        self._q_target = np.asarray(q, dtype=np.float64).copy()
        self._rtde_c.servoJ(
            self._q_target.tolist(),
            0.0,
            0.0,
            self._dt,
            self._config.servo_lookahead,
            self._config.servo_gain,
        )

    def send_velocity(self, qd: NDArray[np.floating]) -> None:
        """Send direct velocity command via speedJ.

        Args:
            qd: Desired joint velocities, shape (n_joints,).
        """
        if not self._connected or self._rtde_c is None:
            raise RuntimeError("URAdapter is not connected. Call connect() first.")

        self._rtde_c.speedJ(
            np.asarray(qd, dtype=np.float64).tolist(),
            10.0,  # acceleration limit
            self._dt,
        )

    def reset(self, q0: NDArray[np.floating] | None = None) -> RobotState:
        """Move robot to initial configuration via moveJ.

        Args:
            q0: Target configuration. If None, uses current position.

        Returns:
            Robot state after reaching the target.
        """
        if not self._connected or self._rtde_c is None or self._rtde_r is None:
            raise RuntimeError("URAdapter is not connected. Call connect() first.")

        if q0 is None:
            q0_list = list(self._rtde_r.getActualQ())
        else:
            q0_list = np.asarray(q0, dtype=np.float64).tolist()

        self._rtde_c.moveJ(q0_list, 1.0, 1.0)  # speed=1 rad/s, accel=1 rad/s²
        self._q_target = np.array(q0_list, dtype=np.float64)

        return self.get_state()
