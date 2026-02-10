"""YAM robot arm adapter using the i2rt SDK.

Implements the RobotInterface protocol for the i2rt YAM 6-DOF arm.
The YAM uses DM-series motors communicating over CAN bus, with an
MIT-style on-board PD + feedforward torque controller.

The i2rt SDK's ``MotorChainRobot`` runs a background control loop at
~250 Hz that:

1. Reads motor feedback (position, velocity, effort/torque).
2. Computes gravity compensation via MuJoCo inverse dynamics.
3. Sends ``kp * (pos_d - pos) + kd * (vel_d - vel) + tau_ff`` to each motor.

``YAMAdapter`` wraps this in the OpenForce ``RobotInterface`` protocol,
enabling GMO/EKF force estimation, impedance control, and all other
framework algorithms to run directly on the real hardware.

Requires: pip install -e /path/to/i2rt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from openforce.core.robot_state import RobotState

try:
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.motor_chain_robot import MotorChainRobot
    from i2rt.robots.utils import GripperType

    _HAS_I2RT = True
except ImportError:
    _HAS_I2RT = False


def _require_i2rt() -> None:
    if not _HAS_I2RT:
        raise ImportError(
            "i2rt SDK is required for YAMAdapter. "
            "Install it with: pip install -e /path/to/i2rt"
        )


@dataclass
class YAMConfig:
    """Configuration for YAM robot connection.

    Attributes:
        channel: CAN bus channel name (e.g. "can0").
        gripper_type: Gripper variant. Use "no_gripper" for force estimation
            experiments (avoids extra DOF complexity).
        zero_gravity_mode: If True, the i2rt SDK applies only gravity
            compensation (no PD position tracking). This is the recommended
            mode for force sensing — the arm is compliant and free-floating.
        n_joints: Number of arm joints (always 6 for YAM, excludes gripper).
        dt: Nominal control timestep. The i2rt background loop runs at ~250 Hz
            (4 ms), but the user-facing read/write rate can be different.
        torque_scale: Scale factor applied to feedforward torques before
            sending.  Useful for conservative testing (e.g. 0.5 = half torque).
        max_torque: Per-joint absolute torque clamp (Nm). Joints 1-3 use
            DM4340 (max 28 Nm), joints 4-6 use DM4310 (max 10 Nm).
    """

    channel: str = "can0"
    gripper_type: str = "no_gripper"
    zero_gravity_mode: bool = True
    n_joints: int = 6
    dt: float = 0.004  # 250 Hz
    torque_scale: float = 1.0
    max_torque: NDArray[np.floating] = field(
        default_factory=lambda: np.array([28.0, 28.0, 28.0, 10.0, 10.0, 10.0])
    )


class YAMAdapter:
    """Robot interface adapter for the i2rt YAM 6-DOF arm.

    Implements the ``RobotInterface`` protocol.  The i2rt SDK handles
    gravity compensation internally, so ``send_torque()`` injects
    *additional* feedforward torques on top of gravity comp.

    Three usage patterns:

    1. **Zero-gravity mode** (default, recommended for force sensing)::

           with YAMAdapter() as yam:
               state = yam.get_state()
               # arm is free-floating with gravity comp

    2. **Position hold + torque overlay**::

           cfg = YAMConfig(zero_gravity_mode=False)
           with YAMAdapter(cfg) as yam:
               yam.reset(q0)
               yam.send_torque(tau)  # added on top of PD + gravity

    3. **Custom PD + torque**::

           with YAMAdapter() as yam:
               yam.send_joint_command(pos=q, vel=qd, kp=kp, kd=kd)

    Args:
        config: Connection and control parameters.
    """

    def __init__(self, config: YAMConfig | None = None) -> None:
        _require_i2rt()

        if config is None:
            config = YAMConfig()
        self._config = config
        self._n = config.n_joints
        self._dt = config.dt
        self._robot: MotorChainRobot | None = None
        self._connected = False
        self._last_time: float = 0.0

    # -- RobotInterface protocol ----------------------------------------

    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def robot(self) -> MotorChainRobot:
        """Access the underlying i2rt MotorChainRobot (for advanced use)."""
        if self._robot is None:
            raise RuntimeError("YAMAdapter is not connected. Call connect() first.")
        return self._robot

    # -- Connection lifecycle -------------------------------------------

    def connect(self) -> None:
        """Establish connection to the YAM arm via CAN bus.

        This powers on all motors, starts the background control loop,
        and performs initial joint-limit safety checks.
        """
        _require_i2rt()
        gripper = GripperType.from_string_name(self._config.gripper_type)
        self._robot = get_yam_robot(
            channel=self._config.channel,
            gripper_type=gripper,
            zero_gravity_mode=self._config.zero_gravity_mode,
        )
        self._connected = True
        self._last_time = time.time()

    def disconnect(self) -> None:
        """Safely shut down the arm (zero torques, close CAN)."""
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._connected = False

    def __enter__(self) -> YAMAdapter:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.disconnect()

    # -- State reading --------------------------------------------------

    def get_state(self) -> RobotState:
        """Read current arm state from motor feedback.

        Returns joint positions, velocities, and motor torques for the
        6 arm joints (gripper excluded).
        """
        if not self._connected or self._robot is None:
            raise RuntimeError("YAMAdapter is not connected. Call connect() first.")

        obs = self._robot.get_observations()
        now = time.time()
        self._last_time = now

        q = np.asarray(obs["joint_pos"][:self._n], dtype=np.float64)
        qd = np.asarray(obs["joint_vel"][:self._n], dtype=np.float64)
        eff = np.asarray(obs["joint_eff"][:self._n], dtype=np.float64)

        return RobotState(
            q=q,
            qd=qd,
            tau_motor=eff,
            timestamp=now,
        )

    # -- Torque command -------------------------------------------------

    def send_torque(self, tau: NDArray[np.floating]) -> None:
        """Send feedforward torque command (on top of gravity comp).

        The i2rt SDK's background loop adds this to gravity compensation
        automatically.  In zero-gravity mode (kp=kd=0), this is the only
        torque applied beyond gravity comp.

        Args:
            tau: Desired additional joint torques, shape (n_joints,).
        """
        if not self._connected or self._robot is None:
            raise RuntimeError("YAMAdapter is not connected. Call connect() first.")

        tau = np.asarray(tau, dtype=np.float64) * self._config.torque_scale
        tau = np.clip(tau, -self._config.max_torque, self._config.max_torque)

        # In zero-gravity mode, set kp=kd=0 so only feedforward torque acts.
        # In position mode, this adds tau on top of existing PD + gravity.
        n = self._n
        gripper_dof = self._robot.num_dofs() - n

        # Build full command including gripper zeros
        pos = np.zeros(n + gripper_dof)
        vel = np.zeros(n + gripper_dof)
        kp = np.zeros(n + gripper_dof)
        kd = np.zeros(n + gripper_dof)

        # Use command_joint_state which sets the internal _commands
        # The MotorChainRobot.update() loop will add gravity comp + our torques
        self._robot.command_joint_state({
            "pos": pos,
            "vel": vel,
            "kp": kp,
            "kd": kd,
        })

        # Directly inject feedforward torques into the command buffer
        with self._robot._command_lock:
            self._robot._commands.torques[:n] = tau

    # -- Position / velocity commands -----------------------------------

    def send_position(self, q: NDArray[np.floating]) -> None:
        """Send position command using default PD gains + gravity comp.

        Args:
            q: Target joint positions, shape (n_joints,).
        """
        if not self._connected or self._robot is None:
            raise RuntimeError("YAMAdapter is not connected. Call connect() first.")

        q_full = np.asarray(q, dtype=np.float64)
        gripper_dof = self._robot.num_dofs() - self._n
        if gripper_dof > 0:
            q_full = np.concatenate([q_full, np.zeros(gripper_dof)])
        self._robot.command_joint_pos(q_full)

    def send_joint_command(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating] | None = None,
        kp: NDArray[np.floating] | None = None,
        kd: NDArray[np.floating] | None = None,
    ) -> None:
        """Send custom PD + feedforward command.

        This gives full control over the MIT-style motor controller:
        ``tau = kp * (pos - q) + kd * (vel - qd) + gravity_comp``

        Args:
            pos: Target positions, shape (n_joints,).
            vel: Target velocities (default: zeros).
            kp: Position gains (default: SDK defaults [80,80,80,40,10,10]).
            kd: Velocity gains (default: SDK defaults [5,5,5,1.5,1.5,1.5]).
        """
        if not self._connected or self._robot is None:
            raise RuntimeError("YAMAdapter is not connected. Call connect() first.")

        n = self._n
        cmd: dict[str, NDArray[np.floating]] = {
            "pos": np.asarray(pos, dtype=np.float64),
        }
        if vel is not None:
            cmd["vel"] = np.asarray(vel, dtype=np.float64)
        else:
            cmd["vel"] = np.zeros(n)

        if kp is not None:
            cmd["kp"] = np.asarray(kp, dtype=np.float64)
        if kd is not None:
            cmd["kd"] = np.asarray(kd, dtype=np.float64)

        # Pad for gripper if present
        gripper_dof = self._robot.num_dofs() - n
        if gripper_dof > 0:
            cmd["pos"] = np.concatenate([cmd["pos"], np.zeros(gripper_dof)])
            cmd["vel"] = np.concatenate([cmd["vel"], np.zeros(gripper_dof)])
            if "kp" in cmd:
                cmd["kp"] = np.concatenate([cmd["kp"], np.zeros(gripper_dof)])
            if "kd" in cmd:
                cmd["kd"] = np.concatenate([cmd["kd"], np.zeros(gripper_dof)])

        self._robot.command_joint_state(cmd)

    # -- Reset ----------------------------------------------------------

    def reset(self, q0: NDArray[np.floating] | None = None) -> RobotState:
        """Move arm to a configuration and return the resulting state.

        In zero-gravity mode with no ``q0``, simply returns the current
        state (the arm stays where it is).

        Args:
            q0: Target configuration (radians). If None, reads current.

        Returns:
            Robot state after reaching (or reading) the target.
        """
        if not self._connected or self._robot is None:
            raise RuntimeError("YAMAdapter is not connected. Call connect() first.")

        if q0 is not None:
            q_target = np.asarray(q0, dtype=np.float64)
            gripper_dof = self._robot.num_dofs() - self._n
            if gripper_dof > 0:
                q_target = np.concatenate([q_target, np.zeros(gripper_dof)])
            self._robot.move_joints(q_target, time_interval_s=3.0)

        return self.get_state()

    # -- Convenience ----------------------------------------------------

    def get_temperatures(self) -> dict[str, NDArray[np.floating]]:
        """Read motor temperatures (MOS and rotor) for health monitoring.

        Returns:
            Dict with "temp_mos" and "temp_rotor" arrays, shape (n_joints,).
        """
        if not self._connected or self._robot is None:
            raise RuntimeError("YAMAdapter is not connected. Call connect() first.")

        with self._robot._state_lock:
            js = self._robot._joint_state
            return {
                "temp_mos": js.temp_mos[:self._n].copy(),
                "temp_rotor": js.temp_rotor[:self._n].copy(),
            }
