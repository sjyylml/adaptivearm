"""Extended Kalman Filter (EKF) observer for external force estimation.

Uses an augmented state vector x = [q, qd, tau_ext] (3n dimensions) to
jointly estimate joint states and external torques from encoder measurements.

Reference:
    Magrini, E., De Luca, A. (2016). Estimation of contact forces using a
    virtual force sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from openforce.core.interfaces import ExtendedDynamicsModel
from openforce.core.robot_state import RobotState
from openforce.core.types import ObserverOutput
from openforce.estimation.base_observer import BaseObserver
from openforce.utils.math_utils import pseudoinverse


@dataclass
class EKFParams:
    """Parameters for the EKF force observer.

    Attributes:
        Q_q: Process noise for position states (diagonal), shape (n,).
        Q_qd: Process noise for velocity states (diagonal), shape (n,).
        Q_tau: Process noise for external torque states (diagonal), shape (n,).
            Higher values allow faster force tracking but more noise.
        R: Measurement noise (encoder precision), shape (n,).
        P0_diag: Initial covariance diagonal, shape (3n,). If None, uses defaults.
    """

    Q_q: NDArray[np.floating] | None = None
    Q_qd: NDArray[np.floating] | None = None
    Q_tau: NDArray[np.floating] | None = None
    R: NDArray[np.floating] | None = None
    P0_diag: NDArray[np.floating] | None = None


class EKFObserver(BaseObserver):
    """Extended Kalman Filter observer for external force estimation.

    Augmented state: x = [q, qd, tau_ext]  (3n-dimensional)

    Prediction model:
        q_new  = q + dt * qd
        qd_new = qd + dt * M^{-1}(tau_motor + tau_passive - C*qd - g + tau_ext)
        tau_ext_new = tau_ext  (random walk)

    Measurement model:
        z = q  (encoder readings)

    Args:
        dynamics: Dynamics model providing M(q), C(q,qd)*qd, g(q), passive_torque().
            Must satisfy the ExtendedDynamicsModel protocol.
        n_joints: Number of robot joints.
        dt: Control timestep in seconds.
        params: EKF tuning parameters. Defaults are suitable for typical 6-DOF arms.
    """

    def __init__(
        self,
        dynamics: ExtendedDynamicsModel,
        n_joints: int,
        dt: float,
        params: EKFParams | None = None,
    ) -> None:
        self._dynamics = dynamics
        self._n = n_joints
        self._dt = dt
        self._3n = 3 * n_joints

        if params is None:
            params = EKFParams()
        self._params = params

        n = n_joints

        # Process noise covariance Q (3n x 3n diagonal)
        Q_q = params.Q_q if params.Q_q is not None else np.full(n, 1e-6)
        Q_qd = params.Q_qd if params.Q_qd is not None else np.full(n, 1e-4)
        Q_tau = params.Q_tau if params.Q_tau is not None else np.full(n, 1.0)
        self._Q = np.diag(np.concatenate([Q_q, Q_qd, Q_tau]))

        # Measurement noise covariance R (n x n diagonal)
        R = params.R if params.R is not None else np.full(n, 1e-6)
        self._R = np.diag(R)

        # Measurement matrix H = [I, 0, 0]
        self._H = np.zeros((n, self._3n), dtype=np.float64)
        self._H[:n, :n] = np.eye(n)

        # State and covariance
        self._x = np.zeros(self._3n, dtype=np.float64)
        if params.P0_diag is not None:
            self._P = np.diag(params.P0_diag)
        else:
            P0 = np.concatenate([
                np.full(n, 1e-4),  # q uncertainty
                np.full(n, 1e-2),  # qd uncertainty
                np.full(n, 10.0),  # tau_ext uncertainty (large = uninformative)
            ])
            self._P = np.diag(P0)

        self._initialized = False

    def reset(self) -> None:
        """Reset EKF state and covariance."""
        n = self._n
        self._x[:] = 0.0
        if self._params.P0_diag is not None:
            self._P = np.diag(self._params.P0_diag)
        else:
            P0 = np.concatenate([
                np.full(n, 1e-4),
                np.full(n, 1e-2),
                np.full(n, 10.0),
            ])
            self._P = np.diag(P0)
        self._initialized = False

    def update(self, state: RobotState) -> ObserverOutput:
        """Run one EKF predict-correct cycle.

        Args:
            state: Current robot state with encoder readings and motor torques.

        Returns:
            ObserverOutput with estimated external joint torques.
        """
        n = self._n
        q_meas = state.q
        tau_motor = state.tau_motor

        # Initialize state from first measurement
        if not self._initialized:
            self._x[:n] = q_meas.copy()
            self._x[n : 2 * n] = state.qd.copy()
            self._x[2 * n :] = 0.0
            self._initialized = True
            return ObserverOutput(
                tau_ext=np.zeros(n),
                timestamp=state.timestamp,
                residual_raw=np.zeros(n),
            )

        # === PREDICT ===
        q_est = self._x[:n]
        qd_est = self._x[n : 2 * n]
        tau_ext_est = self._x[2 * n :]

        # Dynamics terms
        M = self._dynamics.mass_matrix(q_est)
        c = self._dynamics.coriolis_vector(q_est, qd_est)
        g = self._dynamics.gravity_vector(q_est)
        tau_passive = self._dynamics.passive_torque(qd_est, q_est)
        # Acceleration: qdd = M^{-1} (tau_motor + tau_passive - C*qd - g + tau_ext)
        M_inv = np.linalg.inv(M)
        qdd = M_inv @ (tau_motor + tau_passive - c - g + tau_ext_est)

        # Forward Euler integration
        x_pred = np.empty(self._3n, dtype=np.float64)
        x_pred[:n] = q_est + self._dt * qd_est
        x_pred[n : 2 * n] = qd_est + self._dt * qdd
        x_pred[2 * n :] = tau_ext_est  # random walk

        # Linearized state transition Jacobian F (simplified)
        F = np.eye(self._3n, dtype=np.float64)
        F[:n, n : 2 * n] = self._dt * np.eye(n)  # dq/dqd
        F[n : 2 * n, 2 * n :] = self._dt * M_inv  # dqd/dtau_ext

        # Predicted covariance
        P_pred = F @ self._P @ F.T + self._Q

        # === CORRECT ===
        z = q_meas
        y = z - self._H @ x_pred  # innovation

        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = x_pred + K @ y

        # Joseph form update for numerical stability: P = (I-KH)P(I-KH)^T + KRK^T
        IKH = np.eye(self._3n) - K @ self._H
        self._P = IKH @ P_pred @ IKH.T + K @ self._R @ K.T

        # Extract estimated external torques
        tau_ext = self._x[2 * n :].copy()

        # Cartesian wrench via Jacobian pseudoinverse
        wrench = None
        if state.jacobian is not None:
            J_pinv = pseudoinverse(state.jacobian)
            wrench = J_pinv.T @ tau_ext

        return ObserverOutput(
            tau_ext=tau_ext,
            wrench_ext=wrench,
            timestamp=state.timestamp,
            residual_raw=tau_ext,
        )
