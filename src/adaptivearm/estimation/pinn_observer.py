"""Physics-Informed Neural Network (PINN) observer for external force estimation.

Combines data-driven learning with physics-based regularization using
the robot's dynamics model. The physics residual loss encourages the network
to respect the equation of motion: M*qdd + C*qd + g = tau_motor + tau_ext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adaptivearm.core.interfaces import DynamicsModel
from adaptivearm.core.robot_state import RobotState
from adaptivearm.core.types import ObserverOutput
from adaptivearm.estimation.neural_base import (
    _HAS_TORCH,
    NeuralObserver,
    _require_torch,
)

if _HAS_TORCH:
    import torch
    import torch.nn as nn


@dataclass
class PINNParams:
    """Configuration for the PINN observer.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        learning_rate: Adam optimizer learning rate.
        physics_weight: Weight for physics residual loss.
        batch_size: Training batch size.
        activation: Activation function name ("tanh" or "relu").
        dropout: Dropout probability (0.0 = disabled).
    """

    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    learning_rate: float = 1e-3
    physics_weight: float = 0.1
    batch_size: int = 64
    activation: str = "tanh"
    dropout: float = 0.0


def _build_pinn_network(
    input_dim: int,
    output_dim: int,
    params: PINNParams,
) -> Any:
    """Build the PINN MLP network.

    Args:
        input_dim: Input feature dimension (3 * n_joints).
        output_dim: Output dimension (n_joints).
        params: Network configuration.

    Returns:
        PyTorch Sequential model.
    """
    assert _HAS_TORCH
    layers: list[nn.Module] = []
    in_features = input_dim

    activation_fn: type[nn.Module] = nn.Tanh if params.activation == "tanh" else nn.ReLU

    for hidden_dim in params.hidden_dims:
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(activation_fn())
        if params.dropout > 0.0:
            layers.append(nn.Dropout(params.dropout))
        in_features = hidden_dim

    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class PINNObserver(NeuralObserver):
    """Physics-Informed Neural Network observer for external force estimation.

    Uses an MLP to predict external torques from joint state, with a physics
    residual loss that enforces consistency with the equation of motion.

    Input: [q, qd, tau_motor] concatenated (3 * n_joints)
    Output: tau_ext prediction (n_joints)

    Args:
        n_joints: Number of robot joints.
        dynamics: Dynamics model for physics residual computation.
        dt: Control timestep in seconds.
        params: PINN configuration parameters.
    """

    def __init__(
        self,
        n_joints: int,
        dynamics: DynamicsModel,
        dt: float,
        params: PINNParams | None = None,
    ) -> None:
        super().__init__(n_joints)
        assert _HAS_TORCH

        self._dynamics = dynamics
        self._dt = dt
        self._params = params or PINNParams()

        input_dim = 3 * n_joints
        self._network = _build_pinn_network(input_dim, n_joints, self._params)
        self._network.eval()

    def _get_network(self) -> Any:
        return self._network

    def update(self, state: RobotState) -> ObserverOutput:
        """Run one observer step.

        In inference mode with a trained model, performs forward pass.
        Otherwise returns zero torques.

        Args:
            state: Current robot state.

        Returns:
            ObserverOutput with estimated external joint torques.
        """
        assert _HAS_TORCH

        if self._trained and self._mode.value == "inference":
            x = np.concatenate([state.q, state.qd, state.tau_motor])
            x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

            self._network.eval()
            with torch.no_grad():
                tau_ext = self._network(x_tensor).squeeze(0).numpy()

            return ObserverOutput(
                tau_ext=tau_ext.astype(np.float64),
                timestamp=state.timestamp,
            )

        return ObserverOutput(
            tau_ext=np.zeros(self._n_joints),
            timestamp=state.timestamp,
        )

    def train(self, epochs: int = 100, verbose: bool = False) -> dict[str, float]:
        """Train the PINN model on collected data.

        Loss = MSE(tau_ext_pred, tau_ext_gt) + physics_weight * physics_loss

        The physics loss measures the equation-of-motion residual:
        ||M*qdd + C*qd + g - tau_motor - tau_ext_pred||^2

        Args:
            epochs: Number of training epochs.
            verbose: Whether to print progress.

        Returns:
            Dictionary with final 'mse_loss', 'physics_loss', 'total_loss'.
        """
        assert _HAS_TORCH
        _require_torch()

        data = self._training_data
        if len(data) < 2:
            raise ValueError("Need at least 2 training samples (for finite differences).")

        # Prepare numpy arrays
        q_all = np.array(data.states_q, dtype=np.float64)
        qd_all = np.array(data.states_qd, dtype=np.float64)
        tau_motor_all = np.array(data.states_tau_motor, dtype=np.float64)
        tau_ext_gt_all = np.array(data.targets_tau_ext, dtype=np.float64)
        timestamps = np.array(data.timestamps, dtype=np.float64)

        # Compute qdd via finite differences (central where possible)
        n_samples = len(q_all)
        qdd_all = np.zeros_like(qd_all)
        for i in range(n_samples):
            if i == 0:
                dt_i = timestamps[1] - timestamps[0] if n_samples > 1 else self._dt
                qdd_all[i] = (qd_all[1] - qd_all[0]) / max(dt_i, 1e-6)
            elif i == n_samples - 1:
                dt_i = timestamps[i] - timestamps[i - 1]
                qdd_all[i] = (qd_all[i] - qd_all[i - 1]) / max(dt_i, 1e-6)
            else:
                dt_i = timestamps[i + 1] - timestamps[i - 1]
                qdd_all[i] = (qd_all[i + 1] - qd_all[i - 1]) / max(dt_i, 1e-6)

        # Precompute physics residual targets (numpy, not through autograd)
        physics_targets = np.zeros((n_samples, self._n_joints), dtype=np.float64)
        for i in range(n_samples):
            M = self._dynamics.mass_matrix(q_all[i])
            c = self._dynamics.coriolis_vector(q_all[i], qd_all[i])
            g = self._dynamics.gravity_vector(q_all[i])
            # EOM: M*qdd + C*qd + g = tau_motor + tau_ext
            # => tau_ext = M*qdd + C*qd + g - tau_motor
            # Physics target: what tau_ext should be per dynamics
            physics_targets[i] = M @ qdd_all[i] + c + g - tau_motor_all[i]

        # Build input features
        x_np = np.concatenate([q_all, qd_all, tau_motor_all], axis=1).astype(np.float32)
        y_np = tau_ext_gt_all.astype(np.float32)
        physics_np = physics_targets.astype(np.float32)

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)
        physics_tensor = torch.from_numpy(physics_np)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, physics_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._params.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self._params.learning_rate
        )
        mse_fn = torch.nn.MSELoss()

        self._network.train()

        final_mse = 0.0
        final_physics = 0.0
        final_total = 0.0

        for epoch in range(epochs):
            epoch_mse = 0.0
            epoch_physics = 0.0
            epoch_total = 0.0
            n_batches = 0

            for x_batch, y_batch, phys_batch in loader:
                optimizer.zero_grad()

                pred = self._network(x_batch)

                # Data loss
                mse_loss = mse_fn(pred, y_batch)

                # Physics loss: pred should be close to physics-derived tau_ext
                physics_loss = mse_fn(pred, phys_batch)

                total_loss = mse_loss + self._params.physics_weight * physics_loss
                total_loss.backward()
                optimizer.step()

                epoch_mse += mse_loss.item()
                epoch_physics += physics_loss.item()
                epoch_total += total_loss.item()
                n_batches += 1

            final_mse = epoch_mse / max(n_batches, 1)
            final_physics = epoch_physics / max(n_batches, 1)
            final_total = epoch_total / max(n_batches, 1)

            if verbose and (epoch + 1) % max(epochs // 10, 1) == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"mse={final_mse:.6f} physics={final_physics:.6f} "
                    f"total={final_total:.6f}"
                )

        self._network.eval()
        self._trained = True

        return {
            "mse_loss": final_mse,
            "physics_loss": final_physics,
            "total_loss": final_total,
        }
