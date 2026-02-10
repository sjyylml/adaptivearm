"""Transformer-based sequence observer for external force estimation.

Uses a causal Transformer encoder with a sliding window of robot states
to predict external torques. Captures temporal dependencies that
single-timestep observers cannot.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

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

    class _PositionalEncoding(nn.Module):
        """Fixed sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]

    class _TransformerNetwork(nn.Module):
        """Causal Transformer encoder for sequence-to-one prediction."""

        def __init__(self, n_joints: int, params: TransformerObserverParams) -> None:
            super().__init__()
            self.n_joints = n_joints
            self.params = params

            input_dim = 3 * n_joints  # [q, qd, tau_motor]

            self.input_proj = nn.Linear(input_dim, params.d_model)
            self.pos_encoding = _PositionalEncoding(params.d_model, params.window_size)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=params.d_model,
                nhead=params.n_heads,
                dim_feedforward=params.d_ff,
                dropout=params.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=params.n_layers
            )

            self.output_proj = nn.Linear(params.d_model, n_joints)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, window_size, 3*n_joints).

            Returns:
                Predicted tau_ext of shape (batch, n_joints), from last timestep.
            """
            seq_len = x.size(1)

            # Causal mask: prevent attending to future timesteps
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

            h = self.input_proj(x)
            h = self.pos_encoding(h)
            h = self.transformer(h, mask=mask)

            # Take the last timestep output
            return self.output_proj(h[:, -1, :])


@dataclass
class TransformerObserverParams:
    """Configuration for the Transformer observer.

    Attributes:
        window_size: Number of timesteps in the sliding window.
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        d_ff: Feedforward network dimension.
        learning_rate: Adam optimizer learning rate.
        batch_size: Training batch size.
        dropout: Dropout probability.
    """

    window_size: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 32
    dropout: float = 0.1


class TransformerObserver(NeuralObserver):
    """Transformer-based sequence observer for external force estimation.

    Maintains a sliding window of recent robot states and uses a causal
    Transformer encoder to predict external torques from the sequence.

    Args:
        n_joints: Number of robot joints.
        params: Transformer configuration parameters.
    """

    def __init__(
        self,
        n_joints: int,
        params: TransformerObserverParams | None = None,
    ) -> None:
        super().__init__(n_joints)
        assert _HAS_TORCH

        self._params = params or TransformerObserverParams()
        self._network: Any = _TransformerNetwork(n_joints, self._params)
        self._network.eval()

        # Sliding window buffer
        self._window: deque[NDArray[np.floating]] = deque(
            maxlen=self._params.window_size
        )

    def _get_network(self) -> Any:
        return self._network

    def update(self, state: RobotState) -> ObserverOutput:
        """Run one observer step.

        Appends current state to the sliding window. When the window is full
        and the model is trained, performs inference. Otherwise returns zeros.

        Args:
            state: Current robot state.

        Returns:
            ObserverOutput with estimated external joint torques.
        """
        assert _HAS_TORCH

        # Append to sliding window
        features = np.concatenate([state.q, state.qd, state.tau_motor])
        self._window.append(features)

        if (
            self._trained
            and self._mode.value == "inference"
            and len(self._window) == self._params.window_size
        ):
            # Build sequence tensor
            seq = np.array(list(self._window), dtype=np.float32)
            x_tensor = torch.from_numpy(seq).unsqueeze(0)  # (1, window, features)

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

    def reset(self) -> None:
        """Reset observer state and clear sliding window."""
        super().reset()
        self._window.clear()

    def train(self, epochs: int = 100, verbose: bool = False) -> dict[str, float]:
        """Train the Transformer model on collected data.

        Constructs sliding window sequences from collected training data
        and trains with MSE loss.

        Args:
            epochs: Number of training epochs.
            verbose: Whether to print progress.

        Returns:
            Dictionary with final 'mse_loss'.
        """
        assert _HAS_TORCH
        _require_torch()

        data = self._training_data
        window = self._params.window_size

        if len(data) < window:
            raise ValueError(
                f"Need at least {window} training samples (window_size), "
                f"got {len(data)}."
            )

        # Build input arrays
        q_all = np.array(data.states_q, dtype=np.float32)
        qd_all = np.array(data.states_qd, dtype=np.float32)
        tau_motor_all = np.array(data.states_tau_motor, dtype=np.float32)
        tau_ext_gt_all = np.array(data.targets_tau_ext, dtype=np.float32)

        # Concatenate features: [q, qd, tau_motor]
        features = np.concatenate([q_all, qd_all, tau_motor_all], axis=1)

        # Create sliding window sequences
        n_samples = len(features) - window + 1
        x_windows = np.zeros(
            (n_samples, window, features.shape[1]), dtype=np.float32
        )
        y_targets = np.zeros((n_samples, self._n_joints), dtype=np.float32)

        for i in range(n_samples):
            x_windows[i] = features[i : i + window]
            y_targets[i] = tau_ext_gt_all[i + window - 1]

        x_tensor = torch.from_numpy(x_windows)
        y_tensor = torch.from_numpy(y_targets)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._params.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self._params.learning_rate
        )
        mse_fn = torch.nn.MSELoss()

        self._network.train()

        final_mse = 0.0

        for epoch in range(epochs):
            epoch_mse = 0.0
            n_batches = 0

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._network(x_batch)
                loss = mse_fn(pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_mse += loss.item()
                n_batches += 1

            final_mse = epoch_mse / max(n_batches, 1)

            if verbose and (epoch + 1) % max(epochs // 10, 1) == 0:
                print(f"Epoch {epoch + 1}/{epochs}: mse={final_mse:.6f}")

        self._network.eval()
        self._trained = True

        return {"mse_loss": final_mse}
