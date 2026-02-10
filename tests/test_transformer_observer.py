"""Tests for TransformerObserver (skipped if torch is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from adaptivearm.core.robot_state import RobotState
from adaptivearm.estimation.neural_base import ObserverMode
from adaptivearm.estimation.transformer_observer import (
    TransformerObserver,
    TransformerObserverParams,
)


@pytest.fixture
def transformer_observer(n_joints):
    """Create a small Transformer observer for testing."""
    params = TransformerObserverParams(
        window_size=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        learning_rate=1e-3,
        batch_size=4,
        dropout=0.0,
    )
    return TransformerObserver(n_joints=n_joints, params=params)


class TestTransformerObserver:
    def test_construct(self, transformer_observer: TransformerObserver) -> None:
        """TransformerObserver should be constructible."""
        assert transformer_observer is not None
        assert not transformer_observer.trained
        assert transformer_observer.mode == ObserverMode.INFERENCE

    def test_update_returns_zeros_before_window_full(
        self, transformer_observer: TransformerObserver, n_joints: int
    ) -> None:
        """Should return zeros when window is not yet full."""
        state = RobotState(
            q=np.zeros(n_joints),
            qd=np.zeros(n_joints),
            tau_motor=np.zeros(n_joints),
            timestamp=0.0,
        )
        output = transformer_observer.update(state)
        np.testing.assert_array_equal(output.tau_ext, np.zeros(n_joints))

    def test_update_untrained_returns_zeros(
        self, transformer_observer: TransformerObserver, n_joints: int
    ) -> None:
        """Even with full window, untrained model should return zeros."""
        for i in range(4):
            state = RobotState(
                q=np.zeros(n_joints),
                qd=np.zeros(n_joints),
                tau_motor=np.zeros(n_joints),
                timestamp=i * 0.002,
            )
            output = transformer_observer.update(state)

        np.testing.assert_array_equal(output.tau_ext, np.zeros(n_joints))

    def test_collect_data(
        self, transformer_observer: TransformerObserver, n_joints: int
    ) -> None:
        """Data collection should accumulate samples."""
        state = RobotState(
            q=np.ones(n_joints),
            qd=np.zeros(n_joints),
            tau_motor=np.ones(n_joints) * 0.5,
            timestamp=0.1,
        )
        tau_ext_gt = np.ones(n_joints) * 0.1

        transformer_observer.collect(state, tau_ext_gt)
        assert len(transformer_observer.training_data) == 1

    def test_train_and_infer(
        self, transformer_observer: TransformerObserver, n_joints: int
    ) -> None:
        """Training should produce a model that can make predictions."""
        rng = np.random.default_rng(42)

        # Collect enough data for sliding windows (window_size=4, need >= 4)
        for i in range(10):
            q = rng.uniform(-0.5, 0.5, n_joints)
            qd = rng.uniform(-0.1, 0.1, n_joints)
            tau_motor = rng.uniform(-1.0, 1.0, n_joints)
            tau_ext = rng.uniform(-0.5, 0.5, n_joints)

            state = RobotState(
                q=q, qd=qd, tau_motor=tau_motor, timestamp=i * 0.002
            )
            transformer_observer.collect(state, tau_ext)

        # Train
        losses = transformer_observer.train(epochs=3, verbose=False)
        assert "mse_loss" in losses
        assert transformer_observer.trained

        # Reset window and fill it
        transformer_observer.reset()
        transformer_observer._trained = True  # reset() in inference mode doesn't clear trained

        for i in range(4):
            state = RobotState(
                q=rng.uniform(-0.5, 0.5, n_joints),
                qd=rng.uniform(-0.1, 0.1, n_joints),
                tau_motor=rng.uniform(-1.0, 1.0, n_joints),
                timestamp=i * 0.002,
            )
            output = transformer_observer.update(state)

        # Last update should have window full + trained -> non-zero in general
        assert output.tau_ext.shape == (n_joints,)

    def test_reset_clears_window(
        self, transformer_observer: TransformerObserver, n_joints: int
    ) -> None:
        """Reset should clear the sliding window."""
        for i in range(3):
            state = RobotState(
                q=np.zeros(n_joints),
                qd=np.zeros(n_joints),
                tau_motor=np.zeros(n_joints),
                timestamp=i * 0.002,
            )
            transformer_observer.update(state)

        assert len(transformer_observer._window) == 3
        transformer_observer.reset()
        assert len(transformer_observer._window) == 0

    def test_train_insufficient_data(
        self, transformer_observer: TransformerObserver, n_joints: int
    ) -> None:
        """Training with fewer samples than window_size should raise ValueError."""
        for i in range(3):  # window_size=4, need at least 4
            state = RobotState(
                q=np.zeros(n_joints),
                qd=np.zeros(n_joints),
                tau_motor=np.zeros(n_joints),
                timestamp=i * 0.002,
            )
            transformer_observer.collect(state, np.zeros(n_joints))

        with pytest.raises(ValueError, match="at least 4"):
            transformer_observer.train(epochs=1)

    def test_sliding_window_maxlen(
        self, transformer_observer: TransformerObserver, n_joints: int
    ) -> None:
        """Window should not exceed maxlen."""
        for i in range(10):
            state = RobotState(
                q=np.zeros(n_joints),
                qd=np.zeros(n_joints),
                tau_motor=np.zeros(n_joints),
                timestamp=i * 0.002,
            )
            transformer_observer.update(state)

        # window_size=4, so deque maxlen=4
        assert len(transformer_observer._window) == 4
