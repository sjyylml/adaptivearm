"""Tests for PINNObserver (skipped if torch is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from openforce.core.robot_state import RobotState
from openforce.estimation.neural_base import ObserverMode, TrainingData
from openforce.estimation.pinn_observer import PINNObserver, PINNParams


@pytest.fixture
def pinn_observer(dynamics, n_joints):
    """Create a small PINN observer for testing."""
    params = PINNParams(
        hidden_dims=[16, 16],
        learning_rate=1e-3,
        physics_weight=0.1,
        batch_size=8,
    )
    return PINNObserver(n_joints=n_joints, dynamics=dynamics, dt=0.002, params=params)


class TestPINNObserver:
    def test_construct(self, pinn_observer: PINNObserver) -> None:
        """PINNObserver should be constructible."""
        assert pinn_observer is not None
        assert not pinn_observer.trained
        assert pinn_observer.mode == ObserverMode.INFERENCE

    def test_update_untrained_returns_zeros(
        self, pinn_observer: PINNObserver, n_joints: int
    ) -> None:
        """Untrained observer should return zero torques."""
        state = RobotState(
            q=np.zeros(n_joints),
            qd=np.zeros(n_joints),
            tau_motor=np.zeros(n_joints),
            timestamp=0.0,
        )
        output = pinn_observer.update(state)
        np.testing.assert_array_equal(output.tau_ext, np.zeros(n_joints))

    def test_collect_data(
        self, pinn_observer: PINNObserver, n_joints: int
    ) -> None:
        """Data collection should accumulate samples."""
        state = RobotState(
            q=np.ones(n_joints),
            qd=np.zeros(n_joints),
            tau_motor=np.ones(n_joints) * 0.5,
            timestamp=0.1,
        )
        tau_ext_gt = np.ones(n_joints) * 0.1

        pinn_observer.collect(state, tau_ext_gt)
        assert len(pinn_observer.training_data) == 1

        pinn_observer.collect(state, tau_ext_gt)
        assert len(pinn_observer.training_data) == 2

    def test_train_and_infer(
        self, pinn_observer: PINNObserver, n_joints: int
    ) -> None:
        """Training should produce a model that gives non-zero predictions."""
        rng = np.random.default_rng(42)

        # Collect training data
        for i in range(20):
            q = rng.uniform(-0.5, 0.5, n_joints)
            qd = rng.uniform(-0.1, 0.1, n_joints)
            tau_motor = rng.uniform(-1.0, 1.0, n_joints)
            tau_ext = rng.uniform(-0.5, 0.5, n_joints)

            state = RobotState(
                q=q, qd=qd, tau_motor=tau_motor, timestamp=i * 0.002
            )
            pinn_observer.collect(state, tau_ext)

        # Train
        losses = pinn_observer.train(epochs=5, verbose=False)
        assert "mse_loss" in losses
        assert "physics_loss" in losses
        assert "total_loss" in losses
        assert pinn_observer.trained

        # Inference should now return non-zero (in general)
        state = RobotState(
            q=rng.uniform(-0.5, 0.5, n_joints),
            qd=rng.uniform(-0.1, 0.1, n_joints),
            tau_motor=rng.uniform(-1.0, 1.0, n_joints),
            timestamp=1.0,
        )
        output = pinn_observer.update(state)
        assert output.tau_ext.shape == (n_joints,)

    def test_reset_clears_training_data(
        self, pinn_observer: PINNObserver, n_joints: int
    ) -> None:
        """Reset in training mode should clear collected data."""
        state = RobotState(
            q=np.zeros(n_joints),
            qd=np.zeros(n_joints),
            tau_motor=np.zeros(n_joints),
            timestamp=0.0,
        )
        pinn_observer.collect(state, np.zeros(n_joints))
        assert len(pinn_observer.training_data) == 1

        pinn_observer.set_mode(ObserverMode.TRAINING)
        pinn_observer.reset()
        assert len(pinn_observer.training_data) == 0

    def test_save_load(
        self, pinn_observer: PINNObserver, n_joints: int, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Model should be saveable and loadable."""
        rng = np.random.default_rng(42)

        # Collect and train
        for i in range(20):
            state = RobotState(
                q=rng.uniform(-0.5, 0.5, n_joints),
                qd=rng.uniform(-0.1, 0.1, n_joints),
                tau_motor=rng.uniform(-1.0, 1.0, n_joints),
                timestamp=i * 0.002,
            )
            pinn_observer.collect(state, rng.uniform(-0.5, 0.5, n_joints))

        pinn_observer.train(epochs=3)

        # Save
        path = tmp_path / "pinn_model.pt"  # type: ignore[operator]
        pinn_observer.save(path)

        # Load into new observer
        params = PINNParams(hidden_dims=[16, 16])
        new_observer = PINNObserver(
            n_joints=n_joints,
            dynamics=pinn_observer._dynamics,
            dt=0.002,
            params=params,
        )
        new_observer.load(path)
        assert new_observer.trained

    def test_train_insufficient_data(
        self, pinn_observer: PINNObserver, n_joints: int
    ) -> None:
        """Training with < 2 samples should raise ValueError."""
        state = RobotState(
            q=np.zeros(n_joints),
            qd=np.zeros(n_joints),
            tau_motor=np.zeros(n_joints),
            timestamp=0.0,
        )
        pinn_observer.collect(state, np.zeros(n_joints))

        with pytest.raises(ValueError, match="at least 2"):
            pinn_observer.train(epochs=1)
