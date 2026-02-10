"""Tests for AutoTuner."""

from __future__ import annotations

import numpy as np
import pytest

from openforce.adapters.sim import SimAdapter
from openforce.dynamics.mujoco_dynamics import MuJoCoDynamics
from openforce.estimation.momentum_observer import MomentumObserver
from openforce.tuning.auto_tuner import AutoTuner, AutoTunerConfig, TuningResult


@pytest.fixture
def tuner_setup(sim_adapter: SimAdapter, dynamics: MuJoCoDynamics):
    """Set up AutoTuner with short simulation."""
    n_joints = sim_adapter.n_joints
    dt = sim_adapter.dt

    def observer_factory(gains):
        return MomentumObserver(
            dynamics=dynamics,
            n_joints=n_joints,
            dt=dt,
            gains=gains,
        )

    config = AutoTunerConfig(
        gain_range=(5.0, 50.0),
        metric="rmse",
        method="grid",
        sim_duration=0.3,
        settling_time=0.1,
        force_profile=np.array([5.0, -5.0]),
        n_grid_points=3,
        verbose=False,
    )

    return sim_adapter, dynamics, observer_factory, config


class TestAutoTuner:
    def test_construct(self, tuner_setup) -> None:
        """AutoTuner should be constructible."""
        adapter, dynamics, factory, config = tuner_setup
        tuner = AutoTuner(adapter, dynamics, factory, config)
        assert tuner is not None

    def test_evaluate(self, tuner_setup) -> None:
        """Single evaluation should return a finite score."""
        adapter, dynamics, factory, config = tuner_setup
        tuner = AutoTuner(adapter, dynamics, factory, config)
        n_joints = adapter.n_joints

        gains = np.full(n_joints, 20.0)
        score = tuner._evaluate(gains)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_grid_search(self, tuner_setup) -> None:
        """Grid search should return a valid TuningResult."""
        adapter, dynamics, factory, config = tuner_setup
        tuner = AutoTuner(adapter, dynamics, factory, config)

        result = tuner.optimize()
        assert isinstance(result, TuningResult)
        assert result.n_evaluations == config.n_grid_points
        assert len(result.all_scores) == config.n_grid_points
        assert len(result.all_gains) == config.n_grid_points
        assert result.best_score <= max(result.all_scores)
        assert result.best_gains.shape == (adapter.n_joints,)

    def test_config_defaults(self) -> None:
        """AutoTunerConfig defaults should be reasonable."""
        config = AutoTunerConfig()
        assert config.gain_range == (1.0, 200.0)
        assert config.metric == "rmse"
        assert config.method == "grid"
        assert config.sim_duration > 0
        assert config.n_grid_points > 0

    def test_different_metrics(self, tuner_setup) -> None:
        """Should work with all supported metrics."""
        adapter, dynamics, factory, config = tuner_setup

        for metric in ["rmse", "mae", "max_error"]:
            config.metric = metric
            tuner = AutoTuner(adapter, dynamics, factory, config)
            n_joints = adapter.n_joints

            score = tuner._evaluate(np.full(n_joints, 20.0))
            assert np.isfinite(score)
