"""Automatic gain tuning for force/torque observers.

Optimizes observer gains by running simulations with known applied forces
and minimizing the estimation error against ground truth from VirtualForceSensor.
Supports grid search and scipy optimization methods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.core.types import ObserverOutput
from adaptivearm.dynamics.mujoco_dynamics import MuJoCoDynamics
from adaptivearm.estimation.base_observer import BaseObserver
from adaptivearm.sim.virtual_sensor import VirtualForceSensor


@dataclass
class AutoTunerConfig:
    """Configuration for the automatic gain tuner.

    Attributes:
        gain_range: Min and max gain values to search.
        metric: Error metric ("rmse", "mae", or "max_error").
        method: Optimization method ("grid", "nelder-mead", or "powell").
        sim_duration: Total simulation duration in seconds.
        settling_time: Time to wait before applying force (seconds).
        force_profile: Sequence of force magnitudes to apply on the last body.
        n_grid_points: Number of points per dimension for grid search.
        verbose: Whether to print progress.
    """

    gain_range: tuple[float, float] = (1.0, 200.0)
    metric: str = "rmse"
    method: str = "grid"
    sim_duration: float = 2.0
    settling_time: float = 0.5
    force_profile: NDArray[np.floating] = field(
        default_factory=lambda: np.array([5.0, 0.0, -5.0])
    )
    n_grid_points: int = 10
    verbose: bool = False


@dataclass
class TuningResult:
    """Result of the automatic tuning process.

    Attributes:
        best_gains: Optimal gain vector found.
        best_score: Best error metric value achieved.
        all_scores: All evaluated scores.
        all_gains: All evaluated gain vectors.
        n_evaluations: Total number of evaluations performed.
    """

    best_gains: NDArray[np.floating]
    best_score: float
    all_scores: list[float]
    all_gains: list[NDArray[np.floating]]
    n_evaluations: int


class AutoTuner:
    """Automatic gain optimizer for force/torque observers.

    Uses simulation with known applied forces to find optimal observer gains
    by minimizing estimation error against VirtualForceSensor ground truth.

    Args:
        adapter: SimAdapter for running simulations.
        dynamics: MuJoCo dynamics model.
        observer_factory: Callable that takes a gains array and returns an observer.
        config: Tuner configuration.
    """

    def __init__(
        self,
        adapter: SimAdapter,
        dynamics: MuJoCoDynamics,
        observer_factory: Callable[[NDArray[np.floating]], BaseObserver],
        config: AutoTunerConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._dynamics = dynamics
        self._observer_factory = observer_factory
        self._config = config or AutoTunerConfig()
        self._n_joints = adapter.n_joints
        self._sensor = VirtualForceSensor(adapter.env)

    def optimize(self) -> TuningResult:
        """Run the optimization process.

        Returns:
            TuningResult with the best gains found and evaluation history.
        """
        if self._config.method == "grid":
            return self._grid_search()
        else:
            return self._scipy_optimize()

    def _grid_search(self) -> TuningResult:
        """Perform grid search over gain space.

        Uses uniform gains (same value for all joints) to keep search tractable.
        """
        cfg = self._config
        gain_values = np.linspace(
            cfg.gain_range[0], cfg.gain_range[1], cfg.n_grid_points
        )

        all_scores: list[float] = []
        all_gains: list[NDArray[np.floating]] = []
        best_score = float("inf")
        best_gains = np.full(self._n_joints, gain_values[0])

        for gain_val in gain_values:
            gains = np.full(self._n_joints, gain_val)
            score = self._evaluate(gains)
            all_scores.append(score)
            all_gains.append(gains.copy())

            if score < best_score:
                best_score = score
                best_gains = gains.copy()

            if cfg.verbose:
                print(f"  gain={gain_val:.1f} -> {cfg.metric}={score:.6f}")

        return TuningResult(
            best_gains=best_gains,
            best_score=best_score,
            all_scores=all_scores,
            all_gains=all_gains,
            n_evaluations=len(all_scores),
        )

    def _scipy_optimize(self) -> TuningResult:
        """Optimize using scipy.optimize.minimize."""
        cfg = self._config
        all_scores: list[float] = []
        all_gains: list[NDArray[np.floating]] = []

        def objective(x: NDArray[Any]) -> float:
            gains = np.clip(x, cfg.gain_range[0], cfg.gain_range[1])
            score = self._evaluate(gains)
            all_scores.append(score)
            all_gains.append(gains.copy())
            if cfg.verbose:
                print(f"  gains={gains[:3]}... -> {cfg.metric}={score:.6f}")
            return score

        x0 = np.full(self._n_joints, np.mean(cfg.gain_range))
        bounds = [cfg.gain_range] * self._n_joints

        result = minimize(
            objective,
            x0,
            method=cfg.method,
            bounds=bounds if cfg.method == "powell" else None,
            options={"maxiter": 50, "xatol": 1.0, "fatol": 1e-4},
        )

        best_gains = np.clip(result.x, cfg.gain_range[0], cfg.gain_range[1])

        return TuningResult(
            best_gains=best_gains,
            best_score=float(result.fun),
            all_scores=all_scores,
            all_gains=all_gains,
            n_evaluations=len(all_scores),
        )

    def _evaluate(self, gains: NDArray[np.floating]) -> float:
        """Evaluate a gain setting by running a simulation.

        Steps:
        1. Reset adapter and create observer with given gains.
        2. Run settling phase (no force).
        3. Apply force profile and measure observer error vs ground truth.

        Args:
            gains: Observer gain vector, shape (n_joints,).

        Returns:
            Error metric value (lower is better).
        """
        cfg = self._config
        dt = self._adapter.dt

        # Create fresh observer
        observer = self._observer_factory(gains)

        # Reset simulation
        self._adapter.reset()
        observer.reset()

        # Settling phase
        n_settle = int(cfg.settling_time / dt)
        for _ in range(n_settle):
            state = self._adapter.get_state()
            observer.update(state)
            # Gravity compensation
            g = self._dynamics.gravity_vector(state.q)
            self._adapter.send_torque(g)

        # Force application phase
        force_duration = cfg.sim_duration - cfg.settling_time
        n_force_steps = int(force_duration / dt)
        force_per_segment = n_force_steps // max(len(cfg.force_profile), 1)

        errors: list[NDArray[np.floating]] = []
        model = self._adapter.env.model
        last_body = model.nbody - 1

        for step in range(n_force_steps):
            # Determine current force
            segment_idx = min(
                step // max(force_per_segment, 1),
                len(cfg.force_profile) - 1,
            )
            force_mag = cfg.force_profile[segment_idx]

            # Apply force to last body (z-direction)
            self._adapter.env.data.xfrc_applied[last_body] = [
                0.0,
                0.0,
                force_mag,
                0.0,
                0.0,
                0.0,
            ]

            state = self._adapter.get_state()
            output: ObserverOutput = observer.update(state)

            # Ground truth
            tau_ext_true = self._sensor.get_external_torques()

            error = output.tau_ext - tau_ext_true
            errors.append(error)

            # Gravity compensation
            g = self._dynamics.gravity_vector(state.q)
            self._adapter.send_torque(g)

        # Clear applied forces
        self._adapter.env.data.xfrc_applied[last_body] = [0.0] * 6

        if not errors:
            return float("inf")

        error_array = np.array(errors)

        if cfg.metric == "rmse":
            return float(np.sqrt(np.mean(error_array**2)))
        elif cfg.metric == "mae":
            return float(np.mean(np.abs(error_array)))
        elif cfg.metric == "max_error":
            return float(np.max(np.abs(error_array)))
        else:
            raise ValueError(f"Unknown metric: {cfg.metric}")
