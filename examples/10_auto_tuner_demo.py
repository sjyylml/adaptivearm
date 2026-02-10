#!/usr/bin/env python3
"""AutoTuner demo: automatically optimize observer gains.

This demonstrates:
1. Configuring the AutoTuner with a gain search range
2. Running grid search to find optimal gains
3. Comparing default vs optimized gains
"""

import numpy as np

from openforce.adapters.sim import SimAdapter
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation import MomentumObserver
from openforce.tuning.auto_tuner import AutoTuner, AutoTunerConfig


def main() -> None:
    # Setup
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints

    # Define observer factory: creates a new observer for each gain setting
    def observer_factory(gains):
        return MomentumObserver(
            dynamics=dynamics,
            n_joints=n,
            dt=adapter.dt,
            gains=gains,
            lowpass_cutoff=10.0,
        )

    # Configure grid search
    config = AutoTunerConfig(
        gain_range=(5.0, 100.0),
        metric="rmse",
        method="grid",
        sim_duration=1.0,
        settling_time=0.3,
        force_profile=np.array([5.0, 0.0, -5.0, 10.0]),
        n_grid_points=10,
        verbose=True,
    )

    print("=== AutoTuner: Grid Search ===")
    print(f"Gain range: {config.gain_range}")
    print(f"Grid points: {config.n_grid_points}")
    print(f"Metric: {config.metric}")
    print(f"Sim duration: {config.sim_duration}s\n")

    tuner = AutoTuner(adapter, dynamics, observer_factory, config)
    result = tuner.optimize()

    print(f"\n--- Results ---")
    print(f"Best gains: {result.best_gains[0]:.1f} (uniform)")
    print(f"Best {config.metric}: {result.best_score:.6f}")
    print(f"Evaluations: {result.n_evaluations}")

    # Compare with default gains
    print("\n--- Comparison ---")
    default_gains = np.full(n, 10.0)
    adapter.reset()
    default_score = tuner._evaluate(default_gains)
    print(f"Default gains (10.0): {config.metric}={default_score:.6f}")
    print(f"Optimized gains ({result.best_gains[0]:.1f}): {config.metric}={result.best_score:.6f}")

    improvement = (default_score - result.best_score) / default_score * 100
    print(f"Improvement: {improvement:.1f}%")

    # Optional: plot gain vs error
    try:
        import matplotlib.pyplot as plt

        gains = [g[0] for g in result.all_gains]
        scores = result.all_scores

        plt.figure(figsize=(8, 5))
        plt.plot(gains, scores, "bo-", markersize=8)
        plt.axvline(result.best_gains[0], color="r", linestyle="--", label="Best")
        plt.xlabel("Gain value")
        plt.ylabel(f"{config.metric.upper()}")
        plt.title("AutoTuner: Gain vs Estimation Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("auto_tuner_demo.png", dpi=150)
        print("\nPlot saved to auto_tuner_demo.png")
    except ImportError:
        print("\n(Install matplotlib for plotting: pip install openforce[viz])")


if __name__ == "__main__":
    main()
