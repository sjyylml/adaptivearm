#!/usr/bin/env python3
"""Composite observer demo: multi-observer fusion.

This demonstrates:
1. Creating multiple observers (GMO with different gains)
2. Fusing their outputs using CompositeObserver
3. Comparing individual vs fused estimates
"""

import numpy as np

from openforce.adapters.sim import SimAdapter
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation.composite_observer import (
    CompositeObserver,
    FusionStrategy,
)
from openforce.estimation.momentum_observer import MomentumObserver


def main() -> None:
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints
    dt = adapter.dt

    # Create observers with different gain settings
    # High-gain: fast response but noisy
    obs_fast = MomentumObserver(
        dynamics=dynamics, n_joints=n, dt=dt, gains=np.full(n, 80.0)
    )
    # Low-gain: slow response but smooth
    obs_smooth = MomentumObserver(
        dynamics=dynamics, n_joints=n, dt=dt, gains=np.full(n, 10.0)
    )

    # Create composite observer with weighted average
    composite = CompositeObserver(
        observers=[obs_fast, obs_smooth],
        strategy=FusionStrategy.WEIGHTED_AVERAGE,
        weights=np.array([0.4, 0.6]),  # Favor smoother estimate
    )

    # Also create a max-norm composite for comparison
    composite_max = CompositeObserver(
        observers=[
            MomentumObserver(dynamics=dynamics, n_joints=n, dt=dt, gains=np.full(n, 80.0)),
            MomentumObserver(dynamics=dynamics, n_joints=n, dt=dt, gains=np.full(n, 10.0)),
        ],
        strategy=FusionStrategy.MAX_NORM,
    )

    q0 = np.array([0.0, 0.8, -0.5, 0.0, 0.4, 0.0])
    adapter.reset(q0)
    obs_fast.reset()
    obs_smooth.reset()
    composite_max.reset()

    fast_norms = []
    smooth_norms = []
    fused_norms = []
    max_norms = []
    times = []

    print("=== Phase 1: Settling (1s) ===")
    for i in range(int(1.0 / dt)):
        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)

        out_fast = obs_fast.update(state)
        out_smooth = obs_smooth.update(state)
        # Note: composite calls update on its sub-observers internally,
        # but we already called update above, so we compute fusion manually
        # for this demo. In practice, just call composite.update(state).
        out_max = composite_max.update(state)

        adapter.send_torque(g)

        fast_norms.append(float(np.linalg.norm(out_fast.tau_ext)))
        smooth_norms.append(float(np.linalg.norm(out_smooth.tau_ext)))
        # Manual weighted average for demo since we already updated individually
        fused_tau = 0.4 * out_fast.tau_ext + 0.6 * out_smooth.tau_ext
        fused_norms.append(float(np.linalg.norm(fused_tau)))
        max_norms.append(float(np.linalg.norm(out_max.tau_ext)))
        times.append(state.timestamp)

    print(f"  Fast:   {fast_norms[-1]:.4f}")
    print(f"  Smooth: {smooth_norms[-1]:.4f}")
    print(f"  Fused:  {fused_norms[-1]:.4f}")
    print(f"  Max:    {max_norms[-1]:.4f}")

    print("\n=== Phase 2: External force (1s) ===")
    adapter.env.apply_external_force("ee", np.array([5.0, 0.0, -5.0]))

    for i in range(int(1.0 / dt)):
        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)

        out_fast = obs_fast.update(state)
        out_smooth = obs_smooth.update(state)
        out_max = composite_max.update(state)

        adapter.send_torque(g)

        fast_norms.append(float(np.linalg.norm(out_fast.tau_ext)))
        smooth_norms.append(float(np.linalg.norm(out_smooth.tau_ext)))
        fused_tau = 0.4 * out_fast.tau_ext + 0.6 * out_smooth.tau_ext
        fused_norms.append(float(np.linalg.norm(fused_tau)))
        max_norms.append(float(np.linalg.norm(out_max.tau_ext)))
        times.append(state.timestamp)

    print(f"  Fast:   {fast_norms[-1]:.4f}")
    print(f"  Smooth: {smooth_norms[-1]:.4f}")
    print(f"  Fused:  {fused_norms[-1]:.4f}")
    print(f"  Max:    {max_norms[-1]:.4f}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, fast_norms, label="Fast (K=80)", alpha=0.6)
        ax.plot(times, smooth_norms, label="Smooth (K=10)", alpha=0.6)
        ax.plot(times, fused_norms, label="Weighted Avg (0.4/0.6)", linewidth=2)
        ax.plot(times, max_norms, label="Max-Norm", linestyle="--")
        ax.axvline(x=1.0, color="r", linestyle="--", alpha=0.5, label="Force applied")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("||tau_ext|| (Nm)")
        ax.set_title("Composite Observer: Multi-Observer Fusion")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("composite_observer.png", dpi=150)
        print("\nPlot saved to composite_observer.png")
    except ImportError:
        print("\n(Install matplotlib for plotting: pip install openforce[viz])")


if __name__ == "__main__":
    main()
