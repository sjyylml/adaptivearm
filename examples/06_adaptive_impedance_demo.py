#!/usr/bin/env python3
"""Adaptive impedance controller demo: stiffness adaptation visualization.

This demonstrates:
1. Setting up an adaptive impedance controller
2. Observing stiffness softening when external forces are applied
3. Observing stiffness recovery when forces are removed
"""

import numpy as np

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.control.adaptive_impedance import (
    AdaptiveImpedanceController,
    AdaptiveImpedanceParams,
)
from adaptivearm.core.types import ObserverOutput
from adaptivearm.dynamics import MuJoCoDynamics
from adaptivearm.estimation.momentum_observer import MomentumObserver


def main() -> None:
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints
    dt = adapter.dt

    q_desired = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])

    # Create adaptive impedance controller
    ctrl = AdaptiveImpedanceController(
        dynamics=dynamics,
        n_joints=n,
        dt=dt,
        params=AdaptiveImpedanceParams(
            stiffness_init=np.full(n, 500.0),
            stiffness_min=np.full(n, 50.0),
            stiffness_max=np.full(n, 2000.0),
            damping_init=np.full(n, 50.0),
            damping_min=np.full(n, 5.0),
            damping_max=np.full(n, 200.0),
            alpha_force=5.0,
            alpha_error=2.0,
            force_threshold=1.0,
            q_desired=q_desired,
            use_coriolis_comp=True,
        ),
    )

    # Create observer for force feedback
    observer = MomentumObserver(
        dynamics=dynamics, n_joints=n, dt=dt, gains=np.full(n, 30.0)
    )

    adapter.reset(q_desired)
    observer.reset()
    ctrl.reset()

    stiffness_history = []
    error_history = []
    times = []

    print("=== Phase 1: Holding position (2s) ===")
    for i in range(int(2.0 / dt)):
        state = adapter.get_state()
        obs = observer.update(state)
        output = ctrl.compute(state, obs)
        adapter.send_torque(output.tau_cmd)

        stiffness_history.append(ctrl.stiffness[1])  # Track joint 1
        error_history.append(float(np.linalg.norm(state.q - q_desired)))
        times.append(state.timestamp)

    print(f"  Stiffness: {ctrl.stiffness[1]:.1f} Nm/rad")
    print(f"  Position error: {error_history[-1]:.4f} rad")

    print("\n=== Phase 2: External force applied (2s) ===")
    adapter.env.apply_external_force("ee", np.array([8.0, 0.0, -8.0]))

    for i in range(int(2.0 / dt)):
        state = adapter.get_state()
        obs = observer.update(state)
        output = ctrl.compute(state, obs)
        adapter.send_torque(output.tau_cmd)

        stiffness_history.append(ctrl.stiffness[1])
        error_history.append(float(np.linalg.norm(state.q - q_desired)))
        times.append(state.timestamp)

    print(f"  Stiffness: {ctrl.stiffness[1]:.1f} Nm/rad (should be lower)")
    print(f"  Position error: {error_history[-1]:.4f} rad")

    print("\n=== Phase 3: Force removed, recovery (2s) ===")
    adapter.env.apply_external_force("ee", np.zeros(3))

    for i in range(int(2.0 / dt)):
        state = adapter.get_state()
        obs = observer.update(state)
        output = ctrl.compute(state, obs)
        adapter.send_torque(output.tau_cmd)

        stiffness_history.append(ctrl.stiffness[1])
        error_history.append(float(np.linalg.norm(state.q - q_desired)))
        times.append(state.timestamp)

    print(f"  Stiffness: {ctrl.stiffness[1]:.1f} Nm/rad (should recover)")
    print(f"  Position error: {error_history[-1]:.4f} rad")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(times, stiffness_history, "b-")
        ax1.axvline(x=2.0, color="r", linestyle="--", label="Force ON")
        ax1.axvline(x=4.0, color="g", linestyle="--", label="Force OFF")
        ax1.set_ylabel("Stiffness (Nm/rad)")
        ax1.set_title("Adaptive Impedance: Joint 1 Stiffness")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(times, error_history, "r-")
        ax2.axvline(x=2.0, color="r", linestyle="--", label="Force ON")
        ax2.axvline(x=4.0, color="g", linestyle="--", label="Force OFF")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position Error (rad)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("adaptive_impedance.png", dpi=150)
        print("\nPlot saved to adaptive_impedance.png")
    except ImportError:
        print("\n(Install matplotlib for plotting: pip install adaptivearm[viz])")


if __name__ == "__main__":
    main()
