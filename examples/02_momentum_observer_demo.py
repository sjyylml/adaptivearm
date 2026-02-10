#!/usr/bin/env python3
"""Momentum Observer demo: estimate external forces on a simulated arm.

This demonstrates:
1. Setting up the Generalized Momentum Observer (GMO)
2. Applying an external force in simulation
3. Comparing GMO estimates with ground truth
4. Computing RMSE of the force estimate
"""

import numpy as np

from openforce.adapters.sim import SimAdapter
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation import MomentumObserver
from openforce.sim.virtual_sensor import VirtualForceSensor


def main() -> None:
    # Setup
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints

    observer = MomentumObserver(
        dynamics=dynamics,
        n_joints=n,
        dt=adapter.dt,
        gains=np.full(n, 30.0),
        lowpass_cutoff=10.0,  # 10 Hz low-pass filter on residual
    )
    sensor = VirtualForceSensor(adapter.env)

    # Reset to a configuration
    q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
    adapter.reset(q0)
    observer.reset()

    # Phase 1: Let the observer settle (no external force)
    settle_steps = 500
    print("Phase 1: Settling observer (no external force)...")
    for _ in range(settle_steps):
        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)
        observer.update(state)
        adapter.send_torque(g)

    # Phase 2: Apply external force and record data
    force_steps = 1000
    applied_force = np.array([0.0, 0.0, -10.0])  # 10N downward on end-effector
    adapter.env.apply_external_force("ee", applied_force)

    print(f"Phase 2: Applying {applied_force} N force on end-effector...")

    estimated_torques = []
    true_torques = []
    timestamps = []

    for i in range(force_steps):
        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)
        output = observer.update(state)
        adapter.send_torque(g)

        true_tau = sensor.get_external_torques()
        estimated_torques.append(output.tau_ext.copy())
        true_torques.append(true_tau.copy())
        timestamps.append(state.timestamp)

        if i % 200 == 0:
            err = np.linalg.norm(output.tau_ext - true_tau)
            print(
                f"  t={state.timestamp:.3f}s  "
                f"|τ_est|={np.linalg.norm(output.tau_ext):.3f}  "
                f"|τ_true|={np.linalg.norm(true_tau):.3f}  "
                f"error={err:.3f}"
            )

    estimated_torques = np.array(estimated_torques)
    true_torques = np.array(true_torques)

    # Compute RMSE (per-joint and total)
    errors = estimated_torques - true_torques
    rmse_per_joint = np.sqrt(np.mean(errors**2, axis=0))
    rmse_total = np.sqrt(np.mean(errors**2))

    print(f"\n--- Results ---")
    print(f"RMSE per joint: {rmse_per_joint}")
    print(f"Total RMSE: {rmse_total:.4f} Nm")

    # Convert to Cartesian force estimate if wrench available
    state = adapter.get_state()
    output = observer.update(state)
    if output.wrench_ext is not None:
        print(f"\nEstimated Cartesian wrench: {output.wrench_ext[:3]} N (force)")
        print(f"Applied force:              {applied_force} N")
        force_error = np.linalg.norm(output.wrench_ext[:3] - applied_force)
        print(f"Cartesian force error: {force_error:.2f} N")

    # Optional: plot
    try:
        import matplotlib.pyplot as plt

        t = np.array(timestamps)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Joint torques comparison
        ax = axes[0]
        for j in range(min(n, 3)):  # Plot first 3 joints
            ax.plot(t, estimated_torques[:, j], label=f"Est joint {j}", linewidth=1.5)
            ax.plot(
                t, true_torques[:, j], "--", label=f"True joint {j}", linewidth=1, alpha=0.7
            )
        ax.set_ylabel("Torque (Nm)")
        ax.set_title("GMO: Estimated vs True External Torques")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True)

        # Error
        ax = axes[1]
        total_error = np.linalg.norm(errors, axis=1)
        ax.plot(t, total_error, "r-", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error norm (Nm)")
        ax.set_title(f"Estimation Error (RMSE = {rmse_total:.3f} Nm)")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("momentum_observer_demo.png", dpi=150)
        print("\nPlot saved to momentum_observer_demo.png")
    except ImportError:
        print("\n(Install matplotlib for plotting: pip install openforce[viz])")


if __name__ == "__main__":
    main()
