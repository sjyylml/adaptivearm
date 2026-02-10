#!/usr/bin/env python3
"""Transformer Observer demo: train a sequence model for force estimation.

This demonstrates:
1. Collecting sequential training data from simulation
2. Training a Transformer observer with sliding window sequences
3. Comparing Transformer estimates with ground truth
"""

import numpy as np

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.dynamics import MuJoCoDynamics
from adaptivearm.estimation.transformer_observer import (
    TransformerObserver,
    TransformerObserverParams,
)
from adaptivearm.sim.virtual_sensor import VirtualForceSensor


def main() -> None:
    # Setup
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints

    params = TransformerObserverParams(
        window_size=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        learning_rate=1e-3,
        batch_size=32,
        dropout=0.0,
    )
    observer = TransformerObserver(n_joints=n, params=params)
    sensor = VirtualForceSensor(adapter.env)

    # --- Phase 1: Collect training data ---
    print("Phase 1: Collecting training data...")
    q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
    adapter.reset(q0)

    rng = np.random.default_rng(42)
    collect_steps = 500
    last_body = adapter.env.model.nbody - 1

    for i in range(collect_steps):
        # Apply varying forces
        if i % 50 == 0:
            fz = rng.uniform(-10.0, 10.0)
            adapter.env.data.xfrc_applied[last_body] = [0, 0, fz, 0, 0, 0]

        state = adapter.get_state()
        tau_ext_gt = sensor.get_external_torques()
        observer.collect(state, tau_ext_gt)

        g = dynamics.gravity_vector(state.q)
        adapter.send_torque(g)

    print(f"  Collected {len(observer.training_data)} samples")

    # --- Phase 2: Train ---
    print("\nPhase 2: Training Transformer observer...")
    losses = observer.train(epochs=30, verbose=True)
    print(f"\nFinal losses: {losses}")

    # --- Phase 3: Evaluate ---
    print("\nPhase 3: Evaluating on new data...")
    adapter.reset(q0)
    observer.reset()

    adapter.env.data.xfrc_applied[last_body] = [0] * 6

    # Fill the window with settling data
    for _ in range(200):
        state = adapter.get_state()
        observer.update(state)
        g = dynamics.gravity_vector(state.q)
        adapter.send_torque(g)

    # Apply test force
    test_force = np.array([0.0, 0.0, -8.0])
    adapter.env.data.xfrc_applied[last_body] = [0, 0, test_force[2], 0, 0, 0]

    estimated_torques = []
    true_torques = []
    timestamps = []

    eval_steps = 500
    for i in range(eval_steps):
        state = adapter.get_state()
        output = observer.update(state)
        true_tau = sensor.get_external_torques()

        estimated_torques.append(output.tau_ext.copy())
        true_torques.append(true_tau.copy())
        timestamps.append(state.timestamp)

        g = dynamics.gravity_vector(state.q)
        adapter.send_torque(g)

        if i % 100 == 0:
            err = np.linalg.norm(output.tau_ext - true_tau)
            print(f"  t={state.timestamp:.3f}s  error={err:.3f} Nm")

    estimated_torques = np.array(estimated_torques)
    true_torques = np.array(true_torques)
    errors = estimated_torques - true_torques
    rmse = np.sqrt(np.mean(errors**2))
    print(f"\nTotal RMSE: {rmse:.4f} Nm")

    # Optional: plot
    try:
        import matplotlib.pyplot as plt

        t = np.array(timestamps)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        for j in range(min(n, 3)):
            ax.plot(t, estimated_torques[:, j], label=f"Transformer joint {j}")
            ax.plot(t, true_torques[:, j], "--", label=f"True joint {j}", alpha=0.7)
        ax.set_ylabel("Torque (Nm)")
        ax.set_title("Transformer Observer: Estimated vs True External Torques")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True)

        ax = axes[1]
        ax.plot(t, np.linalg.norm(errors, axis=1), "r-")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error norm (Nm)")
        ax.set_title(f"Estimation Error (RMSE = {rmse:.3f} Nm)")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("transformer_observer_demo.png", dpi=150)
        print("\nPlot saved to transformer_observer_demo.png")
    except ImportError:
        print("\n(Install matplotlib for plotting: pip install adaptivearm[viz])")


if __name__ == "__main__":
    main()
