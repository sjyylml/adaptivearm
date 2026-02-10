#!/usr/bin/env python3
"""EKF observer demo: compare EKF vs GMO force estimation.

This demonstrates:
1. Running both GMO and EKF observers on the same simulation
2. Applying an external force and comparing estimation quality
3. Visualizing convergence behavior differences
"""

import numpy as np

from openforce.adapters.sim import SimAdapter
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation.ekf_observer import EKFObserver, EKFParams
from openforce.estimation.momentum_observer import MomentumObserver


def main() -> None:
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints
    dt = adapter.dt

    # Create GMO observer
    gmo = MomentumObserver(
        dynamics=dynamics,
        n_joints=n,
        dt=dt,
        gains=np.full(n, 30.0),
    )

    # Create EKF observer
    ekf = EKFObserver(
        dynamics=dynamics,
        n_joints=n,
        dt=dt,
        params=EKFParams(
            Q_q=np.full(n, 1e-6),
            Q_qd=np.full(n, 1e-4),
            Q_tau=np.full(n, 1.0),
            R=np.full(n, 1e-6),
        ),
    )

    # Initialize
    q0 = np.array([0.0, 0.8, -0.5, 0.0, 0.4, 0.0])
    adapter.reset(q0)
    gmo.reset()
    ekf.reset()

    print("=== Phase 1: Settling (no external force) ===")
    gmo_norms = []
    ekf_norms = []
    times = []

    for i in range(500):
        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)

        out_gmo = gmo.update(state)
        out_ekf = ekf.update(state)

        adapter.send_torque(g)

        gmo_norms.append(float(np.linalg.norm(out_gmo.tau_ext)))
        ekf_norms.append(float(np.linalg.norm(out_ekf.tau_ext)))
        times.append(state.timestamp)

        if i % 100 == 0:
            print(
                f"  t={state.timestamp:.3f}s  "
                f"GMO={gmo_norms[-1]:.4f}  "
                f"EKF={ekf_norms[-1]:.4f}"
            )

    print(f"\nSettled residuals: GMO={gmo_norms[-1]:.4f}, EKF={ekf_norms[-1]:.4f}")

    print("\n=== Phase 2: External force applied ===")
    adapter.env.apply_external_force("ee", np.array([5.0, 0.0, -5.0]))

    for i in range(300):
        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)

        out_gmo = gmo.update(state)
        out_ekf = ekf.update(state)

        adapter.send_torque(g)

        gmo_norms.append(float(np.linalg.norm(out_gmo.tau_ext)))
        ekf_norms.append(float(np.linalg.norm(out_ekf.tau_ext)))
        times.append(state.timestamp)

        if i % 50 == 0:
            print(
                f"  t={state.timestamp:.3f}s  "
                f"GMO={gmo_norms[-1]:.4f}  "
                f"EKF={ekf_norms[-1]:.4f}"
            )

    print(f"\nPeak GMO: {max(gmo_norms[500:]):.4f}")
    print(f"Peak EKF: {max(ekf_norms[500:]):.4f}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, gmo_norms, label="GMO", alpha=0.8)
        ax.plot(times, ekf_norms, label="EKF", alpha=0.8)
        ax.axvline(x=times[500], color="r", linestyle="--", label="Force applied")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("||tau_ext|| (Nm)")
        ax.set_title("EKF vs GMO Force Estimation")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("ekf_vs_gmo.png", dpi=150)
        print("\nPlot saved to ekf_vs_gmo.png")
    except ImportError:
        print("\n(Install matplotlib for plotting: pip install openforce[viz])")


if __name__ == "__main__":
    main()
