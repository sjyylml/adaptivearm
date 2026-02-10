#!/usr/bin/env python3
"""Quick-start example: run a simulated 6-DOF arm with gravity compensation.

This demonstrates:
1. Creating a SimAdapter (MuJoCo simulation)
2. Reading robot state
3. Computing gravity compensation torques
4. Running a control loop
"""

import numpy as np

from openforce.adapters.sim import SimAdapter
from openforce.dynamics import MuJoCoDynamics


def main() -> None:
    # Create simulation adapter (uses built-in 6-DOF arm model)
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)

    # Reset to a non-zero initial configuration
    q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
    state = adapter.reset(q0)
    print(f"Initial joint positions: {state.q}")
    print(f"Number of joints: {adapter.n_joints}")
    print(f"Timestep: {adapter.dt} s")

    # Run gravity compensation for 2 seconds
    n_steps = int(2.0 / adapter.dt)
    positions = []

    for i in range(n_steps):
        state = adapter.get_state()
        # Compute gravity compensation torque
        g = dynamics.gravity_vector(state.q)
        adapter.send_torque(g)
        positions.append(state.q.copy())

        if i % 100 == 0:
            ee_pos = adapter.env.get_ee_position()
            print(
                f"  t={state.timestamp:.3f}s  "
                f"q_norm={np.linalg.norm(state.q):.4f}  "
                f"ee_pos=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
            )

    positions = np.array(positions)
    drift = np.linalg.norm(positions[-1] - positions[0])
    print(f"\nFinal joint positions: {state.q}")
    print(f"Position drift over 2s: {drift:.6f} rad")
    print("(Small drift indicates good gravity compensation)")

    # Optional: plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        t = np.arange(n_steps) * adapter.dt
        fig, ax = plt.subplots(figsize=(10, 5))
        for j in range(adapter.n_joints):
            ax.plot(t, positions[:, j], label=f"Joint {j}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (rad)")
        ax.set_title("Joint Positions Under Gravity Compensation")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("gravity_compensation.png", dpi=150)
        print("Plot saved to gravity_compensation.png")
    except ImportError:
        print("(Install matplotlib for plotting: pip install openforce[viz])")


if __name__ == "__main__":
    main()
