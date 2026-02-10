#!/usr/bin/env python3
"""Impedance control demo: arm tracks a target with spring-damper behavior.

Demonstrates:
1. Setting up an impedance controller
2. Moving to a target configuration
3. Applying external disturbance and observing compliant response
"""

import numpy as np

from openforce.adapters.sim import SimAdapter
from openforce.control.impedance import ImpedanceController, ImpedanceParams
from openforce.dynamics import MuJoCoDynamics


def main() -> None:
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints

    q_target = np.array([0.0, 0.6, -0.4, 0.0, 0.3, 0.0])
    ctrl = ImpedanceController(
        dynamics=dynamics,
        n_joints=n,
        params=ImpedanceParams(
            stiffness=np.full(n, 150.0),
            damping=np.full(n, 30.0),
            q_desired=q_target,
            use_coriolis_comp=True,
        ),
    )

    adapter.reset(np.zeros(n))
    print(f"Target: {q_target}")
    print("Phase 1: Moving to target...")

    for i in range(1500):
        state = adapter.get_state()
        output = ctrl.compute(state)
        adapter.send_torque(output.tau_cmd)

        if i % 300 == 0:
            err = np.linalg.norm(state.q - q_target)
            print(f"  t={state.timestamp:.2f}s  error={err:.4f} rad")

    print(f"\nPhase 2: Applying disturbance force at t={adapter.env.time:.2f}s...")
    adapter.env.apply_external_force("ee", np.array([5.0, 0.0, -5.0]))

    for i in range(1000):
        state = adapter.get_state()
        output = ctrl.compute(state)
        adapter.send_torque(output.tau_cmd)

        if i % 200 == 0:
            err = np.linalg.norm(state.q - q_target)
            print(f"  t={state.timestamp:.2f}s  error={err:.4f} rad  (with disturbance)")

    print("\nPhase 3: Removing disturbance...")
    adapter.env.clear_external_forces()

    for i in range(1000):
        state = adapter.get_state()
        output = ctrl.compute(state)
        adapter.send_torque(output.tau_cmd)

    state = adapter.get_state()
    final_err = np.linalg.norm(state.q - q_target)
    print(f"  Final error after recovery: {final_err:.4f} rad")
    print(f"  (Impedance controller {'recovered' if final_err < 0.1 else 'still settling'})")


if __name__ == "__main__":
    main()
