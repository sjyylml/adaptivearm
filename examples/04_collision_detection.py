#!/usr/bin/env python3
"""Collision detection demo: detect and react to external contacts.

Demonstrates:
1. Running GMO + collision detector together
2. Detecting a simulated collision
3. Safety monitor stopping the robot
"""

import numpy as np

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.control.safety_monitor import SafetyMonitor
from adaptivearm.dynamics import MuJoCoDynamics
from adaptivearm.estimation import CollisionDetector, MomentumObserver
from adaptivearm.sim.virtual_sensor import VirtualForceSensor


def main() -> None:
    adapter = SimAdapter()
    dynamics = MuJoCoDynamics(adapter.env.model)
    n = adapter.n_joints

    observer = MomentumObserver(
        dynamics=dynamics, n_joints=n, dt=adapter.dt,
        gains=np.full(n, 50.0), lowpass_cutoff=15.0,
    )
    detector = CollisionDetector(
        n_joints=n, thresholds=np.full(n, 3.0), holdoff_time=0.2,
    )
    safety = SafetyMonitor(n_joints=n, collision_detector=detector)

    q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
    adapter.reset(q0)
    observer.reset()
    safety.reset()

    print("Phase 1: Normal operation (gravity compensation)...")
    for i in range(500):
        state = adapter.get_state()
        obs_out = observer.update(state)
        g = dynamics.gravity_vector(state.q)
        safe_out = safety.filter(g, state.q, state.qd, obs_out)
        adapter.send_torque(safe_out.tau_cmd)

    print(f"  Status: {safety.state.name}")

    print(f"\nPhase 2: Simulating collision at t={adapter.env.time:.2f}s...")
    adapter.env.apply_external_force("ee", np.array([20.0, 0.0, -15.0]))

    collision_detected = False
    for i in range(300):
        state = adapter.get_state()
        obs_out = observer.update(state)
        g = dynamics.gravity_vector(state.q)
        safe_out = safety.filter(g, state.q, state.qd, obs_out)
        adapter.send_torque(safe_out.tau_cmd)

        if safety.state.name == "COLLISION_DETECTED" and not collision_detected:
            collision_detected = True
            event = safety.last_collision
            print(f"  COLLISION at t={state.timestamp:.3f}s!")
            if event is not None:
                print(f"  Severity: {event.severity:.2f}x threshold")
                print(f"  Joints: {np.where(event.joint_mask)[0]}")
            print(f"  Torque output: {np.linalg.norm(safe_out.tau_cmd):.2f} Nm (should be ~0)")

    print(f"\nPhase 3: Removing collision force...")
    adapter.env.clear_external_forces()
    safety.reset()

    for i in range(500):
        state = adapter.get_state()
        obs_out = observer.update(state)
        g = dynamics.gravity_vector(state.q)
        safe_out = safety.filter(g, state.q, state.qd, obs_out)
        adapter.send_torque(safe_out.tau_cmd)

    print(f"  Status: {safety.state.name}")
    print("  Robot resumed normal operation." if safety.state.name == "NORMAL" else "  Still in safe mode.")


if __name__ == "__main__":
    main()
