#!/usr/bin/env python3
"""Custom model demo: load robots by name from the model registry.

This demonstrates:
1. Listing available robot models
2. Loading a model by name via SimAdapter
3. Registering a custom model
4. Running gravity compensation with a named model
"""

import numpy as np

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.dynamics import MuJoCoDynamics
from adaptivearm.models import RobotModelInfo, get_model, list_models, register_model


def main() -> None:
    # --- 1. List available models ---
    print("=== Available Robot Models ===")
    for name in list_models():
        info = get_model(name)
        print(f"  {name}: {info.description}")
        print(f"    path: {info.model_path}")
        print(f"    joints: {info.n_joints}, ee_site: {info.ee_site_name}")

    # --- 2. Load by name ---
    print("\n=== Loading 'default_6dof' by name ===")
    adapter = SimAdapter(model_name="default_6dof")
    dynamics = MuJoCoDynamics(adapter.env.model)

    print(f"Joints: {adapter.n_joints}")
    print(f"Timestep: {adapter.dt}s")
    print(f"EE site: {adapter.env.ee_site_name}")

    # --- 3. Gravity compensation ---
    q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
    adapter.reset(q0)

    print("\nRunning gravity compensation for 1s...")
    n_steps = int(1.0 / adapter.dt)
    for i in range(n_steps):
        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)
        adapter.send_torque(g)

    state = adapter.get_state()
    drift = np.linalg.norm(state.q - q0)
    print(f"Position drift: {drift:.6f} rad (should be near zero)")

    # --- 4. How to register a custom model ---
    print("\n=== Registering a Custom Model (example) ===")
    print("To add your own robot, place the URDF/MJCF file and register it:")
    print()
    print("  from pathlib import Path")
    print("  from adaptivearm.models import RobotModelInfo, register_model")
    print()
    print("  register_model(RobotModelInfo(")
    print('      name="ur5e",')
    print('      model_path=Path("path/to/ur5e.urdf"),')
    print("      n_joints=6,")
    print('      ee_site_name="ee_site",')
    print('      ee_body_name="tool0",')
    print("  ))")
    print()
    print("  adapter = SimAdapter(model_name='ur5e')")
    print()
    print(f"Currently registered: {list_models()}")


if __name__ == "__main__":
    main()
