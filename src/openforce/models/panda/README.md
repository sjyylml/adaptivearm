# Franka Panda Model

Place your Franka Emika Panda URDF and mesh files in this directory.

## Required files

```
panda/
├── panda.urdf         # Main URDF file
└── meshes/            # Collision and visual meshes
    ├── collision/
    └── visual/
```

## Where to get the URDF

- **Official**: [franka_ros2](https://github.com/frankaemika/franka_ros2) → `franka_description`
- **MuJoCo models**: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) (MJCF format, can be used directly)

## Registration

After placing the files, register the model in your code:

```python
from pathlib import Path
from openforce.models import RobotModelInfo, register_model

register_model(RobotModelInfo(
    name="panda",
    model_path=Path(__file__).parent / "panda.urdf",
    n_joints=7,
    ee_site_name="ee_site",      # Must match your URDF/MJCF site name
    ee_body_name="panda_hand",   # Panda end-effector link
    description="Franka Emika Panda 7-DOF collaborative robot",
))
```

## Usage

```python
from openforce.adapters.sim import SimAdapter
from openforce.dynamics import MuJoCoDynamics

adapter = SimAdapter(model_name="panda")
dynamics = MuJoCoDynamics(adapter.env.model)
```

## Notes

- The Panda has 7 DOF (joints) + 2 finger joints. The `n_joints=7` refers to the arm joints only.
- If using MuJoCo Menagerie MJCF files, use the `.xml` file directly — no URDF conversion needed.
- Ensure inertia parameters are accurate for sensorless force estimation to work reliably.
