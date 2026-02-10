# UR5e Model

Place your UR5e URDF and mesh files in this directory.

## Required files

```
ur5e/
├── ur5e.urdf          # Main URDF file
└── meshes/            # Collision and visual meshes
    ├── collision/
    └── visual/
```

## Where to get the URDF

- **Official**: [Universal Robots ROS2 description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
- **ROS package**: `ros2 pkg prefix ur_description`

## Registration

After placing the files, register the model in your code:

```python
from pathlib import Path
from adaptivearm.models import RobotModelInfo, register_model

register_model(RobotModelInfo(
    name="ur5e",
    model_path=Path(__file__).parent / "ur5e.urdf",
    n_joints=6,
    ee_site_name="ee_site",      # Must match your URDF site name
    ee_body_name="tool0",        # UR5e end-effector link
    description="Universal Robots UR5e 6-DOF collaborative robot",
))
```

## Usage

```python
from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.dynamics import MuJoCoDynamics

adapter = SimAdapter(model_name="ur5e")
dynamics = MuJoCoDynamics(adapter.env.model)
```

## Notes

- MuJoCo can load URDF files directly and converts them internally to MJCF.
- Make sure the URDF includes `<inertial>` tags with accurate mass and inertia values for correct dynamics computation.
- You may need to add a `<site>` element in the URDF for the end-effector if one is not present.
