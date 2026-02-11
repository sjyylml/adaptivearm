<p align="center">
  <h1 align="center">OpenForce</h1>
  <p align="center">
    <strong>Physics-Aware Sensorless Force Estimation for Robot Arms</strong><br>
    <em>Bridging VLA models and real-world manipulation — no F/T sensor required.</em>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#中文说明">中文</a>
  </p>
</p>

---

**OpenForce** is an open-source Python framework that estimates external contact forces on collaborative robot arms using only motor currents and joint encoders. It serves as a **physics execution layer** between high-level AI models (VLA, foundation models) and low-level motor control — translating semantic intent into physically consistent joint torques.

> **Status**: All algorithms are validated in simulation (MuJoCo). Hardware adapter interfaces exist for YAM and UR robots. Real-robot validation is in progress.

## Why OpenForce?

The embodied AI stack has a gap: VLA models output high-level actions, but robots need precise, physics-aware torque commands at 200Hz+. Force/torque sensors cost $2k-10k per joint and add integration complexity. OpenForce fills this gap with model-based estimation:

```
VLA / Foundation Model
        ↓  (semantic actions)
   ┌─────────────────────────────┐
   │        OpenForce            │  ← physics execution layer
   │  force estimation + control │
   └─────────────────────────────┘
        ↓  (joint torques @ 200Hz+)
   Robot Hardware (any arm)
```

**Core equation**: `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ_motor + τ_ext`

If we know the dynamics terms precisely, we can recover external forces `τ_ext` from motor torques alone.

## Features

| Category | What's Included |
|----------|----------------|
| **Estimation** | Generalized Momentum Observer (GMO), Extended Kalman Filter, PINN, Transformer observer, composite fusion |
| **Control** | Impedance, admittance, adaptive impedance, collision detection, safety monitor |
| **Simulation** | MuJoCo (CPU, precise dynamics), Isaac Gym (GPU, massively parallel) |
| **Hardware** | Adapter protocol for any robot. Built-in: SimAdapter, YAMAdapter, URAdapter |
| **Dynamics** | MuJoCo backend, Pinocchio backend (optional) |
| **Identification** | Payload estimation, friction identification, AutoTuner |

## Architecture

All algorithms interact with robots through the `RobotInterface` protocol — hardware-agnostic by design.

```
Robot → RobotState → Observer → ObserverOutput → Controller → ControlOutput → Robot
        (q, q̇, τ)    (GMO/EKF)   (τ_ext, F_ext)   (impedance)   (τ_cmd)
```

### Adapter Pattern

```
RobotInterface (Protocol)
├── SimAdapter        — MuJoCo simulation (development & testing)
├── IsaacGymAdapter   — Isaac Gym (GPU parallel, RL training)
├── YAMAdapter        — i2rt YAM hardware (CAN bus)
└── URAdapter         — Universal Robots (RTDE)

DynamicsModel (Protocol)
├── MuJoCoDynamics    — M(q), C(q,q̇), g(q) via MuJoCo
└── PinocchioDynamics — via Pinocchio (optional)
```

## Installation

```bash
# Basic (MuJoCo simulation)
pip install -e .

# Development
pip install -e ".[dev]"

# Full (all optional backends)
pip install -e ".[all]"

# Specific extras
pip install -e ".[nn]"         # PINN/Transformer (requires PyTorch)
pip install -e ".[isaacgym]"   # Isaac Gym support
pip install -e ".[web]"        # Web monitoring dashboard
```

**Requirements**: Python 3.10+, NumPy, SciPy, MuJoCo 3.0+

## Quick Start

### Force Estimation with GMO

```python
import numpy as np
from openforce.adapters.sim import SimAdapter
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation import MomentumObserver

adapter = SimAdapter()
dynamics = MuJoCoDynamics(adapter.env.model)

observer = MomentumObserver(
    dynamics=dynamics,
    n_joints=adapter.n_joints,
    dt=adapter.dt,
    gains=np.full(adapter.n_joints, 30.0),
    lowpass_cutoff=10.0,
)

state = adapter.reset()
observer.reset()

for _ in range(1000):
    state = adapter.get_state()
    output = observer.update(state)
    # output.tau_ext  — estimated external joint torques (Nm)
    # output.wrench_ext — estimated Cartesian wrench (N, Nm)
    g = dynamics.gravity_vector(state.q)
    adapter.send_torque(g)
```

### Hardware (YAM Example)

```python
import numpy as np
from openforce.adapters.yam import YAMAdapter
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation import MomentumObserver
from openforce.models import get_model

info = get_model("yam")
dynamics = MuJoCoDynamics.from_xml(info.model_path)

with YAMAdapter() as yam:
    observer = MomentumObserver(
        dynamics=dynamics, n_joints=6, dt=0.004, gains=np.full(6, 20.0)
    )
    observer.reset()
    for _ in range(5000):
        state = yam.get_state()
        output = observer.update(state)
```

### Run Examples

```bash
python examples/01_quickstart_simulation.py    # Gravity compensation
python examples/02_momentum_observer_demo.py   # GMO estimation vs ground truth
python examples/03_impedance_control.py        # Impedance control
python examples/04_collision_detection.py      # Collision detection
python examples/08_pinn_observer_demo.py       # PINN observer
python examples/10_auto_tuner_demo.py          # Auto gain tuning
```

## Core Algorithm: Generalized Momentum Observer

Based on De Luca & Mattone (2005):

```
p(t) = M(q) · q̇                                     — generalized momentum
β(t) = ∫[τ_motor + τ_passive - C·q̇ - g(q) + r] dt   — integral term
r(t) = K_O · (p(t) - β(t))                           — residual → converges to τ_ext
```

Implementation: trapezoidal integration, per-joint diagonal gain matrix `K_O`, optional low-pass filtering, automatic Jacobian-based Cartesian wrench conversion.

## Project Structure

```
src/openforce/
├── core/           — Protocol definitions, data types (RobotInterface, RobotState)
├── models/         — Robot model registry (MJCF/URDF, built-in: 6-DOF, YAM)
├── dynamics/       — Rigid-body dynamics (MuJoCoDynamics, PinocchioDynamics)
├── estimation/     — Force observers (GMO, EKF, PINN, Transformer, composite)
├── sim/            — Simulation environments (MuJoCo, Isaac Gym)
├── adapters/       — RobotInterface implementations (Sim, IsaacGym, YAM, UR)
├── control/        — Compliant controllers (impedance, admittance, safety)
├── identification/ — Parameter identification (payload, friction)
├── tuning/         — AutoTuner (gain optimization)
├── monitoring/     — Web dashboard (real-time visualization)
└── utils/          — Math helpers, signal processing
tests/              — Unit & integration tests
examples/           — Example scripts (01-12)
```

## Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| Phase 1 | Core framework + GMO + MuJoCo simulation | Done (sim validated) |
| Phase 2 | Compliant control + collision detection | Done (sim validated) |
| Phase 3 | UR/YAM adapters + Pinocchio + EKF | Done (sim validated) |
| Phase 4 | PINN/Transformer + AutoTuner + Web monitor | Done (sim validated) |
| Phase 5 | Real-robot validation + sim-to-real | In progress |
| Phase 6 | Cross-hardware identification database | Planned |

## Development

```bash
pip install -e ".[dev]"
pytest tests/           # Run tests
ruff check src/         # Lint
mypy src/               # Type check
```

## Contributing

We welcome contributions — especially hardware adapters for new robot arms and real-robot validation data. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — free to use, modify, and distribute.

---

# 中文说明

**OpenForce** 是一个开源的无传感器力估计与自适应控制框架，适用于协作机械臂。仅利用电机电流和关节编码器信号即可估计外部接触力，无需力/力矩传感器。

### 定位

OpenForce 作为**物理执行层**，向上对接 VLA / 大模型的语义指令，向下屏蔽异构硬件差异，输出物理一致的关节力矩。

### 核心能力

- **力估计算法**: GMO 广义动量观测器、EKF 扩展卡尔曼滤波、PINN 物理信息神经网络、Transformer 序列观测器
- **柔顺控制**: 阻抗控制、导纳控制、自适应阻抗、碰撞检测
- **仿真后端**: MuJoCo（CPU 精确仿真）、Isaac Gym（GPU 大规模并行）
- **硬件适配**: 通过 `RobotInterface` 协议支持任意机械臂，内置 YAM、UR 适配器
- **参数辨识**: 负载估计、摩擦辨识、AutoTuner 自动增益优化

### 当前状态

所有算法均在 MuJoCo 仿真环境中验证通过。硬件适配器接口已实现（YAM、UR），真机验证正在进行中。

### 快速上手

```bash
pip install -e ".[dev]"
python examples/01_quickstart_simulation.py
python examples/02_momentum_observer_demo.py
```

详细使用方法请参考上方英文文档中的 [Quick Start](#quick-start) 部分。

### 商业模式

| 开源 (MIT) | 付费服务 |
|-----------|---------|
| 全部算法与适配器代码 | 定制硬件适配与参数调优 |
| 仿真环境 + 示例 + 文档 | 跨硬件系统辨识数据库 |
| 社区支持 | 具身数据预处理工具链 |
