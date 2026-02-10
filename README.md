# AdaptiveArm

**无传感器力感知与自适应控制框架** — 让普通协作机械臂（无力矩传感器）具备力估计、负载自适应和主动柔顺能力。

## 这个项目解决什么问题？

工业协作机械臂（cobot）要实现力控，通常需要加装昂贵的六维力/力矩传感器。AdaptiveArm 通过**基于模型的软件算法**，仅利用电机电流和关节编码器信号，就能估计外部接触力——让一台普通机械臂获得类似力传感器的能力。

**核心思路**：机器人运动方程为 `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ_motor + τ_ext`，如果我们精确知道左边的动力学项，就能从电机力矩中反推出外力 `τ_ext`。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        AdaptiveArm 框架                          │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌────────┐  │
│  │  Robot    │───▶│ RobotState│───▶│  Observer   │───▶│Control │  │
│  │ Interface │    │ q,q̇,τ    │    │ (GMO/EKF)  │    │Output  │  │
│  └──────────┘    └───────────┘    └────────────┘    └────────┘  │
│       ▲                                ▲                  │     │
│       │                                │                  │     │
│       │                          ┌─────┴──────┐          │     │
│       │                          │ Dynamics    │          │     │
│       │                          │ M(q),C,g(q)│          │     │
│       │                          └────────────┘          │     │
│       └──────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### 核心数据流

```
Robot ──▶ RobotState ──▶ Observer ──▶ ObserverOutput ──▶ Controller ──▶ ControlOutput ──▶ Robot
 (硬件/仿真)  (q, q̇, τ)    (力估计)    (τ_ext, F_ext)     (阻抗/导纳)    (τ_cmd)        (执行)
```

### Adapter 模式（硬件解耦）

所有算法通过统一的 `RobotInterface` 协议运行，不直接依赖任何具体机械臂或仿真器：

```
RobotInterface (Protocol)
├── SimAdapter        ← MuJoCo 仿真（CPU，精确动力学，开发调试用）
├── IsaacGymAdapter   ← Isaac Gym 仿真（GPU 并行，大规模训练 / sim-to-real）
├── YAMAdapter        ← i2rt YAM 硬件（Phase 2）
└── URAdapter         ← Universal Robots 硬件（Phase 3）

DynamicsModel (Protocol)
├── MuJoCoDynamics    ← MuJoCo 后端（计算 M(q), C(q,q̇), g(q)）
└── PinocchioDynamics ← Pinocchio 后端（Phase 3，可选）
```

### 模块职责

| 模块 | 职责 | 关键类 |
|------|------|--------|
| `core/` | 协议定义、数据类型、配置 | `RobotInterface`, `DynamicsModel`, `RobotState`, `ObserverOutput` |
| `models/` | 机器人模型注册与管理 | `RobotModelInfo`, `get_model()`, `register_model()` |
| `dynamics/` | 刚体动力学计算 | `MuJoCoDynamics`, `CoulombViscousFriction` |
| `estimation/` | 力/力矩观测器 | `MomentumObserver` (GMO), `EKFObserver`, `PINNObserver`, `TransformerObserver`, `CompositeObserver` |
| `sim/` | 仿真环境封装 | `MuJoCoArmEnv`, `IsaacGymArmEnv`, `VirtualForceSensor` |
| `adapters/` | RobotInterface 的具体实现 | `SimAdapter`, `IsaacGymAdapter` |
| `control/` | 柔顺控制器（Phase 2+） | 阻抗控制、导纳控制 |
| `identification/` | 参数辨识（Phase 2+） | 负载估计、摩擦辨识 |
| `tuning/` | 自动参数调优 | `AutoTuner` |
| `monitoring/` | Web 实时监控 | `WebMonitor` |
| `utils/` | 数学工具、信号处理 | `LowPassFilter`, `pseudoinverse` |

## 核心算法：广义动量观测器（GMO）

基于 De Luca & Mattone (2005) 的广义动量方法：

```
p(t) = M(q) · q̇                                    ← 广义动量
β(t) = ∫[τ_motor + τ_passive - C·q̇ - g(q) + r] dt  ← 积分项
r(t) = K_O · (p(t) - β(t))                          ← 残差 → 收敛到 τ_ext
```

**原理**：如果没有外力，`p` 的变化率完全由已知力矩解释；有外力时，`p` 的实际变化会超出预期，残差 `r(t)` 就反映了这个差异。

**实现细节**：
- 使用**梯形积分**（非欧拉），数值精度更高
- `K_O` 为对角增益矩阵，逐关节可调（增益越大收敛越快，但噪声放大也越多）
- 可选低通滤波器抑制高频噪声
- 通过 Jacobian 伪逆将关节力矩转换为笛卡尔力/力矩
- 自动补偿 MuJoCo 的被动阻尼力（`qfrc_passive`）

## 双仿真后端

### MuJoCo（默认 — CPU 精确仿真）

- 内置 6-DOF 机械臂模型，开箱即用
- 精确的接触动力学和刚体动力学
- 适合算法开发、单臂调试、单元测试
- 通过 `VirtualForceSensor` 提供力估计的 ground truth

### Isaac Gym（GPU 并行仿真）

- 数千个环境并行运行，适合 RL 训练和大规模参数扫描
- GPU tensor 运算，与 PyTorch 无缝集成
- 适合 sim-to-real 迁移、域随机化
- 通过 `IsaacGymAdapter` 使用统一的 `RobotInterface` 接口

## 安装

```bash
# 基础安装（MuJoCo 仿真）
pip install adaptivearm

# 开发模式
pip install -e ".[dev]"

# 完整安装（含可视化）
pip install -e ".[all]"

# Isaac Gym 支持（需要先安装 Isaac Gym）
pip install -e ".[isaacgym]"

# 神经网络观测器（PINN/Transformer，需要 PyTorch）
pip install -e ".[nn]"

# Web 监控面板（可选 Flask 依赖）
pip install -e ".[web]"
```

**依赖**: Python 3.10+, NumPy, SciPy, MuJoCo 3.0+

**Isaac Gym**: 需要单独安装 [Isaac Gym Preview](https://developer.nvidia.com/isaac-gym)，并确保 GPU 驱动可用。

**PyTorch**: PINN 和 Transformer 观测器需要 PyTorch 2.0+。Web 监控的核心功能仅使用标准库（http.server）。

## 快速上手

### 示例 1：MuJoCo 仿真 + 重力补偿

```python
import numpy as np
from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.dynamics import MuJoCoDynamics

adapter = SimAdapter()  # 内置 6-DOF 机械臂
dynamics = MuJoCoDynamics(adapter.env.model)

state = adapter.reset(q0=np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0]))

for _ in range(1000):
    state = adapter.get_state()
    g = dynamics.gravity_vector(state.q)
    adapter.send_torque(g)  # 纯重力补偿
```

### 示例 2：GMO 力估计

```python
import numpy as np
from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.dynamics import MuJoCoDynamics
from adaptivearm.estimation import MomentumObserver

adapter = SimAdapter()
dynamics = MuJoCoDynamics(adapter.env.model)

observer = MomentumObserver(
    dynamics=dynamics,
    n_joints=adapter.n_joints,
    dt=adapter.dt,
    gains=np.full(adapter.n_joints, 30.0),  # 观测器增益
    lowpass_cutoff=10.0,                      # 10Hz 低通滤波
)

state = adapter.reset()
observer.reset()

for _ in range(1000):
    state = adapter.get_state()
    output = observer.update(state)

    # output.tau_ext — 估计的外部关节力矩 (Nm)
    # output.wrench_ext — 估计的笛卡尔力/力矩 (N, Nm)

    g = dynamics.gravity_vector(state.q)
    adapter.send_torque(g)
```

### 示例 3：Isaac Gym 并行仿真

```python
import numpy as np
from adaptivearm.adapters.isaacgym import IsaacGymAdapter

# 启动 1024 个并行环境
adapter = IsaacGymAdapter(
    num_envs=1024,
    asset_file="urdf/my_arm.urdf",  # 你的 URDF 文件
)

states = adapter.reset()  # 批量重置
# states.q.shape = (1024, n_joints)

for _ in range(1000):
    states = adapter.get_state()
    tau = compute_control(states)   # 你的控制逻辑
    adapter.send_torque(tau)         # 批量发送力矩
```

## 运行示例脚本

```bash
# MuJoCo 重力补偿仿真
python examples/01_quickstart_simulation.py

# GMO 力估计 demo（估计值 vs 真实值对比）
python examples/02_momentum_observer_demo.py

# PINN 力观测器训练 + 推理 demo
python examples/08_pinn_observer_demo.py

# Transformer 力观测器训练 + 推理 demo
python examples/09_transformer_observer_demo.py

# 自动增益优化 demo
python examples/10_auto_tuner_demo.py

# 自定义模型加载 demo
python examples/11_custom_model_demo.py
```

## 项目结构

```
src/adaptivearm/
├── core/              # 协议定义、数据类型、配置
│   ├── interfaces.py  #   RobotInterface, DynamicsModel (Protocol)
│   ├── robot_state.py #   RobotState 数据容器
│   ├── types.py       #   ObserverOutput, ControlOutput
│   └── config.py      #   AdaptiveArmConfig
├── models/            # 机器人模型文件 + 注册表
│   ├── __init__.py          # RobotModelInfo + register/get/list
│   ├── default_6dof/        # 内置 6-DOF 臂 (MJCF)
│   ├── ur5e/                # UR5e 占位（用户添加 URDF）
│   └── panda/               # Panda 占位（用户添加 URDF）
├── dynamics/          # 动力学计算
│   ├── mujoco_dynamics.py   # MuJoCo 后端: M(q), C(q,q̇), g(q)
│   └── friction_models.py   # Coulomb + 粘性摩擦模型
├── estimation/        # 力估计算法
│   ├── base_observer.py          # 观测器抽象基类
│   ├── momentum_observer.py      # GMO 广义动量观测器
│   ├── ekf_observer.py           # EKF 扩展卡尔曼滤波观测器
│   ├── collision_detector.py     # 碰撞检测器
│   ├── composite_observer.py     # 多观测器融合
│   ├── neural_base.py            # 神经网络观测器基类
│   ├── pinn_observer.py          # PINN 物理信息神经网络观测器
│   └── transformer_observer.py   # Transformer 序列观测器
├── sim/               # 仿真环境
│   ├── mujoco_env.py        # MuJoCo 环境（含内置 6-DOF 臂）
│   ├── isaacgym_env.py      # Isaac Gym GPU 并行环境
│   └── virtual_sensor.py    # 虚拟力传感器（ground truth）
├── adapters/          # RobotInterface 实现
│   ├── sim/           #   MuJoCo SimAdapter
│   └── isaacgym/      #   Isaac Gym IsaacGymAdapter
├── control/           # 柔顺控制器
│   ├── impedance.py          # 阻抗控制器
│   ├── admittance.py         # 导纳控制器
│   ├── adaptive_impedance.py # 自适应阻抗控制器
│   └── safety_monitor.py     # 安全监控
├── identification/    # 参数辨识
│   ├── payload_estimator.py  # 负载估计
│   └── friction_identifier.py # 摩擦辨识
├── tuning/            # 自动调优
│   └── auto_tuner.py         # 自动增益优化
├── monitoring/        # Web 监控
│   └── web_monitor.py        # HTTP 实时仪表盘
└── utils/             # 数学工具、信号处理
    ├── math_utils.py        # 伪逆、反对称矩阵、角度归一化
    └── signal_processing.py # 一阶 IIR 低通滤波器
tests/                 # 单元测试 + 集成测试
examples/              # 示例脚本
docs/tutorials/        # 教程文档
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码检查
ruff check src/

# 类型检查
mypy src/
```

## 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 核心框架 + GMO + MuJoCo/IsaacGym 仿真 | ✅ 已完成 |
| Phase 2 | YAM 硬件适配 + 阻抗/导纳控制 + 碰撞检测 | ✅ 已完成 |
| Phase 3 | UR 适配器 + Pinocchio 后端 + EKF 观测器 | ✅ 已完成 |
| Phase 4 | PINN/Transformer 力估计 + AutoTuner + Web 监控 | ✅ 已完成 |

## 商业模式

| 免费开源 (MIT) | 付费服务 |
|---------------|---------|
| 全部算法代码 | AutoTuner 自动增益调优 |
| 全部适配器（MuJoCo / Isaac Gym / 硬件） | 系统辨识向导 |
| 仿真环境 + 文档 + 示例 | Web 实时监控面板 |
| | 定制适配器开发 + 技术支持 |

## License

MIT
