# YAM 真机测试指南

## 前置条件

- YAM 机械臂已上电，USB-CAN 适配器已连接
- 臂周围有足够安全空间，无障碍物
- 已安装 conda 环境 `gmr`（Python 3.10 + torch + pinocchio + i2rt）

---

## 第一步：启动 CAN 接口

```bash
# 查看 CAN 设备是否识别
ip link show | grep can

# 启动 CAN 接口（波特率 1000000）
sudo ip link set can0 up type can bitrate 1000000

# 验证 CAN 接口状态（应该显示 UP）
ip link show can0
```

如果有多个 CAN 设备，用 `can1` 等替换 `can0`，后续命令相应修改。

如果 CAN 接口报错，重置后重试：

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

---

## 第二步：激活环境并确认依赖

```bash
conda activate gmr

# 确认关键包版本
python -c "import openforce; print('openforce ok')"
python -c "import i2rt; print('i2rt ok')"
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import torch; print('torch', torch.__version__)"
```

如果 `openforce` 或 `i2rt` 导入失败：

```bash
cd /home/limenglian/data/claudetest && pip install -e ".[dev]"
cd /home/limenglian/data/i2rt && pip install -e .
```

---

## 第三步：（可选）关闭电机超时保护

YAM 出厂默认 400ms 超时——如果电机超过 400ms 没收到指令会自动断电进入阻尼模式。
测试期间如果遇到电机突然掉电，可以关闭超时：

```bash
cd /home/limenglian/data/i2rt
python -m i2rt.motor_config_tool.set_timeout --channel can0
```

测试结束后建议恢复超时保护：

```bash
python -m i2rt.motor_config_tool.set_timeout --channel can0 --timeout
```

---

## 第四步：快速连接验证

先用 i2rt SDK 单独验证连接是否正常：

```bash
cd /home/limenglian/data/i2rt
python -c "
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType
robot = get_yam_robot(channel='can0', gripper_type=GripperType.NO_GRIPPER, zero_gravity_mode=True)
print('连接成功！')
obs = robot.get_observations()
print(f'关节位置: {obs[\"joint_pos\"]}')
print(f'关节速度: {obs[\"joint_vel\"]}')
print(f'关节力矩: {obs[\"joint_eff\"]}')
import time; time.sleep(3)
robot.close()
print('已安全断开')
"
```

此时臂应进入零重力模式（只有重力补偿，可自由拖动）。确认无异常后 Ctrl+C 或等待自动退出。

---

## 第五步：运行完整测试脚本

```bash
cd /home/limenglian/data/claudetest
python examples/12_yam_hardware_test.py --channel can0
```

脚本包含 6 个阶段，每个阶段之间会暂停等你按 Enter 确认：

| 阶段 | 内容 | 预期结果 |
|------|------|---------|
| 1 | 连接 & 状态读取 | 打印 6 个关节的 q, qd, tau_motor，读取频率约 50Hz |
| 2 | 模型注册表 & 动力学 | 打印 YAM 模型信息，计算 g(q) 和 M(q)，质量矩阵正定 |
| 3 | 重力补偿对比 | 10 次采样，对比框架计算的 g(q) 与实际 tau_motor 的差异 |
| 4 | GMO 力估计 (30s) | 用手推末端，观察 tau_ext 是否响应（推时数值增大，松开归零）|
| 5 | EKF 力估计 (30s) | 同上，对比 EKF 和 GMO 的效果 |
| 6 | 碰撞检测 (30s) | 轻拍臂体，观察是否检测到碰撞事件 |

如果只想跳到某个阶段（比如跳过前两步）：

```bash
python examples/12_yam_hardware_test.py --channel can0 --skip-to 3
```

随时按 Ctrl+C 中止当前阶段或整个测试。

---

## 第六步：单独测试各功能

如果需要单独调试某个功能，可以在 Python 交互模式下操作：

### 6.1 单独测试 YAMAdapter 连接

```python
import numpy as np
from openforce.adapters.yam import YAMAdapter, YAMConfig

config = YAMConfig(channel="can0", zero_gravity_mode=True)

with YAMAdapter(config) as yam:
    state = yam.get_state()
    print(f"q = {state.q}")
    print(f"qd = {state.qd}")
    print(f"tau = {state.tau_motor}")
    print(f"温度 = {yam.get_temperatures()}")
```

### 6.2 单独测试 GMO 观测器

```python
import time
import numpy as np
import mujoco
from openforce.adapters.yam import YAMAdapter, YAMConfig
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation import MomentumObserver
from openforce.models import get_model

info = get_model("yam")
model = mujoco.MjModel.from_xml_path(str(info.model_path))
dynamics = MuJoCoDynamics(model)

config = YAMConfig(channel="can0", zero_gravity_mode=True)

with YAMAdapter(config) as yam:
    observer = MomentumObserver(
        dynamics=dynamics,
        n_joints=6,
        dt=0.004,
        gains=np.full(6, 20.0),   # 观测器增益，可调
        lowpass_cutoff=5.0,        # 低通滤波截止频率 Hz，可调
    )
    observer.reset()

    # 持续读取并打印
    for i in range(5000):  # 20 秒
        state = yam.get_state()
        output = observer.update(state)
        if i % 50 == 0:  # 每 0.2 秒打印一次
            print(f"tau_ext = {np.array2string(output.tau_ext, precision=2)}, "
                  f"|tau_ext| = {np.linalg.norm(output.tau_ext):.3f}")
        time.sleep(0.004)
```

### 6.3 单独测试碰撞检测

```python
import time
import numpy as np
import mujoco
from openforce.adapters.yam import YAMAdapter, YAMConfig
from openforce.dynamics import MuJoCoDynamics
from openforce.estimation import MomentumObserver
from openforce.estimation.collision_detector import CollisionDetector
from openforce.models import get_model

info = get_model("yam")
model = mujoco.MjModel.from_xml_path(str(info.model_path))
dynamics = MuJoCoDynamics(model)

config = YAMConfig(channel="can0", zero_gravity_mode=True)

with YAMAdapter(config) as yam:
    observer = MomentumObserver(
        dynamics=dynamics, n_joints=6, dt=0.004,
        gains=np.full(6, 20.0), lowpass_cutoff=5.0,
    )
    detector = CollisionDetector(
        thresholds=np.array([3.0, 3.0, 3.0, 2.0, 2.0, 2.0]),  # Nm，可调
        holdoff_time=0.5,
    )
    observer.reset()
    detector.reset()

    for i in range(5000):
        state = yam.get_state()
        output = observer.update(state)
        event = detector.update(output.tau_ext, state.timestamp)
        if event.in_collision:
            print(f"*** 碰撞! severity={event.severity:.2f}")
        time.sleep(0.004)
```

---

## 参数调优建议

测试过程中可能需要调整的关键参数：

| 参数 | 位置 | 说明 | 建议范围 |
|------|------|------|---------|
| `gains` | MomentumObserver | 观测器增益，越大收敛越快但噪声越大 | 10~50 |
| `lowpass_cutoff` | MomentumObserver | 低通滤波截止频率 (Hz) | 3~15 |
| `thresholds` | CollisionDetector | 碰撞触发阈值 (Nm) | 1~5 |
| `torque_scale` | YAMConfig | 输出力矩缩放系数 | 0.5~1.0 |

调优思路：
1. 先用 Stage 3 观察静止时 `tau_motor - g(q)` 的残差大小，这就是系统噪声底
2. GMO 的 `gains` 从 20 开始，如果噪声太大就降低或降低 `lowpass_cutoff`
3. 碰撞检测的 `thresholds` 应设为噪声底的 3~5 倍

---

## 常见问题

### CAN 连接失败

```
CanError: Failed to transmit
```

检查：CAN 线是否松动、波特率是否匹配（1000000）、设备是否上电。

### 电机报错后无法重新连接

```bash
# 重置 CAN 接口
sudo ip link set can0 down && sudo ip link set can0 up type can bitrate 1000000
```

### 关节超限报错

```
RuntimeError: Joint limit violation detected
```

将臂手动移回安全范围内，重新上电后再连接。

### 重力补偿力矩异常大

检查臂是否被卡住或碰到障碍物。如果 `g(q)` 计算值 > 20 Nm，i2rt SDK 会自动抛出异常保护。

---

## 测试完成后

```bash
# 恢复电机超时保护（如果之前关闭了）
cd /home/limenglian/data/i2rt
python -m i2rt.motor_config_tool.set_timeout --channel can0 --timeout

# 关闭 CAN 接口
sudo ip link set can0 down
```
