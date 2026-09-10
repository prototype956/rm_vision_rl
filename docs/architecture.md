# 架构与代码职责

## 当前闭环

```text
Rust 同步仿真 daedalus_training
  │ Unix socket：二维检测帧、自身反馈、裁判采样
  ▼
Python 会话管理
  │ JSONL 标准输入/输出：只转发控制需要的字段
  ▼
C++ rmvision-rl-bridge
  ├─ ArmorPnp → ArmorPredictor
  ├─ ControlSession → 候选装甲板与动作掩码
  ├─ 规则策略，或同步询问 Python 回调策略
  └─ 弹道解算 → TinyMPC → 云台命令与开火脉冲
  │
  ▼
Python 提交 Advance → 仿真推进 10 ms → 确认命令发布
  └─ 独立事件和真值送入评估计分，不进入策略输入
```

无渲染模式生成带噪二维角点，随后使用真实 PnP 和预测器。它不运行图像检测网络，也不把真实目标位置直接当作预测器输出。带图像的 Talos/视觉主程序是另一条链路，不由当前 Python 会话启动。

## Python 运行库

| 文件 | 作用 |
| --- | --- |
| [transport/client.py](../rmvision_rl/transport/client.py) | `TrainingClient`：仿真协议、请求/回合/步编号及未确认请求重取 |
| [transport/processes.py](../rmvision_rl/transport/processes.py) | `training_worker`：启动仿真进程，分配独立 socket，设置动态库路径并回收进程 |
| [transport/vision_bridge.py](../rmvision_rl/transport/vision_bridge.py) | `VisionBridge` / `vision_worker`：桥接通信、策略询问、动作与发布确认 |
| [environment/warmup.py](../rmvision_rl/environment/warmup.py) | `WarmupSession`：禁射搜索、连续新帧确认、就绪或超时 |
| [environment/evaluation.py](../rmvision_rl/environment/evaluation.py) | `EvaluationSession`：预热、正式窗口和尾部结算生命周期 |
| [policy/observations.py](../rmvision_rl/policy/observations.py) | `encode` / `TensorPolicy`：语义字段编码、8 个控制周期的历史和有效性标记 |
| [scoring/window.py](../rmvision_rl/scoring/window.py) | `WindowScore`：关联实际出膛与弹丸结果，计算窗口归属伤害 |

`EvaluationSession` 接管仿真和桥接后，不应再从外部单独调用它们的步进接口，以免破坏控制、发布确认和计分时序。

## C++ 桥接

| 文件 | 作用 |
| --- | --- |
| [bridge.cpp](../native/vision_bridge/bridge.cpp) | 处理 JSONL 操作，适配检测、反馈和裁判，调用视觉核心并输出命令 |
| [policy_wire.hpp](../native/vision_bridge/policy_wire.hpp) / [policy_wire.cpp](../native/vision_bridge/policy_wire.cpp) | 序列化策略观测和解析同步动作应答 |
| [CMakeLists.txt](../native/vision_bridge/CMakeLists.txt) | 从 `VISION_ROOT` 编译 PnP、预测器、火控、轨迹规划及 TinyMPC |

桥接不需要相机 SDK 或 OpenVINO 检测推理，但需要 OpenCV、Eigen、Ceres 等依赖。视觉源码改变后，需要重新构建桥接才能使用新实现。

## 时间与信息边界

| 子系统 | 当前节拍 |
| --- | --- |
| 控制与 Advance | 10 ms / 100 Hz |
| 物理 | 默认 1 ms、2 子步；训练入口支持 0.5/1/2 ms |
| 云台积分 | 最多 1 ms 积分间隔 |
| 合成检测 | 30 Hz，按物理步边界采样 |
| 自身裁判 | 10 Hz，包含独立采样时间 |

训练推进使用仿真时间；等待 Python 策略时世界不继续演化。资产加载和通信超时仍使用宿主机时间。反馈合成时间戳为 `10^18 + 回合仿真纳秒`，不能当作真实日期。控制会复用最近估计，没有新检测帧时不重复更新预测器。

策略输入来自目标估计、自身反馈和自身裁判数据。种子、实验倒计时、真实目标状态、实际伤害和完整事件账本不送给策略。`EvaluationSession.advance()` 的完整返回对象包含评估真值，不能整体作为网络输入。

当前己方底盘固定，目标支持静止、正弦平移或匀速旋转。合成观测的几何可见性近似不等价于真实渲染检测结果，限制见[当前状态](development.md)。
