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
| [transport/client.py](../src/transport/client.py) | `TrainingClient`：仿真协议、请求/回合/步编号及未确认请求重取 |
| [transport/processes.py](../src/transport/processes.py) | `training_worker`：启动仿真进程，分配独立 socket，设置动态库路径并回收进程 |
| [transport/vision_bridge.py](../src/transport/vision_bridge.py) | `VisionBridge` / `vision_worker`：桥接通信、策略询问、动作与发布确认 |
| [environment/warmup.py](../src/environment/warmup.py) | `WarmupSession`：禁射搜索、连续新帧确认、就绪或超时 |
| [environment/evaluation.py](../src/environment/evaluation.py) | `EvaluationSession`：预热、正式窗口和尾部结算生命周期 |
| [environment/fire.py](../src/environment/fire.py) | `FireEnv`：规则两动作/联合九动作 Gym 采样、时间截断和独立进程生命周期；静止/旋转子类提供场景规则 |
| [training/environment.py](../src/training/environment.py) | 任务与场景序列适配、Monitor 和单环境 DummyVecEnv，覆盖自动 Reset |
| [training/models.py](../src/training/models.py) | 模型构建/加载、观测契约及配置指纹、原子检查点 |
| [training/train.py](../src/training/train.py) | 训练预算、完整更新、日志、周期保存和运行生命周期 |
| [policy/observations.py](../src/policy/observations.py) | `encode` / `TensorPolicy`：语义字段编码、8 个控制周期的历史和有效性标记 |
| [scoring/window.py](../src/scoring/window.py) | `WindowScore`：关联实际出膛与弹丸结果，计算窗口归属伤害 |

`EvaluationSession` 接管仿真和桥接后，不应再从外部单独调用它们的步进接口，以免破坏控制、发布确认和计分时序。

`StaticFireEnv` 和 `RotationFireEnv` 通过公共 `FireEnv` 独占各自进程对。`reset()` 完成禁射预热后启动桥接训练模式，
`prepare()` 在当周期策略回调处暂停并返回观测，`step(action)` 通过 `submit()` 恢复该回调，
随后提交命令、推进一次物理并确认发布，再准备下一观测。无回调时缓存已计算的禁射命令，
不跳过该周期。准备观测不推进物理，取消待决策周期不提交动作；取消后桥接必须 Reset。
旧 `VisionBridge.step(response, policy)` 是这些操作的同步封装，原有评估调用方式不变。

训练模式没有评估截止时间开关。30 秒上限由 Gym 管理，截止时准备真实末观测后取消未提交
周期，返回 `truncated=True`，不额外结算或自动 Reset。环境只给实际发生的伤害奖励，
由训练器使用末观测价值自举；评估仍单独用 `WindowScore` 做实际出膛窗口归属和尾部结算。

## PPO 与未来循环策略

当前配置显式使用 `algorithm=maskable_ppo`、`policy_kind=mlp`。模型构建/加载集中在
`training.models`；Gym 和传输不依赖 PyTorch。`MultiInputPolicy` 在特征提取内部展平历史，
环境输出 fire_only 的 `[8,90]` 或 joint 的 `[8,107]`，不额外做观测或奖励归一化。场景种子独立于 PPO seed。旋转任务在每次自动 Reset 采样下一场景；
静止任务重用场景种子和参数，也会重复测量噪声。单参考场景评估不代表泛化效果。

单环境 `DummyVecEnv` 保留终止观测和 `TimeLimit.truncated`，MaskablePPO 负责价值自举。
Monitor 在此之前记录原始伤害奖励，不把价值估计记成实际伤害。训练端每次只请求一个完整
rollout 的 `learn(reset_num_timesteps=False)`，保持当前观测和优化器连续；固定学习率与
clip 参数没有分段调度问题。更新结束后才记录损失并允许保存检查点。

当前 MaskablePPO 不原生支持循环策略。将来接入 GRU 需要同时处理序列 rollout、隐藏状态、
回合起点、目标重建和动作掩码，不能只替换 MLP 层，也不能把八行历史当成跨 rollout 的记忆。
这些能力应放在训练/策略层，新增独立的模型与训练实现；当前不提供空的 GRU 类。
检查点的策略类型和观测契约用于阻止不兼容加载。

## 分析与验证工具

`tools/training/` 单向依赖运行库和训练接口；`src/training/` 保留配置、环境包装、
模型管理与 PPO 入口，不导入工具。训练结束后按需独立执行分析，训练命令不自动启动评估或窗口。

| 文件 | 作用 |
| --- | --- |
| [tools/training/analysis.py](../tools/training/analysis.py) | 训练结束后的检查点评估、最佳模型选优、独立分析产物及双窗口编排 |
| [tools/training/report.py](../tools/training/report.py) | 从日志和评估结果生成离线交互曲线与模型对比报告 |
| [tools/training/view.py](../tools/training/view.py) | 单模型确定性评估、真值回放记录及单／多回放进程管理 |
| [tools/training/manual.py](../tools/training/manual.py) | 原生窗口 JSONL 操作、手动 Gym 步进、原始奖励日志及进程清理 |
| [tools/training/diagnose.py](../tools/training/diagnose.py) | 短期 PPO 更新的学习信号诊断 |
| [tools/training/diagnostic_report.py](../tools/training/diagnostic_report.py) | 诊断数据关联与可视化报告 |

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
| PPO 新训练射击决策 | 默认随机 50–100 ms，独立时钟门控；原控制步不变 |
| 物理 | 默认 1 ms、2 子步；训练入口支持 0.5/1/2 ms |
| 云台积分 | 最多 1 ms 积分间隔 |
| 合成检测 | 30 Hz，按物理步边界采样 |
| 自身裁判 | 10 Hz，包含独立采样时间 |
| 自身武器机械状态 | 每物理步更新，每控制步读取；独立于裁判采样 |

训练推进使用仿真时间；等待 Python 策略时世界不继续演化。资产加载和通信超时仍使用宿主机时间。反馈合成时间戳为 `10^18 + 回合仿真纳秒`，不能当作真实日期。控制会复用最近估计，没有新检测帧时不重复更新预测器。

策略输入来自目标估计、自身反馈和自身裁判数据。种子、实验倒计时、真实目标状态、实际伤害和完整事件账本不送给策略。`EvaluationSession.advance()` 的完整返回对象包含评估真值，不能整体作为网络输入。

Python 仅合并自身武器通道的机械间隔、供弹忙和枪口无效三个位，避免把 50 ms 的机械冷却锁在 100 ms 裁判采样周期里。热量、生命和弹量继续使用裁判快照；物理后端仍最终裁决每次出膛。合成检测的整板可见性测试排除小弹丸，防止开火本身造成系统性的虚假暂时丢失。

当前己方底盘固定，目标支持静止、正弦平移或匀速旋转。合成观测的几何可见性近似不等价于真实渲染检测结果，限制见[当前状态](development.md)。

## 联合策略的数据流

`environment.action_mode` 与静止/旋转任务独立。`make_policy` 创建共用 `DecisionPolicy`
生命周期的两动作或联合策略；Gym、独立评估都在预热后启动时钟，每个已完成物理步后推进一次。
`TensorPolicy` 根据 v1/v2 schema 编码历史，目标代次变化只清历史，不重置射击时钟。

联合模式通过桥接 `begin_training(policy_mode="nine")` 获取所有有效弹道候选，跳过
`FireOnlyPolicyAdapter`。RL 输出槽位和射击请求，C++ 核心负责该槽位的弹道、MPC 和执行门控。
观测中的当前/命中朝向及选板时长均由估计和控制历史生成，真值只写入日志/回放。
`tools.training.compare` 先确定共同合法出生，再用相同 seed 和场景分别重置两策略，不重采样失败场景。
