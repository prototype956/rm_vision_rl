# RM Vision RL

RoboMaster 火控强化学习项目。策略选择当前目标的装甲板并决定何时请求开火；PnP、目标预测、弹道解算和 TinyMPC 复用相邻视觉仓库的实现。项目研究的是给定视觉估计和自身反馈下的火控决策，训练环境使用带噪二维角点，不运行图像检测网络。

服务器完成仿真、视觉估计、完整回合采集、循环 PPO 更新和评估。NUC 加载导出的 TorchScript Actor，并在本地维护 GRU 隐藏状态。串口链路和固件接入仍待实现。

## 从哪里开始

| 需要完成的工作 | 文档 |
| --- | --- |
| 构建、训练、恢复、手动调试和回放 | [构建与运行](docs/running.md) |
| 带第一人称和第三人称画面的 MCAP | [离线渲染回放](docs/foxglove_rendered_replay.md) |
| 理解模块职责、控制时序和信息边界 | [架构](docs/architecture.md) |
| 理解阶段、奖励、固定场景与晋级 | [课程 v2](docs/curriculum_training_v2.md) |
| 使用平移分层、传统火控校准和全路径可见性准入 | [课程 v3](docs/curriculum_training_v3.md) |
| 接入单步观测、管理隐藏状态、导出和部署 | [持续记忆 GRU](docs/persistent_gru.md) |
| 查询通信、动作、计分和检查点契约 | [接口与配置](docs/interfaces.md) |
| 查询车型、资源和脚本运动能力 | [多车型环境](docs/multirobot_environment.md) |
| 选择工具或确认实现边界 | [辅助工具](tools/README.md)、[开发状态](docs/development.md) |

## 配置与训练入口

`config/training/` 包含四份预设：步兵、哨兵各有 v2 和 v3。训练 CLI 的默认值仍是步兵 v2；配置文件存在不代表对应版本已通过正式发布门槛。

| 课程 | 步兵配置 | 哨兵配置 | 使用条件 |
| --- | --- | --- | --- |
| v2 | [infantry v2](config/training/rmul_2027_infantry_curriculum_v2.json) | [sentry v2](config/training/rmul_2027_sentry_curriculum_v2.json) | 固定阶段与门槛 |
| v3，发布修订 3 | [infantry v3](config/training/rmul_2027_infantry_curriculum_v3.json) | [sentry v3](config/training/rmul_2027_sentry_curriculum_v3.json) | 先预检、校准，再按 v3 流程迁移和资格验证 |

四份预设均使用 `recurrent_maskable_ppo/asymmetric_gru`、v5 单步观测、九动作和热锁奖励。默认单 GPU、128 环境、每角色累计十亿步；控制周期 10 ms，射击机会间隔 50–100 ms，单回合最多 30 s。十亿步是预算目标，战斗与热锁是否达标分别报告。

完成[依赖安装和原生构建](docs/running.md#依赖与构建)后，从本仓库根目录运行：

```bash
# 默认步兵 v2；这是长期训练入口。
python -m src.training.train --output-dir artifacts/training/infantry-v2
python -m src.training.train \
  --config config/training/rmul_2027_sentry_curriculum_v2.json \
  --output-dir artifacts/training/sentry-v2
# 使用新的输出目录，完成原累计预算的剩余部分。
python -m src.training.train \
  --resume artifacts/training/infantry-v2/latest.zip \
  --output-dir artifacts/training/infantry-v2-resumed
```

新训练可显式覆盖预算和环境数；课程恢复保持原累计预算、环境数与课程配置。采集须完成当前回合，实际步数可能超过目标。预检、评估和资格验证不计入正式预算。短资格验证使用对应版本的专用流程，见 [v2](docs/curriculum_training_v2.md#运行) 和 [v3](docs/curriculum_training_v3.md#命令)。

## 项目目录

| 位置 | 职责 |
| --- | --- |
| `../rm_simulator_2027` | Rust/Bevy 物理、机器人资产、弹丸、裁判和同步训练进程 |
| `../rm_vision_2027` | C++ 视觉估计、弹道和云台控制 |
| `src/transport/` | 仿真与桥接通信、进程生命周期 |
| `src/environment/` | Gym 环境、禁射预热、奖励和回合边界 |
| `src/policy/` | 观测编码、GRU Actor 和本地推理 |
| `src/training/` | 课程、完整回合采集、循环 PPO、评估与检查点 |
| `config/` | 训练预设、观测契约和工具参数 |
| `tools/training/` | 课程运行、诊断、报告、回放和部署 |
| `artifacts/` | 构建、训练及模型产物，不纳入 Git |

旧 MLP、静止/旋转靶实现及检查点读取用于历史实验，仓库不再提供 MLP 新训练预设。相关用法集中在[历史 MLP 工具](docs/legacy_mlp.md)和[历史接口](docs/legacy_interfaces.md)。[2026-09-24 CPU 吞吐测量](docs/training_performance.md)只适用于当时的配置，不能代表当前 GRU/128 环境性能。
