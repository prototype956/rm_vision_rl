# RM Vision RL

用于 RoboMaster 火控强化学习的环境适配项目。策略负责当前跟踪目标的装甲板选择和开火时机，继续复用视觉项目的 PnP、目标预测、弹道解算和 TinyMPC。

**当前可以运行规则或 Python 回调策略的仿真闭环；尚未提供 Gymnasium 环境和 PPO 训练入口。** 当前接口面向本机进程通信，远端训练、模型导出和 NUC 实机推理属于后续工作。

## 项目关系

| 项目 | 职责 |
| --- | --- |
| `../bevy_robomaster_simulator` | Rust/Bevy 物理、机器人资产、云台、弹丸和裁判机制；提供同步训练进程 |
| `../rm_vision_2027` | C++ 视觉估计与火控算法；桥接程序直接编译其相关源码 |
| 本项目 | Python 通信、预热、策略回调、观测编码和评估计分 |

## 目录

```text
rmvision_rl/
  transport/                仿真客户端、进程管理、C++ 桥接通信
  environment/              禁射预热和有界评估会话
  policy/                   观测编码与短历史
  scoring/                  按实际出膛时间归属伤害的评估计分
native/vision_bridge/       C++ 桥接源码及构建入口
config/observations/       观测特征 schema
config/policy/             MPC 配置副本参数
tools/config/              配置生成工具（当前有待修复的导入依赖）
docs/                      项目说明
artifacts/                 本地构建、配置副本和运行输出，不纳入 Git
logs/                      本地日志
```

## 文档

- [架构与代码职责](docs/architecture.md)：控制数据流、模块分工和时间模型。
- [构建与运行](docs/running.md)：构建、手动步进、短闭环及策略示例。
- [接口与配置](docs/interfaces.md)：场景、动作、观测、会话和计分语义。
- [当前状态与开发顺序](docs/development.md)：现有能力、缺口及阶段二完成标准。

Python 示例从本仓库根目录执行。历史验收、诊断和基线工具已删除，旧 `tools.validation` 等命令不再适用。本文档根据当前源码重建，不沿用历史测试通过结论，也不表示已完成强化学习训练。
