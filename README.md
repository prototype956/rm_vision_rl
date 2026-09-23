# RM Vision RL

用于 RoboMaster 火控强化学习的环境适配项目。策略负责当前跟踪目标的装甲板选择和开火时机，继续复用视觉项目的 PnP、目标预测、弹道解算和 TinyMPC。

**当前提供随机原地旋转靶 `RMRotationFire-v0`、静止靶 `RMStaticFire-v0`、MaskablePPO/MLP 训练入口，以及规则和 Python 回调评估闭环。** Gym 使用规则选板和两个开火动作，每步推进 10 ms，基础环境默认预热后采样 30 秒。PPO 训练默认使用每回合随机出生的原地旋转靶（2–8 米，双向 1–7 rad/s 匀速）、5 秒（500 步）回合，支持检查点恢复、CSV 和 TensorBoard 日志。GRU、模型导出和实机推理属于后续工作。

## 项目关系

| 项目 | 职责 |
| --- | --- |
| `../rm_simulator_2027` | Rust/Bevy 物理、机器人资产、云台、弹丸和裁判机制；提供同步训练进程 |
| `../rm_vision_2027` | C++ 视觉估计与火控算法；桥接程序直接编译其相关源码 |
| 本项目 | Python 通信、预热、策略回调、观测编码和评估计分 |

## 目录

```text
src/
  transport/                仿真客户端、进程管理、C++ 桥接通信
  environment/              静止/旋转靶 Gym、禁射预热和有界评估会话
  policy/                   观测编码与短历史
  training/                 任务选择与场景序列、模型构建加载、PPO 入口及训练配置
  scoring/                  按实际出膛时间归属伤害的评估计分
native/vision_bridge/       C++ 桥接源码及构建入口
config/observations/       观测特征 schema
config/policy/             MPC 配置副本参数
tools/training/            训练结果分析、回放、诊断和人工验证工具
docs/                      项目说明
artifacts/                 本地构建、配置副本和运行输出，不纳入 Git
logs/                      本地日志
```

## 文档

- [架构与代码职责](docs/architecture.md)：控制数据流、模块分工和时间模型。
- [构建与运行](docs/running.md)：构建、手动步进、短闭环及策略示例。
- [接口与配置](docs/interfaces.md)：场景、动作、观测、会话和计分语义。
- [当前状态与开发顺序](docs/development.md)：现有能力、缺口及阶段二完成标准。

Python 主包名为 `src`，通过 `import src` 注册 Gym 环境。
Python 示例从本仓库根目录执行；Gym 安装 `requirements.txt`，PPO 安装 `requirements-training.txt`。
训练命令为 `python -m src.training.train`，负责训练、日志和模型保存。
训练结束后，独立运行 `python -m tools.training.analysis --run-dir <训练目录>`，评估已保存
检查点、按完整窗口伤害选出 `best.zip`，生成报告并打开 Final／Best 回放窗口。
分析命令的 `--no-view` 只禁止打开窗口，仍生成全部分析产物。
`python -m tools.training.manual` 可在实时运行的真实 Gym 环境中手动连发，并显示单步奖励、
累计奖励和最近非零奖励；预热后自动运行，按住 F 连发、松开停止请求，空格暂停／继续、R 生成下一场景。
默认配置为 `config/training/rotation_fire_ppo.json`；显式选择 `static_fire_ppo.json` 可继续使用固定静止靶。
新训练及默认手动配置启用 **50–100 ms 随机射击决策时钟**，每次机会后独立重采样，其他控制步仍每 10 ms 更新。
范围、采样方式和独立种子在 `environment.decision_clock` 配置；HUD 显示下一次机会的倒计时。
旧检查点未保存时钟配置时沿用旧行为；时钟增加观测/动作契约，需要新建训练。详见[运行文档](docs/running.md)。
`python -m tools.training.view --checkpoint <模型.zip>` 可手动查看，
`--replay <replay.json>` 可重新播放；构建、计分口径及操作见运行文档。
历史验收、诊断和基线工具已删除，旧 `tools.validation` 等命令不再适用。本次不新增验收程序，
最终由用户手工验收；短训练可运行不表示策略已收敛或超过传统火控。
