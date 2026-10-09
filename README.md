# RM Vision RL

用于 RoboMaster 火控强化学习的环境适配项目。策略负责当前跟踪目标的装甲板选择和开火时机，继续复用视觉项目的 PnP、目标预测、弹道解算和 TinyMPC。

**当前提供随机原地旋转靶 `RMRotationFire-v0`、静止靶 `RMStaticFire-v0`、MaskablePPO/MLP 训练入口，以及规则和 Python 回调评估闭环。** Gym 支持规则选板的两动作模式和 RL 联合选板/开火的九动作模式，每步推进 10 ms，基础环境默认预热后采样 30 秒。PPO 训练默认使用联合九动作策略和每回合随机出生的原地旋转靶（2–8 米，双向 1–7 rad/s 匀速）、5 秒（500 步）回合，支持检查点恢复、CSV 和 TensorBoard 日志。GRU、模型导出和实机推理属于后续工作。

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
- [CPU 并行采样实测](docs/training_performance.md)：保持 1 ms 物理步长的 1/2/4/8/16 环境吞吐对照。
- [接口与配置](docs/interfaces.md)：场景、动作、观测、会话和计分语义。
- [当前状态与开发顺序](docs/development.md)：现有能力、缺口及阶段二完成标准。

Python 主包名为 `src`，通过 `import src` 注册 Gym 环境。
Python 示例从本仓库根目录执行；Gym 安装 `requirements.txt`，PPO 安装 `requirements-training.txt`。
训练命令为 `python -m src.training.train`，负责训练、日志和模型保存。
训练结束后，独立运行 `python -m tools.training.analysis`，评估默认配置中的训练目录；也可用 `--run-dir <训练目录>` 临时覆盖。评估已保存
检查点、在多个共享随机场景上按平均完整窗口伤害选出 `best.zip`，生成报告并打开 Final／Best 回放窗口。
分析命令的 `--no-view` 只禁止打开窗口，仍生成全部分析产物。
`python -m tools.training.manual` 可在实时运行的真实 Gym 环境中手动连发，并显示单步奖励、
累计奖励和最近非零奖励；预热后自动运行，联合模式先按 1–4 选板，再按住 F 请求射击、松开停止新请求，空格暂停／继续、R 生成下一场景。
默认配置为 `config/training/rotation_joint_shared_score_ppo.json`（共享评分器联合选板/开火策略，命中一发奖励 1，并加入切板、快速往返及预测命中朝向机会损失）。
多环境 CPU 并行采样通过 `--n-envs N` 覆盖配置中的环境数量，物理步长保持 1 ms。支持旧检查点恢复；
速度对照命令及日志说明见[运行文档](docs/running.md#3-ppo-训练恢复和推理)。
新训练及默认手动配置启用 **50–100 ms 随机射击决策时钟**，每次机会后独立重采样，其他控制步仍每 10 ms 更新。
范围、采样方式和独立种子在 `environment.decision_clock` 配置；HUD 显示下一次机会的倒计时。
旧检查点未保存时钟配置时沿用旧行为；时钟增加观测/动作契约，需要新建训练。详见[运行文档](docs/running.md)。
`python -m tools.training.view --checkpoint <模型.zip>` 可手动查看，
`--replay <replay.json>` 可重新播放；构建、计分口径及操作见运行文档。
离线评估可用 `--scene-count N` 选择场景数量；未指定种子时每次运行随机生成场景，或用 `--scene-seeds S1 S2 ...` 指定种子以复现，
并用 `--workers N` 并行运行场景；每个场景都会保存独立回放，同时生成一个 `playlist.json`。
多场景查看只打开一个窗口，按 `[`/`]` 循环切换场景；单个 `replay.json` 仍可单独播放。
增加 `--live` 可加载指定模型进行实时推理和实时播放；窗口中按 ↑ 使用下一个随机种子，
空格暂停/继续，关闭窗口退出。实时模式不生成离线 `replay.json`，过程日志保存在运行目录。
旧 `tools.validation` 等命令不再适用。短训练可运行不表示策略已收敛或超过传统火控。

工具参数默认位于 `config/tools/` 下对应的 JSON 文件中。日常运行可直接使用
`python -m tools.training.view`、`analysis`、`compare`、`manual`、`diagnose` 或 `benchmark`；
使用 `--config PATH` 切换工具配置，命令行显式参数覆盖配置文件。

本地训练 Web 控制台需要额外安装 `requirements-web.txt`：

```bash
artifacts/venv/bin/python -m pip install -r requirements-web.txt
artifacts/venv/bin/python -m tools.training.dashboard
```

打开 `http://127.0.0.1:8765` 后可选择训练预设、设置预算和并行环境数、监控任务、优雅停止、
从完整 checkpoint 继续训练，并在 **Tools 分析工具** 页面提交 analysis、compare、diagnose、
benchmark、离线 view/replay 和本机 manual 任务。工具沿用现有 CLI 产物（HTML、JSON、CSV、回放），
可从任务详情直接打开；manual/live viewer 仍在本机原生窗口运行。所有任务共用一个活动任务锁，
不会修改仓库中的预设。

联合模式设定 `environment.action_mode="joint"`，选板每 10 ms 更新，不套用传统选板角度门限。
非射击机会仍可切板；已有脉冲继续执行。省略该字段保持 `fire_only`，原
旧两动作模型仍可通过检查点继续评估；当前唯一保留的新训练配置使用 v2/v3 的
`features[8,107]` 和九动作掩码，不能恢复两动作权重。

```bash
# 短链路检查：一个 rollout；不会启动默认的长训练预算
artifacts/venv/bin/python -m src.training.train --timesteps 1024 --output-dir artifacts/training/joint-check
# 3/5/7 m × ±1/±3/±7 rad/s，共 18 个配对场景；两个模型均须为 5 秒回合
artifacts/venv/bin/python -m tools.training.compare \
  --checkpoint artifacts/training/joint-check/final.zip \
  --baseline-checkpoint <旧两动作模型.zip>
```

比较命令输出逐场景回放、JSON/CSV 和 `comparison.html`；相同出生、时钟和配置的双方完整结果
才进入配对汇总，失败场景单列。场景和短训练的结果不能直接外推实机效果。


当前策略 `shared_score_mlp` 对四个候选使用同一评分器，选板特征不再按固定槽位分别输出；
它使用联合模式的九动作协议，必须从头训练，不能恢复其他策略类型的检查点。
详见 [策略接口](docs/interfaces.md#shared_score_mlp-策略契约)。
