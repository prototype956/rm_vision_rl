# 辅助工具

此目录维护训练分析和人工验证工具。工具依赖 `src` 的运行接口，
运行库和训练入口不反向依赖工具。所有命令从仓库根目录执行，完整参数见[构建与运行](../docs/running.md)。

工具参数默认从 `config/tools/<tool>.json` 读取，也可以用 `--config PATH` 切换配置；显式命令行参数覆盖配置文件。
每次运行会在产物目录保存最终生效的 `tool_config.json`。

## 按任务选择工具

| 任务 | 模块 | 说明 |
| --- | --- | --- |
| v2 预检、资格验证、正式训练 | `curriculum_v2` | 默认配置仍为步兵 v2 |
| v2 冻结运行 | `curriculum_v2_pipeline` | 校验并复制运行环境 |
| v3 预检、传统校准、资格验证、正式发布 | `curriculum_v3` | 发布修订 3，须提供校准和验收证据 |
| v3 固定案例成对 MCAP | `curriculum_v3_replays` | 预先选定案例，不按成绩挑选 |
| 课程报告与诊断 | `curriculum_report`、`curriculum_v2_diagnostics` | 报告支持 v2/v3 的角色目录或任务组 |
| 检查点迁移 | `checkpoint_migration clearance / warmup / curriculum` | 共用载荷保存逻辑，各模式保留独立兼容性门禁 |
| v2 冻结副本检查 | `curriculum_v2 check-snapshot` | 检查运行路径、控制时钟和初始热锁 |
| Actor 导出与本地仿真部署 | `recurrent_export`、`recurrent_deploy` | v5 GRU Actor |
| GRU Checkpoint/Actor MCAP | `view --record-mcap` | 共同离线评估器与冻结场景 |
| 手动 C0 场景 | `manual --mode config` | 不训练、不晋级 |

完整流程见[课程 v2](../docs/curriculum_training_v2.md)、[课程 v3](../docs/curriculum_training_v3.md)，导出与状态管理见[持续记忆 GRU](../docs/persistent_gru.md)。`config/training/` 有两类角色各自的 v2/v3 四份预设。

通用 `benchmark` 默认读取步兵 v2，会进入完整课程预检和最终评估；短资格验证使用对应版本的专用 `curriculum_v2/v3 benchmark`。

## 历史 MLP 分析与人工验证

下列模型分析、对比、诊断和实时检查点查看工具用于已有 MLP 实验；
GRU 使用上表工具；`view` 的 `--record-mcap` 是 GRU 专用分支。独立回放不受模型类型限制。

| 文件 | 职责 |
| --- | --- |
| [training/analysis.py](training/analysis.py) | 独立评估已完成训练的检查点、选择最佳模型、组织报告和回放 |
| [training/report.py](training/report.py) | 生成训练曲线和模型对比 HTML 报告 |
| [training/compare.py](training/compare.py) | 18 个共同出生场景的联合/两动作策略成对评估、报告与回放 |
| [training/view.py](training/view.py) | 单模型离线评估、回放记录、原生窗口显示和实时策略推理 |
| [training/manual.py](training/manual.py) | 人工操作实际 Gym 环境，记录逐步奖励和事件 |
| [training/diagnose.py](training/diagnose.py) | 从检查点执行短期 PPO 更新并采集学习信号 |
| [training/diagnostic_report.py](training/diagnostic_report.py) | 生成学习信号诊断报告 |
| [training/dashboard/](training/dashboard/) | 本地 Web 控制台：启动、监控、停止、恢复训练，并提交全部工具任务 |

```bash
python -m tools.training.analysis
python -m tools.training.view
python -m tools.training.manual
python -m tools.training.compare
python -m tools.training.diagnose
python -m tools.training.benchmark
```

本地 Web 控制台需要额外安装 `requirements-web.txt`，从仓库根目录启动：

```bash
python -m pip install -r requirements-web.txt
python -m tools.training.dashboard
```

默认只绑定 `127.0.0.1:8765`。控制台使用 `config/training/` 中的预设，训练和工具运行仍由
独立子进程执行；任务状态和日志保存在 `artifacts/training/.web-jobs/`。Tools 页面把现有
analysis、compare、diagnose、benchmark、离线 view/replay 和 manual 接入统一任务列表，报告与
回放通过安全 artifact 链接访问。manual/live viewer 需要本机 DISPLAY/WAYLAND，按键仍在原生窗口完成。
默认所有训练和工具任务共用一个活动任务锁；恢复训练只接受有效的完整 checkpoint，并始终创建新的输出目录。

默认配置文件分别为：

```text
config/tools/view.json
config/tools/analysis.json
config/tools/compare.json
config/tools/manual.json
config/tools/diagnose.json
config/tools/benchmark.json
config/tools/dashboard.json
```

配置字段按工具划分：`view` 使用 `mode`、`checkpoint`、`replay`、`device`、`output_dir`、
`viewer_binary`、`label`、`open_view`、`scene_count`、`scene_seeds`、`workers` 和
`scene_seed`；`analysis` 使用 `run_dir`、`device`、`evaluation_scenes`、`viewer_binary`、
`open_view`；`compare` 使用 `checkpoint`、`baseline_checkpoint`、`output_dir`、`device`、
`workers`；`manual` 使用 `mode`、`config`、`checkpoint`、`episode_steps`、`output_dir`、
`viewer_binary`、`scene_seed`；`diagnose` 使用 `checkpoint`、`rollouts`、`device`、
`output_dir`；`benchmark` 使用 `config`、`output_dir`、`n_envs`、`repeats`、`rollout_mode`、
`timesteps`、`updates`。路径均相对仓库根目录；`manual` 的训练环境配置临时覆盖参数为
`--training-config`，避免与工具配置的 `--config` 混淆。

训练使用 `python -m src.training.train`，完成后按需运行上述工具。
分析、回放、人工操作和诊断入口统一位于 `tools.training`，运行库位于 `src`。
`report.py` 和 `diagnostic_report.py` 由对应工具调用。

`view` 的多场景评估会保留每个场景的 `replay.json`，并在同一目录生成 `playlist.json`。
查看器只启动一个窗口；按 `[`/`]` 循环切换场景。单个 `replay.json` 仍可直接通过
`--replay` 打开。

历史 `tools.validation` 等验收命令不再使用。

当前配置模式的手动环境读取步兵或哨兵 v2 预设，按 C0 生成场景，R 使用下一个场景种子。
使用 `--mode config --training-config config/training/rmul_2027_infantry_curriculum_v2.json`。
默认工具配置中的 `checkpoint` 模式仍读取指定的历史 MLP 检查点。

手动操作使用 1–4 选板、F 请求射击；当前配置采用 v5 单步观测和九动作。
历史模型工具按检查点内保存的契约加载。离线 `view` 未指定 `scene_seeds` 时随机生成种子，
实际种子写入 `tool_config.json` 和回放元数据。


旧 MLP 的详细参数、选优规则和诊断产物见[历史工具文档](../docs/legacy_mlp.md)。

## 工具维护与临时产物

`tests/` 保留行为回归测试；迁移、发布和离线评估的测试分别归入
`test_checkpoint_migration.py`、`test_curriculum_release.py` 和 `test_offline_evaluation.py`。
测试临时文件使用 `TemporaryDirectory`；报告、截图、诊断日志和录制输出写到
`artifacts/` 或调用方指定的新目录，不写进测试与工具源码目录。

三种迁移共用 `python -m tools.training.checkpoint_migration <模式>`，
均要求 `--checkpoint`、`--destination`、`--config`；`curriculum` 另需
`--scenarios` 与 `--output`。模式分别保留原净空修复、预热修复和 v2→v3 的白名单，
不允许通过选择其他模式放宽训练契约。迁移不会自动替换原检查点。

课程报告使用 `python -m tools.training.curriculum_report --run <角色目录或任务组>`。
资格覆盖、历史基线例外与动态库检查集中在 `curriculum_v3_release.py`，
仍由 v3 入口调用，所有原有发布门槛保留。固定历史目录、日志字符串和
3,800,921 步检查点的 `resume_clearance_pipeline` 一次性入口已删除。
