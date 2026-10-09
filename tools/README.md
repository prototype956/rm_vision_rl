# 辅助工具

此目录维护训练分析和人工验证工具。工具依赖 `src` 的运行接口，
运行库和训练入口不反向依赖工具。所有命令从仓库根目录执行，完整参数见[构建与运行](../docs/running.md)。

工具参数默认从 `config/tools/<tool>.json` 读取，也可以用 `--config PATH` 切换配置；显式命令行参数覆盖配置文件。
每次运行会在产物目录保存最终生效的 `tool_config.json`。

## 训练分析与人工验证

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

默认手动环境每回合随机生成原地旋转靶，R 切换下一场景。当前唯一保留的新训练配置为
`config/training/rotation_joint_shared_score_ppo.json`；已有静止靶检查点仍可继续评估。
检查点评估默认使用 5 个由保存场景种子派生的共享随机场景，
按平均窗口归属伤害选优；不将该评估集合分数解释为更大范围的泛化表现。

默认手动模式用 1–4 选板、F 请求射击，R 换场景。joint 使用 107 维特征和九动作；检查点工具自动选择保存模式。
离线 `view` 未指定 `scene_seeds` 时，每次运行随机生成场景种子；本次实际种子会写入 `tool_config.json` 和回放元数据。
