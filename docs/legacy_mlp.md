# 历史 MLP：推理、分析与诊断

本页供已有 MLP 配置和检查点使用，保留旧实验参数与操作方法。当前训练见[构建与运行](running.md)，历史观测/策略契约见[历史接口](legacy_interfaces.md)。示例中的模型和实验产物需自行准备，不能用当前 GRU 配置替代。

## 检查点推理

以下为历史 MLP 检查点的带掩码推理示例；GRU 使用前文的 Actor 部署入口：

```bash
artifacts/venv/bin/python - <<'PYCODE'
from pathlib import Path
from sb3_contrib.common.maskable.utils import get_action_masks
from src.training.environment import make_environment
from src.training.models import read_checkpoint_metadata, load_model

checkpoint = Path('artifacts/training/rotation-v1/final.zip')
metadata = read_checkpoint_metadata(checkpoint)
env = make_environment(metadata['config'], Path('artifacts/inference'))
try:
    model, _ = load_model(checkpoint, env, device='cpu')
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(env))
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            print(info['episode_damage'], info['actual_shots'], info['end_reason'])
            break
finally:
    env.close()
PYCODE
```

GRU 的推理与状态管理见[持续记忆 GRU](persistent_gru.md)。

### 旧 MLP 检查点：训练结束后的模型回放

训练 CLI 正常结束后保存 `final.zip` 并关闭训练进程。独立运行 `tools.training.analysis`，
逐个评估本次目录的
`checkpoint_*.zip` 和 `final.zip`。`latest.zip` 与已有 `best.zip` 不参与；同一步数只评估一次，
优先 `final.zip`。评估沿用模型场景、种子和窗口，使用确定性带掩码推理，完整结算伤害
默认在 5 个共享随机场景上评估各个检查点，按平均 `official_damage` 最高者保存为 `best.zip`，
同分选累计步数较大的模型。可用 `--evaluation-scenes N` 调整场景数量（至少 2 个）。

每次分析使用模型固定副本，记录 SHA-256、来源、步数、排名、得分、回放和状态；分析目录的
`best.zip` 不受后来重新分析影响，训练目录的 `best.zip` 指向最近一次成功选优的结果。
预热失败、结算不完整或异常的候选不参与排名；部分失败标注“有效候选中的最佳”。全部失败
不发布新 `best.zip`，之前的文件保留但不代表本次结果；本次结果以 `analysis.json` 为准。
报告仍展示已有训练曲线和失败说明。这里的最佳仅指本次目录中已保存模型在本次共享随机场景集合上的平均表现，
不代表所有更新时刻或更大范围的跨场景泛化表现；恢复训练不扫描此前目录。

评估结束后打开 HTML 报告，并同时启动 Final／Best 两个三维回放窗口。标题和 HUD 标识角色、
来源模型、累计步数与完整结算得分；如果两者相同，复用同一次评估数据，仍开两个窗口并标注
`Final = Best`。窗口独立操作，关闭一个不影响另一个；Ctrl+C／SIGTERM 会清理所有所属回放进程。

训练 CLI 和 `run_training()` 均只负责训练，不依赖工具模块，也不自动评估或打开窗口。
分析、回放、诊断和人工操作位于 `tools/training/`，单向复用训练层的模型与环境接口。
分析命令的 `--no-view` 只禁止打开窗口，仍评估并保存最佳模型、回放和报告。
训练命令不再接受 `--no-view`、`--skip-analysis` 或 `--viewer-binary`。

给已有完整训练目录补做分析（每次新建分析目录，不复用旧评分）：

```bash
artifacts/venv/bin/python -m tools.training.analysis \
  --run-dir artifacts/training/rotation-v1
# 无窗口评估；仍生成 best.zip、回放和报告
artifacts/venv/bin/python -m tools.training.analysis \
  --run-dir artifacts/training/rotation-v1 --no-view
```

分析入口也支持 `--device` 和 `--viewer-binary`。它只接受 `run.json` 状态为 `complete` 的目录；
全部评估失败或报告生成失败返回非零退出码；独立分析不改变原训练成功状态。

工具参数默认位于 `config/tools/`：`view.json`、`analysis.json`、`compare.json`、`manual.json`、
`diagnose.json`、`benchmark.json` 和 `dashboard.json`。不带参数运行工具时读取对应默认文件；使用 `--config PATH`
切换配置，显式命令行参数覆盖文件中的值。每次工具运行会在产物目录写入最终生效的
`tool_config.json`，便于复现。

字段约束与入口保持一致：`view` 配置模型／回放模式、设备、回放输出、窗口、场景种子和 worker；
`analysis` 配置训练目录、评估场景数和窗口；`compare` 配置两个检查点、输出目录、设备和 worker；
`manual` 配置检查点或训练环境配置、回合步数、窗口和场景种子；`diagnose` 配置检查点、rollout 数、
设备和输出目录；`benchmark` 配置训练配置、环境数、重复次数、rollout 模式及 `timesteps` 或
`updates` 预算。配置文件中的路径相对 `rm_vision_rl` 根目录解析；`--config` 只切换工具配置，
手动和吞吐工具分别用 `--training-config` 临时指定训练环境配置。

常用入口可以直接运行：

```bash
artifacts/venv/bin/python -m tools.training.view
artifacts/venv/bin/python -m tools.training.analysis
artifacts/venv/bin/python -m tools.training.compare
artifacts/venv/bin/python -m tools.training.manual
artifacts/venv/bin/python -m tools.training.diagnose
artifacts/venv/bin/python -m tools.training.benchmark
```

例如临时将回放场景数和并行数覆盖为新值：

```bash
artifacts/venv/bin/python -m tools.training.view --scene-count 10 --workers 8
```

首次使用时构建回放程序：

```bash
cd /home/dgx_spark/RM_VISION_WORKSPACE/rm_simulator_2027
cargo build --offline --release --no-default-features --features training --example training_preview -j 6
```

在 `rm_vision_rl` 根目录手动查看检查点，或重新播放已有数据：

```bash
artifacts/venv/bin/python -m tools.training.view \
  --checkpoint artifacts/training/rotation-v1/final.zip
artifacts/venv/bin/python -m tools.training.view \
  --replay artifacts/training/rotation-v1/evaluation/eval-XXXX/replay.json
```

若要让模型直接驱动实时仿真并观察不同随机场景，可使用：

```bash
artifacts/venv/bin/python -m tools.training.view \
  --checkpoint artifacts/training/rotation-v1/final.zip --live
```

也可以指定首个场景种子：

```bash
artifacts/venv/bin/python -m tools.training.view \
  --checkpoint artifacts/training/rotation-v1/final.zip --live --scene-seed 20260928
```

离线评估也可以一次选择多个场景。下面命令每次运行随机生成 5 个不重复的场景种子：

```bash
artifacts/venv/bin/python -m tools.training.view \
  --checkpoint artifacts/training/rotation-v1/final.zip \
  --scene-count 5 --workers 4 --no-view
```

如果需要完全复现某次评估，可以直接给出指定的随机种子：

```bash
artifacts/venv/bin/python -m tools.training.view \
  --checkpoint artifacts/training/rotation-v1/final.zip \
  --scene-seeds 17 23 41 89 --workers 4 --no-view
```

`--workers` 控制同时运行的独立场景数；每个 worker 都有自己的仿真器、视觉桥接和模型副本。
每个场景仍会单独生成一个 `replay.json`，同时生成包含场景顺序和随机种子的 `playlist.json`；
省略 `--no-view` 时只打开一个回放窗口，使用 `[`/`]` 循环切换场景。

实时窗口启动后会自动进行禁射预热并运行模型；按 `↑` 重置到下一个随机种子，按空格暂停或继续，
关闭窗口退出。每个场景的 Gym 步进和奖励日志保存在 `artifacts/manual/` 下的独立目录中。
实时模式不生成离线 `replay.json`，需要固定场景和完整结算时仍使用上面的离线评估模式。

`--checkpoint` 使用确定性、带动作掩码的推理，不更新参数。场景、种子、测量和回合长度沿用
检查点，该历史实验使用 5 秒窗口；旧检查点若是 100 步，则演示仍只有 1 秒。仍检查观测、算法、策略和
环境／视觉配置指纹。`--device cpu` 可切换推理设备，`--output-dir DIR` 指定评估产物父目录，
每次在其中新建唯一子目录。默认位于检查点同级的 `evaluation/`。`--replay` 只读回放数据，
不加载模型、不启动物理环境，也不需要训练依赖；仍需要原仿真配置和资产。

评估经历禁射预热、正式窗口和自然结算，终端输出进度。每 10 ms 记录真实姿态、云台、弹丸和
事件；完整运行并回收进程后原子保存 `replay.json`，随后才打开窗口。窗口内原始累计伤害与
窗口内出膛弹丸的最终归属伤害分开显示，结算结果不写回 PPO 奖励。出膛恰好位于窗口结束时
或之后的弹丸不计入窗口发射数，HUD 单列这些出膛。命中率为窗口内造成伤害的弹丸数除以窗口
实际出膛数，无出膛显示“—”（窗口字体使用 ASCII `--`）；结算超时标记 `INCOMPLETE`，不显示完整归属伤害。

默认第三人称，`C` 切换己方相机，方向键环绕，`PageUp/PageDown` 缩放；空格暂停，`-`/`+`
选择 0.25、0.5、1、2 倍速，`R` 从头重播。多场景列表中 `[`/`]` 切换上一个／下一个场景，
切换从新场景第 0 帧开始；HUD 显示当前场景编号、随机种子、目标距离和目标转速。末帧会保留，关闭窗口退出。黄色线段是采样轨迹，
红色标记表示该机器人受到伤害，并非精确撞击点。回放不推进物理，变速和重播不会改变计分。

分析和查看命令均可用 `--viewer-binary PATH` 指定回放程序。单模型查看命令还支持
`--label TEXT` 设置窗口及 HUD 标签。修改后需重新构建 `training_preview`，以支持角色标签。
无图形会话、浏览器不可用或程序缺失时保留数据，打印报告路径及回放打开命令；推理或显示
失败单独报错，已完成的训练和模型不受影响。分析工具只接受已完成的训练目录；评估故障不发布
完整回放文件，进程日志留在该次评估目录供定位问题。

### 旧 MLP 检查点：PPO 奖励学习信号诊断

检查已有模型的“命中奖励 → GAE 优势 → PPO 更新 → 发射概率变化”：

```bash
artifacts/venv/bin/python -m tools.training.diagnose \
  --checkpoint artifacts/training/rotation-v1/final.zip --rollouts 3
```

`--checkpoint` 必填；`--rollouts` 默认为 3，按保存的 `n_steps` 执行完整采样与更新。
支持 `--device` 与 `--output-dir DIR`（独立会话目录的父目录）。默认产物在
`artifacts/diagnostics/diagnose-*/`。无需图形环境；结束后打印本地 `report.html` 路径，
可手动用浏览器打开，图表脚本内嵌，支持离线悬停、缩放和图例切换。

诊断从检查点快照恢复权重、优化器及场景配置，并从新回合开始；不会重建此前训练的随机数
或物理现场。沿用原奖励、掩码、回合上限和 PPO 参数，不强制发射，不运行选优／回放，不覆盖
原模型或发布 `best.zip`、`latest.zip`。没有命中时也不自动增加预算。

主要产物：

- `run.json`：生效配置、原模型摘要、运行状态、三轮汇总及检查结论。
- `sampling.jsonl`：即使采样中途失败也保留的原始步骤；`steps.csv`：完成更新后的奖励、优势及概率对照。
- `minibatches.csv`：实际小批次索引、每次标准化优势、概率比、裁剪是否生效及是否执行优化步骤。
- `rollout-*.npz`：观测、缓冲区、bootstrap 值、独立重算优势及逐个移除奖励的贡献矩阵；
  `update-*.npz`：同一观测／掩码上的更新后概率与价值预测。
- `events.jsonl`：自动重置前读取的物理请求、出膛、伤害事件及请求／弹丸编号关联；
  `event-associations.jsonl` 再通过桥接命令的精确时间戳关联到 Gym 动作，不使用时间邻近猜测。
- `parity.json`：首轮同一缓冲区、相同权重／优化器／随机数状态下，有无跟踪的对照；不额外采样环境。
- `report.html`、环境及 PPO 日志；`source.zip` 是本次只读来源的完整快照，不是更新后的模型。

报告将缩放后的训练奖励与时间截断的价值补偿分开显示；物理伤害事件保持原单位，奖励接入检查会乘以保存的 reward_scale 并减去切板惩罚。奖励移除只在数据副本上进行，保持状态、
价值预测与回合边界不变；它验证数值传播，不代表“不发射时”的环境反事实。正优势样本也不保证
整轮更新后该动作概率一定增加；需结合全部样本、实际小批次标准化和裁剪一起判断。
解释方差的目标方差为零时显示暂无数据。未完成回合不追加尾部结算，回合末在途弹丸单独计数。

正常关闭、Ctrl+C、SIGTERM 或环境异常均清理所属进程；故障保留已完成记录并标注状态，不保存
部分更新为有效模型。首轮一致性对照失败时停止后续采样。对应诊断测试须使用该历史版本的测试文件；当前工作区未保留 `test_diagnose.py`。

## 历史 MLP：联合选板模式与成对比较

以下保留旧实验的接口与运行方法。复现需提供实验目录中保存的 MLP 配置或检查点；不使用当前课程 v2 预设运行这些新建 MLP 训练命令。

历史联合模式配置设定 `environment.action_mode="joint"`：每 10 ms 选择装甲板，
射击机会保持随机 50–100 ms。预热使用规则且禁射，正式采样由 RL 自主选板，不套用规则面向角门限。
旧两动作模型只能通过已有检查点加载，不能把旧两动作权重恢复为联合模型。
第一次验证建议 `--timesteps 1024`，新建独立输出目录。

```bash
artifacts/venv/bin/python -m tools.training.compare \
  --checkpoint <联合模型.zip> --baseline-checkpoint <旧两动作模型.zip> \
  --output-dir artifacts/comparison
```

默认执行 3、5、7 m × -7、-3、-1、1、3、7 rad/s，共 18 个场景，每种策略正式运行 5 秒并自然结算。
双方复用合法出生种子、场景和时钟；每次运行保留模型快照、实际参数和回放。
加 `--workers 16` 可同时评估最多 16 个场景；每个场景内部依次运行联合模型和旧模型，
共 36 次评估。默认 `--workers 1` 保持串行。并行使用独立 spawn 进程、仿真器及视觉桥接，
不改变场景种子、射击时钟或计分窗口；结果按场景编号归档，失败结果单独记录。
该参数独立于训练的 `n_envs`，不会追加训练。评估期间只输出终端进度，画面通过完成后的回放打开。
查看 `comparison.html` 或 `comparison.csv`，失败场景有明确状态，不能把部分完成结果解释为完整成绩。
`tools.training.view --replay <回放.json>` 可打开指定场景，HUD 显示槽位、切板数和协议动作。

手动窗口中 1–4 更新期望槽位，F 请求该槽位射击；未选板时等待。重复选同板即保持。
非法输入在 Gym 降级并显示掩码原因；R 清除期望槽位并生成下一场景，需要重新按数字键。
释放 F 不撤回已经接受的脉冲。旧两动作配置无需数字键，沿用 F 和规则选板。

构建 bridge 与 training_preview 后再运行工具；按当前 tests 目录选择实际存在的契约测试。


### 选板质量与快速往返奖励（2026-09-26）

该阶段配置采用 `selection_reward`，与旧 `switch_penalty` 互斥；旧检查点仍使用原奖励。
新奖励为 `damage * 0.05 - base_switch_penalty - return_penalty - facing_penalty`。
基础切板成本 0.002；A→B→A 返回时追加 `0.03 * max(0, 1-age_B/0.1s)^2`。
朝向质量取视觉预测命中朝向 `max(0, cos(theta))^2`；每 10 ms 扣除
`0.1/s * 0.01s * max(0, best_valid_quality-selected_quality-0.1)`。
朝向评分和实际执行槽位使用同一动作前候选快照；尚未选板按质量 0。
跟踪非 tracking、无有效候选、跳过回调或目标代次变化不计朝向损失；
首次选板、旧板失效和目标重置不计切板成本，恢复时清除返回历史。

新配置启用 `environment.selection_memory=true`：保留 v2 的 8×107 特征，
另加 5 项 one-hot（-1/0/1/2/3）表示上一个不同的实际选中槽位，契约版本为 v3。
驻留时间复用已有特征；训练和评估共享记忆更新。旧 v1/v2 观测不变，不能直接恢复为 v3。
分项奖励写入 rewards.jsonl、Monitor、CSV 和 TensorBoard；`reward_penalty` 为总扣分，
历史字段 `switch_penalty` 在新模式下是总扣分的兼容别名。

```bash
# 从头训练：64 环境，每个环境 1024 步，共 16 次 PPO 更新。
artifacts/venv/bin/python -m src.training.train \
  --config <历史MLP实验配置副本.json> \
  --n-envs 64 --timesteps 1048576 --output-dir artifacts/training/joint-selection-v3
```


### 共享评分策略

旧共享评分实验使用 `shared_score_mlp`，对四个候选装甲板使用同一评分器，
保持联合模式的九动作协议。它必须从头训练，不能恢复其他策略结构的检查点。

```bash
# 两个真实环境，各采集 1024 步；一次 PPO 更新，验证自动换场景。
artifacts/venv/bin/python -m src.training.train \
  --config <历史MLP实验配置副本.json> \
  --n-envs 2 --timesteps 2048 --output-dir artifacts/training/shared-score-smoke
# 加载检查点，评估后打开回放。
artifacts/venv/bin/python -m tools.training.view \
  --checkpoint artifacts/training/shared-score-smoke/final.zip
```

旧模型继续通过 `--resume` 或 `--checkpoint` 加载。单检查点评估、compare、diagnose、回放及手动
工具共用检查点配置，不需要另行切换动作协议。新结构改变策略表达及确定性解码，
短训练仅验证链路，是否改善命中率需固定场景对照验证。
