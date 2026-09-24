# 构建与运行

以下使用 DGX Spark 工作区布局。迁移到其他机器时修改路径并准备对应架构的依赖。第 3 节提供 PPO 训练，其余示例用于环境采样和脚本策略。

## 1. 依赖与构建

- Python 3.10 或更高版本；旧规则/回调运行库使用标准库；Gym 环境需要 requirements.txt 中的 Gymnasium 和 NumPy，不要求 PyTorch。
- 仿真项目可用的 Rust/Cargo 工具链、Cargo 依赖及 `assets/`。
- 支持 C++20 的编译器、CMake、Eigen3、OpenCV（core/calib3d/imgproc）、Ceres、yaml-cpp、fmt、spdlog 和线程库。

构建仿真训练进程：

```bash
cd /home/dgx_spark/RM_VISION_WORKSPACE/rm_simulator_2027
cargo build --offline --release --no-default-features --features training --bin daedalus_training
```

`--offline` 使用已有 Cargo 缓存；未缓存的依赖需要联网去掉此选项构建。当前 Bevy 仍启用动态链接，无窗口运行不意味着已经裁掉全部图形相关构建依赖。

构建桥接：

```bash
cd /home/dgx_spark/RM_VISION_WORKSPACE/rm_vision_rl
cmake -S native/vision_bridge -B artifacts/build/vision-bridge \
  -DVISION_ROOT=/home/dgx_spark/RM_VISION_WORKSPACE/rm_vision_2027 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build/vision-bridge --parallel 2
```

输出分别是仿真项目的 `target/release/daedalus_training` 和本项目的 `artifacts/build/vision-bridge/rmvision-rl-bridge`。

若 Ceres 等依赖使用本工作区已有的本地安装，在上述 CMake 配置命令中增加：

```bash
-DCMAKE_PREFIX_PATH=/home/dgx_spark/RM_VISION_WORKSPACE/rm_vision_2027/.deps/vision/usr
```

## 2. 静止靶 Gym 采样

从 `rm_vision_rl` 根目录安装最小依赖：

```bash
python3 -m venv artifacts/venv
artifacts/venv/bin/python -m pip install -r requirements.txt
```

完成前面的两个 Release 构建后，可直接运行：

```bash
artifacts/venv/bin/python - <<'PYCODE'
import gymnasium as gym
import src  # 注册 RMStaticFire-v0

env = gym.make('RMStaticFire-v0')
try:
    obs, info = env.reset(seed=17)
    while True:
        # 仅是示例策略：允许发射就请求单发；改为 0 可只跟踪。
        action = int(obs['action_mask'][1])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print({k: info[k] for k in
                   ('episode_steps', 'episode_damage', 'actual_shots', 'end_reason')})
            break
finally:
    env.close()
PYCODE
```

默认 4 m 静止目标、高血量、规则选板、二动作、每步 10 ms、预热后 30 秒。
可用 `gym.make('RMStaticFire-v0', episode_steps=300)` 缩短到 3 秒；预热仍单独计时。
`env.unwrapped.action_masks()` 返回 bool 掩码，`obs['action_mask']` 返回等价的 0/1 数组。
几何无效 seed 会在有限次数内派生重采样；预热失败直接报错。运行日志分别写入
`artifacts/gym/env-*/simulator.log` 和 `bridge/stderr.log`。

构造时可覆盖 `simulator_root`、`vision_root`、`simulator_binary`、`bridge_binary`、
`simulator_config`、`log_dir` 和 `warmup_config`；不修改视觉默认配置。场景覆盖通过
`reset(seed=17, options={'scenario': {'target_distance_m': 3.0}})` 传入，运动必须保持 static。
Gym seed 是随机流种子，实际仿真 seed 见 `info['reset_attempts']`。

若先观察正面静止靶，可将示例中的 Reset 替换为以下显式场景；环境默认仍保留随机出生朝向：

```python
obs, info = env.reset(seed=17, options={'scenario': {
    'target_bearing_rad': 0.0,
    'target_yaw_rad': 0.0,
    'controlled_yaw_rad': 0.0,
}})
```

这是采样示例，不进行学习，也不构成效果验收。最终由用户手工验收；需要辅助判断时再按需
读取具体运行数据，不额外生成验收文件。伤害为零时，应结合实际出膛及现有视觉控制效果判断。

旋转靶可直接通过 Gym 使用同一采样器：

```python
import gymnasium as gym
import src

with gym.make('RMRotationFire-v0', episode_steps=500,
              angular_speed_range_rad_s=[1, 7]) as env:
    obs, info = env.reset(seed=17)
    # 完成回合后 env.reset() 取下一场景；reset(seed=17) 从头复现场景序列。
```

## 3. PPO 训练、恢复和推理

训练依赖与基础 Gym 分开安装；在本仓库根目录执行：

```bash
artifacts/venv/bin/python -m pip install -r requirements-training.txt
artifacts/venv/bin/python -m src.training.train --help
```

当前固定 PyTorch 2.8.0、SB3/SB3-Contrib 2.9.0；PyTorch 的 PyPI ARM64 包支持本次 CPU
训练。默认 `device=cpu`，1 个 PyTorch 计算线程。显式指定 CUDA 而当前安装不可用时会报错，
不会悄悄回退。实际依赖版本保存在每次运行的 `run.json` 和检查点内。

开始一轮训练：

```bash
artifacts/venv/bin/python -m src.training.train \
  --config config/training/rotation_joint_ppo.json \
  --output-dir artifacts/training/rotation-v1
```

输出目录必须尚不存在；省略时自动在 `artifacts/training/` 下创建唯一目录。
配置默认 65536 个采样步、每 1024 步更新一次、minibatch 256、10 epochs、两层 128 单元
MLP，以及单环境 `DummyVecEnv`。`--timesteps` 覆盖本次预算，向上取整到完整 rollout 并在
启动时显示；`--seed` 只改变 PPO 随机种子，场景种子另由配置的 `environment.scene_seed` 控制。
配置中的相对环境路径相对于本仓库根目录解析。

默认配置为 `config/training/rotation_joint_ppo.json`：每回合重新生成双方位置和朝向、
2–8 米目标距离、目标方位，以及双向 1–7 rad/s 的匀速旋转。省略初始云台角度时朝向目标。
旋转从禁射预热开始，进入正式采样时不重置姿态。出生使用仿真器几何和初始可见性检查；
非零旋转还需通过每 5° 的整圈可见性采样，避免高台/障碍物只在初始相位短暂露出装甲。
更新此检查后需要重新构建 `daedalus_training` 和 `training_preview`。被拒绝的种子记录在
`reset_attempts` 中，使用现有最多 32 次出生重试；不会跳过真正的预热超时。
`environment.scene_seed`（默认 17）控制可复现的场景序列，自动 Reset 和手动 R 都取下一场景。
`environment.angular_speed_range_rad_s` 配置速度大小范围，例如 `[1, 3]`；正反方向各半，
要求 `0 < min <= max <= 7`。不要设置 `scenario.motion`，它由环境采样生成。
若要使用原固定 4 米正面静止靶，显式传入 `--config config/training/static_fire_ppo.json`。
旧配置省略 `environment.task` 时仍解释为静止靶，旧检查点保留原行为。
恢复旋转靶训练从保存种子的首个场景重新开始，不接续中断前的场景随机流。
默认 PPO 训练回合为 5 秒（500 步），预热单独计时；要调整回合长度、rollout 或 minibatch，编辑配置副本对应的
`environment.episode_steps`、`ppo.n_steps` 和 `ppo.batch_size`。minibatch 必须整除 rollout。

新训练和不带 `--checkpoint` 的手动入口默认启用随机射击决策时钟：

```json
"decision_clock": {
  "min_interval_ms": 50,
  "max_interval_ms": 100,
  "resample": "decision",
  "seed": 17
}
```

该对象放在 `environment` 下。每次射击决策后，从 50、60、70、80、90、100 ms 等概率抽取下一间隔；无论本次是否开火都推进时钟，不在中间的 10 ms 步重试。非决策步继续瞄准、物理和奖励更新。`resample: "episode"` 可改为每回合采样固定周期，端点相同则是固定周期。省略对象可关闭；`--seed` 只改 PPO 种子，时钟使用自己的 seed。

手动测试当前配置：

```bash
artifacts/venv/bin/python -m tools.training.manual --config config/training/rotation_joint_ppo.json
```

联合模式先用数字键 1–4 选择对应的槽位 0–3；HUD 显示实际槽位、切板数和九项掩码。
HUD 显示距下次决策的时间；长按 F 只在到期且原火控合法时提交请求。R 清零奖励、生成并预热下一场景（静止靶配置仍重复相同场景），并切换到下一个时钟随机流。每回合仍为 500 步/5 秒。时钟会跳过部分仍在冷却中的机会，因此启用后持续发射分数可能低于原无时钟的 1380；随机发射伤害也不保证严格按概率缩放。

旧检查点保存的配置没有该对象，`--resume`、`--checkpoint` 回放及手动模式都会保持旧行为；新增观测/动作时序需要新建训练，不能静默改变旧检查点。所有候选模型评估使用各自保存的时钟配置，并从随机流 0 开始。

新训练默认设置 `environment.scenario.unlimited_heat=true`：双方不累积热量，也不会触发热量
锁定；遥测和观测中的当前热量为 0，热量上限/冷却速率保留预设有限值。射击间隔、供弹、
云台及其他合法动作约束继续生效。启动终端会显示热量模式，配置随检查点和回放保存。
将该字段改为 `false` 可重新启用热量；省略字段时也启用热量，以兼容旧场景和检查点。
旧检查点的 `--resume` 和回放沿用其保存的模式，不受新默认值影响；本次无限热量训练请使用
新建训练命令，不从旧的有热量检查点恢复。

运行输出包括：

- `run.json`：生效配置、模型/观测信息、配置指纹、依赖版本、预算和运行状态。
- `episodes.monitor.csv`：每个完整回合的原始奖励、伤害、实际出膛数、长度及结束原因。
- `environment/env-*/decision-clock.jsonl`：启用时钟时的逐次射击机会、随机间隔、原火控合法性和请求接受结果。
- `logs/progress.csv` 和 TensorBoard event：每次完整更新后的损失、KL、熵损失、采样速度和回合均值。
- `checkpoint_<累计步数>.zip`：每 4 次完整更新保存一次；`latest.zip` 原子指向最近保存的完整状态。
- `final.zip`：正常结束时保存，同时更新 `latest.zip`。尚未结束回合的奖励不会伪装成完整回合数据。
- `best.zip`：CLI 后处理按独立评估完整窗口归属伤害选出的最佳已保存模型；同分优先累计步数较大的模型。
- `analysis/analysis-*/`：每次分析独立保存模型与日志副本、逐模型评估回放、`analysis.json`、本次 `best.zip` 和离线交互 `report.html`。

追加训练：

```bash
artifacts/venv/bin/python -m src.training.train \
  --resume artifacts/training/rotation-v1/latest.zip \
  --timesteps 65536 \
  --output-dir artifacts/training/rotation-v1-resume
```

恢复预算是**追加步数**。恢复权重、优化器和累计步数，环境从保存场景序列的首个回合开始；不恢复
原物理世界或未完成 rollout。恢复沿用检查点内配置，不同时接受 `--config` 或 `--seed`，
只允许覆盖预算、设备和输出目录。观测规格或环境/视觉配置指纹变化时拒绝恢复。
Ctrl+C、SIGTERM 或运行故障会退出并清理进程，保留已有完整检查点，不保存半次更新。

查看训练曲线：

训练 CLI 默认在训练结束后生成并打开本地交互报告，包含完整回合奖励、最近 20 回合移动平均、
伤害、实际出膛数、PPO 指标及独立评估模型对比。前 19 回合的移动平均使用已有回合；缺失或
非有限指标显示空缺，不补零。横轴为累计训练步数，恢复训练的回合曲线加上起始偏移。
报告内嵌 Plotly，可离线打开、悬停查看数值、缩放和切换图例。原有 TensorBoard 仍可使用：

```bash
artifacts/venv/bin/tensorboard --logdir artifacts/training --port 6006
```

加载模型运行一次带掩码推理；这仍是 Gym 采样，不是独立窗口结算评估：

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

当前只实现 `maskable_ppo/mlp`。后续 GRU 必须配套序列采样、隐藏状态复位和动作掩码训练，
不能只替换网络层。短训练仅检查参数更新和保存恢复，最终效果由用户手工判断，不自动生成验收文件。

### 训练结束后的模型回放

训练 CLI 正常结束后保存 `final.zip` 并关闭训练进程。独立运行 `tools.training.analysis`，
逐个评估本次目录的
`checkpoint_*.zip` 和 `final.zip`。`latest.zip` 与已有 `best.zip` 不参与；同一步数只评估一次，
优先 `final.zip`。评估沿用模型场景、种子和窗口，使用确定性带掩码推理，完整结算伤害
`official_damage` 最高者保存为 `best.zip`，同分选累计步数较大的模型。

每次分析使用模型固定副本，记录 SHA-256、来源、步数、排名、得分、回放和状态；分析目录的
`best.zip` 不受后来重新分析影响，训练目录的 `best.zip` 指向最近一次成功选优的结果。
预热失败、结算不完整或异常的候选不参与排名；部分失败标注“有效候选中的最佳”。全部失败
不发布新 `best.zip`，之前的文件保留但不代表本次结果；本次结果以 `analysis.json` 为准。
报告仍展示已有训练曲线和失败说明。这里的最佳仅指本次目录中已保存模型在首个参考场景下的表现，
不代表所有更新时刻或跨场景泛化表现；恢复训练不扫描此前目录。

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

`--checkpoint` 使用确定性、带动作掩码的推理，不更新参数。场景、种子、测量和回合长度沿用
检查点，新训练默认 5 秒；旧检查点若是 100 步，则演示仍只有 1 秒。仍检查观测、算法、策略和
环境／视觉配置指纹。`--device cpu` 可切换推理设备，`--output-dir DIR` 指定评估产物父目录，
每次在其中新建唯一子目录。默认位于检查点同级的 `evaluation/`。`--replay` 只读回放数据，
不加载模型、不启动物理环境，也不需要训练依赖；仍需要原仿真配置和资产。

评估经历禁射预热、正式窗口和自然结算，终端输出进度。每 10 ms 记录真实姿态、云台、弹丸和
事件；完整运行并回收进程后原子保存 `replay.json`，随后才打开窗口。窗口内原始累计伤害与
窗口内出膛弹丸的最终归属伤害分开显示，结算结果不写回 PPO 奖励。出膛恰好位于窗口结束时
或之后的弹丸不计入窗口发射数，HUD 单列这些出膛。命中率为窗口内造成伤害的弹丸数除以窗口
实际出膛数，无出膛显示“—”（窗口字体使用 ASCII `--`）；结算超时标记 `INCOMPLETE`，不显示完整归属伤害。

默认第三人称，`C` 切换己方相机，方向键环绕，`PageUp/PageDown` 缩放；空格暂停，`-`/`+`
选择 0.25、0.5、1、2 倍速，`R` 从头重播。末帧会保留，关闭窗口退出。黄色线段是采样轨迹，
红色标记表示该机器人受到伤害，并非精确撞击点。回放不推进物理，变速和重播不会改变计分。

分析和查看命令均可用 `--viewer-binary PATH` 指定回放程序。单模型查看命令还支持
`--label TEXT` 设置窗口及 HUD 标签。修改后需重新构建 `training_preview`，以支持角色标签。
无图形会话、浏览器不可用或程序缺失时保留数据，打印报告路径及回放打开命令；推理或显示
失败单独报错，已完成的训练和模型不受影响。分析工具只接受已完成的训练目录；评估故障不发布
完整回放文件，进程日志留在该次评估目录供定位问题。

### PPO 奖励学习信号诊断

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

报告将 Gym 原始奖励与时间截断的价值补偿分开显示。奖励移除只在数据副本上进行，保持状态、
价值预测与回合边界不变；它验证数值传播，不代表“不发射时”的环境反事实。正优势样本也不保证
整轮更新后该动作概率一定增加；需结合全部样本、实际小批次标准化和裁剪一起判断。
解释方差的目标方差为零时显示暂无数据。未完成回合不追加尾部结算，回合末在途弹丸单独计数。

正常关闭、Ctrl+C、SIGTERM 或环境异常均清理所属进程；故障保留已完成记录并标注状态，不保存
部分更新为有效模型。首轮一致性对照失败时停止后续采样。针对性数值及更新一致性检查：

```bash
artifacts/venv/bin/python -m unittest discover -s tests -p test_diagnose.py -v
```

### 手动发射与 Gym 奖励调试

手动窗口复用真实 PPO Gym 环境。视觉自动跟踪／瞄准，用户控制连续发射与暂停，不加载策略，
不训练模型。先按上文重新构建 `training_preview`，然后在 `rm_vision_rl` 根目录启动：

```bash
artifacts/venv/bin/python -m tools.training.manual
# 使用已有模型保存的环境配置与契约，不执行模型推理
artifacts/venv/bin/python -m tools.training.manual \
  --checkpoint artifacts/training/rotation-v1/final.zip
# 显式延长回合；窗口和运行记录会标注覆盖值
artifacts/venv/bin/python -m tools.training.manual \
  --config config/training/rotation_joint_ppo.json --episode-steps 1000
```

`--config` 与 `--checkpoint` 互斥，省略时使用默认训练配置。`--checkpoint` 先验证原环境／观测
契约，再应用可选的 `--episode-steps`；环境或视觉配置指纹变化仍拒绝运行。
`--output-dir DIR` 指定独立会话目录的父目录，`--viewer-binary PATH` 指定查看器。
需要图形桌面；无图形环境、查看器缺失或版本不支持手动模式时，不启动仿真进程。

| 按键 | 行为 |
| --- | --- |
| 按住 `F` | 每个环境步持续请求发射；松开停止请求，继续自动跟踪 |
| `Space` | 暂停／继续；暂停时 F 不推进时间，也不积攒请求 |
| `R` | 生成下一场景（静止靶重复原场景）、禁射预热并清零奖励，预热完成后自动运行 |
| `C`、方向键、`PageUp/PageDown` | 切换相机、环绕及缩放 |

预热完成后默认按正常仿真速度连续运行，无需按键推进时间；原 N／F 单步操作已移除。
按住 F 时每一步都请求发射，松开后下一步恢复跟踪；只有一个在途环境操作，不排队保存发射。
暂停后已提交的一步允许完成；暂停期间按 F 不推进环境。重置最多保留一个待处理请求，忙时
不会重复排队。每步仍为 10 ms，计算不足时放慢，不跳过物理步骤。

HUD 的 `Step reward` 是最近一步原始 Gym 奖励，`Total reward` 是当前回合逐步奖励之和，
`Last nonzero` 保留最近一次非零奖励和步数。它们不包含价值自举或独立评估尾部结算。
奖励发生在实际命中的步骤，不一定是按下 F 的步骤；松开 F 后继续运行即可观察在途弹丸命中。
窗口还分别显示当前掩码、上次动作是否被屏蔽、发射请求、请求接受情况、
实际出膛数及已有拒绝原因。掩码禁止时仍沿用 Gym 的跟踪替代行为，不会绕过机械或裁判限制；
持续按住 F 时下一步重新请求，松开后不补发已被屏蔽的请求。回合结束冻结显示，等待 R，不自动重置或追加结算。

默认产物在 `artifacts/manual/manual-*/`：`run.json` 保存配置、覆盖值和状态，`steps.csv`
逐步记录动作、掩码、请求接受情况、出膛数及奖励，`episodes.monitor.csv` 记录完整回合，
环境目录内 `scenes.jsonl` 记录每回合场景序号、复现种子、角速度、实际出生和拒绝原因；
另保存查看器和环境进程日志；不默认录制完整画面。异常时保留最后画面并标记 `FAULT`，
物理进程停止，不自动重试发射。关闭窗口、Ctrl+C 或 SIGTERM 清理该会话拥有的全部进程。

## 4. 手动运行底层物理环境

终端 A：

```bash
cd /home/dgx_spark/RM_VISION_WORKSPACE/rm_simulator_2027
RL_SOCKET_DIR=$(mktemp -d /tmp/rm-rl-manual-XXXXXXXX)
echo "$RL_SOCKET_DIR/training.sock"
cargo run --offline --no-default-features --features training \
  --bin daedalus_training -- --socket "$RL_SOCKET_DIR/training.sock"
```

程序无窗口，等待客户端请求。`cargo run` 设置动态库搜索路径；还可传 `--config PATH`、`--assets PATH`、`--physics-step-us 1000`。

终端 B：将示例 socket 改为终端 A 实际打印的路径。

```bash
cd /home/dgx_spark/RM_VISION_WORKSPACE/rm_vision_rl
python3 - <<'PYCODE'
from src.transport.client import TrainingClient

client = TrainingClient('/tmp/rm-rl-manual-XXXXXXXX/training.sock')
try:
    state = client.reset(17)
    print('initial:', state['round_id'], state['sim_time_ns'])
    for _ in range(100):
        state = client.advance(yaw_rad=0.1, pitch_rad=0.0, fire=False)
    print('after 100 steps:', state['sim_time_ns'], 'ns')
    print('inspect:', client.inspect()['sim_time_ns'])
finally:
    client.close()
PYCODE
```

100 次 Advance 应推进到 `1_000_000_000 ns`。Inspect 不推进，Close 结束进程。本例直接指定云台命令，没有目标跟踪。服务端不覆盖已有 socket，每次使用新目录可避免冲突。

## 5. 运行完整规则闭环

以下示例自动启动和关闭两个进程，不要提前手动启动环境。正式窗口为 3 秒，另有最多 15 秒仿真预热和最多 10 秒尾部结算；实际耗时由机器速度决定。

```bash
cd /home/dgx_spark/RM_VISION_WORKSPACE/rm_vision_rl
python3 - <<'PYCODE'
import json
import tempfile
from pathlib import Path
from src.transport.processes import training_worker
from src.transport.vision_bridge import vision_worker
from src.environment.evaluation import EvaluationConfig, EvaluationSession

root = Path.cwd()
sim = root.parent / 'rm_simulator_2027'
vision = root.parent / 'rm_vision_2027'
sim_binary = sim / 'target/release/daedalus_training'
bridge_binary = root / 'artifacts/build/vision-bridge/rmvision-rl-bridge'
(root / 'artifacts').mkdir(exist_ok=True)
output = Path(tempfile.mkdtemp(prefix='run-', dir=root / 'artifacts'))
print('output:', output, flush=True)

with training_worker(sim, sim_binary, output / 'simulator.log') as client, \
     vision_worker(bridge_binary, vision, output / 'bridge') as bridge:
    session = EvaluationSession(client, bridge, EvaluationConfig(window_ms=3000))
    session.reset(17, {'motion': {'kind': 'static'}})
    while session.status in ('warming', 'evaluating', 'settling'):
        session.advance()
    result = session.info()
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
PYCODE
```

示例使用视觉项目当前默认模块配置。`complete` 表示评估和结算结束，不保证命中或效果达标；`warmup_timed_out` 表示未就绪，`settlement_timed_out` 表示结算未完成。异常时查看 `simulator.log` 和 `bridge/stderr.log`。

`training_worker` 通过 rustc 查找 Rust 动态库目录，并使用二进制旁的 `deps/`。只复制仿真可执行文件到其他机器通常不够。桥接日志和配置写入本次输出目录。

## 6. 接入 Python 策略

把上例创建会话的位置替换为：

```python
def track_only(observation):
    mask = observation['action_mask']
    return next((i for i in (1, 3, 5, 7, 0) if mask[i]), 0)

session = EvaluationSession(
    client, bridge, EvaluationConfig(window_ms=3000),
    policy=track_only, policy_mode='nine',
)
```

这个回调只跟踪、不申请射击，不会学习。`fire_only` 模式由规则策略选板，回调在当前槽位的 TRACK/FIRE 间选择。需要编码后的短历史时使用：

```python
from src.policy.observations import TensorPolicy

def actor(value):
    # value 包含 features、valid、action_mask 和 version。
    return next((i for i in (1, 3, 5, 7, 0) if value['action_mask'][i]), 0)

session = EvaluationSession(
    client, bridge, EvaluationConfig(window_ms=3000),
    policy=TensorPolicy(actor), policy_mode='fire_only',
)
```

动作和数据语义见[接口说明](interfaces.md)。

## 视觉配置

运行读取的是传给 `vision_worker` 的目录下 `src/config/modules/`，MPC 求解迭代上限位于
`gimbal_trajectory_planner.yaml` 的 `max_iterations`。需要实验配置时可手动准备副本，
将副本根目录作为 `vision_root`，并记录实际配置。

`config/policy/mpc-recovery.json` 保留为 50 → 200 次迭代的候选参数记录，不会被运行库自动加载；
本项目不再提供配置副本生成工具。修改实际视觉配置可能导致旧检查点的配置指纹校验失败。

## 常见问题

2026-09-13 修复了连发吞吐偏低的两个环境问题：机械冷却误用低频裁判快照，以及合成检测将经过视线的小弹丸判为整块装甲不可见。更新后重新构建 `daedalus_training`，重启已有手动窗口/环境进程；仅更新 Python 无法修复旧仿真二进制的视觉行为。手动、Gym 和规则评估共用修复后的环境。射速限制、奖励和物理命中结算未放宽。

旧检查点可以继续加载，网络维度和配置格式兼容，但修复前后的运行环境行为不同，历史分数不能视为同一实现上的复现。现有配置指纹不包含源码/二进制摘要，比较时应另外记录运行版本；修复本身不会让旧 PPO 权重自动学会持续发射。5 秒 Gym 奖励只计关窗前伤害，独立评估还会结算窗口内射出但尚未命中的弹丸，比较时须区分两种口径。

| 现象 | 处理方向 |
| --- | --- |
| 训练期间没有画面 | 训练过程无渲染；运行 `tools.training.analysis` 分析并显示结果，或用 `tools.training.view --checkpoint` 查看单个模型 |
| 找不到 Bevy/Rust 动态库 | 使用 `cargo run` 或 `training_worker`，保留匹配工具链和 `deps/` |
| `No module named src` | 从本仓库根目录执行 |
| 缺少桥接二进制 | 完成 CMake 构建，核对输出路径 |
| 预热超时或长期无有效命令 | 检查场景、测量、跟踪及 MPC 配置；保留超时，不静默重复抽样 |
| 找不到旧验收命令 | 历史工具已删除，使用本文运行库示例 |

2026-09-23 的旋转出生可见性修复会改变部分 seed 的接纳结果。旧检查点的观测/动作和
配置指纹仍兼容，可恢复权重及优化器；修复前后的场景分布不能当作完全一致的评估条件。

## 联合选板模式与成对比较

`rotation_joint_ppo.json` 设定 `environment.action_mode="joint"`：每 10 ms 选择装甲板，
射击机会保持随机 50–100 ms。预热使用规则且禁射，正式采样由 RL 自主选板，不套用规则面向角门限。
使用 `--config config/training/rotation_fire_ppo.json` 可新建两动作基线；`--resume` 始终使用保存模式。
不能把旧两动作权重恢复为联合模型。第一次验证建议 `--timesteps 1024`，新建独立输出目录。

```bash
artifacts/venv/bin/python -m tools.training.compare \
  --checkpoint <联合模型.zip> --baseline-checkpoint <旧两动作模型.zip> \
  --output-dir artifacts/comparison
```

默认执行 3、5、7 m × -7、-3、-1、1、3、7 rad/s，共 18 个场景，每种策略正式运行 5 秒并自然结算。
双方复用合法出生种子、场景和时钟；每次运行保留模型快照、实际参数和回放。
查看 `comparison.html` 或 `comparison.csv`，失败场景有明确状态，不能把部分完成结果解释为完整成绩。
`tools.training.view --replay <回放.json>` 可打开指定场景，HUD 显示槽位、切板数和协议动作。
`tools.training.view --checkpoint <模型.zip> --no-view` 只评估并保存参考场景回放，不打开窗口。

手动窗口中 1–4 更新期望槽位，F 请求该槽位射击；未选板时等待。重复选同板即保持。
非法输入在 Gym 降级并显示掩码原因；R 清除期望槽位并生成下一场景，需要重新按数字键。
释放 F 不撤回已经接受的脉冲。旧两动作配置无需数字键，沿用 F 和规则选板。

构建更新后的 bridge 与 training_preview 后再运行工具。契约检查：
`artifacts/venv/bin/python -m unittest discover -s tests -p test_joint_contract.py -v`。
