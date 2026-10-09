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
  --config config/training/rotation_joint_shared_score_ppo.json \
  --output-dir artifacts/training/rotation-v1
```

输出目录必须尚不存在；省略时自动在 `artifacts/training/` 下创建唯一目录。
配置默认 65536 个采样步、每环境 1024 步、minibatch 256、10 epochs、两层 128 单元
网络，以及 16 个并行环境。`--timesteps` 覆盖本次预算，向上取整到完整 rollout 并在
启动时显示；`--seed` 只改变 PPO 随机种子，场景种子另由配置的 `environment.scene_seed` 控制。
配置中的相对环境路径相对于本仓库根目录解析。

多个独立环境同时采样（物理步长仍为 1 ms，控制步长仍为 10 ms）：

```bash
artifacts/venv/bin/python -m src.training.train \
  --config config/training/rotation_joint_shared_score_ppo.json \
  --n-envs 4 \
  --output-dir artifacts/training/joint-parallel
```

这里通过 `--n-envs 4` 覆盖并行环境数量，每环境 `ppo.n_steps` 仍由唯一训练配置控制。
省略顶层 `n_envs` 时使用配置中的默认值；大于 1 时使用
`SubprocVecEnv` 的 `spawn` 多进程，每个环境独占仿真器、视觉桥接、socket 和日志。
各环境同时推进，主进程批量推理并同步等待全部返回，因此最慢的环境仍可能拖慢一批采样。
rank 0 保留原场景和时钟种子，rank i 使用场景种子加 i、时钟种子加 i 后对 2³² 取模。
检查点的 `sampling` 字段记录各 worker 的种子，评估仍使用单环境。

`--n-envs N` 可覆盖环境数，但**不会自动缩短每环境的 n_steps**。例如单环境默认配置加
`--n-envs 4` 后，每次更新是 4096 个样本，预算按 4096 向上取整，检查点间隔也相应变大。
要保持每次更新 1024 个样本，使用上述并行配置。缩短轨迹会改变 GAE 截断位置；吞吐提升
不能直接视为达到相同策略质量所需时间的提升，仍需等样本预算的独立评估。

已补测 8、16 环境，结果见[性能报告](training_performance.md)。保持每次更新 1024 样本时，
分别使用 `n_envs=8, ppo.n_steps=128` 和 `n_envs=16, ppo.n_steps=64`；默认四环境配置保留。

复现完整训练吞吐对照（1、2、4 环境；每组 2048 样本，两轮交替顺序）：

```bash
artifacts/venv/bin/python -m tools.training.benchmark \
  --output-dir artifacts/performance/parallel-check \
  --n-envs 1 2 4 --timesteps 2048 --repeats 2
```

该工具默认固定输入配置的每次更新总样本数，输出逐次训练产物、`results.json` 和 `report.md`。
采样时间含批量推理、通信、环境推进及自动 reset；训练时间还含首次 reset、PPO 更新和日志；
端到端时间另含解释器与 worker 启动、最终模型保存和关闭。它是短期速度实验，不用于判断收敛。

若要只增加环境数量，保持原配置的每环境 `n_steps=1024`，使用 `--rollout-mode per-env`。
下面每组均执行两轮完整采样与 PPO 更新：单环境共 2048 样本，16 环境共 32768 样本。
`--updates` 与 `--timesteps` 互斥；前者固定完整更新轮数，后者指定总样本预算并向上取整。

```bash
artifacts/venv/bin/python -m tools.training.benchmark \
  --config config/training/rotation_joint_shared_score_ppo.json \
  --rollout-mode per-env --updates 2 --n-envs 1 16 --repeats 2 \
  --output-dir artifacts/performance/fixed1024-check
```

该实验比较聚合样本/秒，各组的总样本预算不同，不能用模型伤害直接比较学习质量。
按此设置正式训练可在原配置上指定 `--n-envs 16`，每轮更新采集 16384 个总样本：

```bash
artifacts/venv/bin/python -m src.training.train \
  --config config/training/rotation_joint_shared_score_ppo.json --n-envs 16 \
  --output-dir artifacts/training/joint-16env-fixed1024
```

当前唯一训练配置使用 16 个环境，20 点伤害对应 1 分，并包含选板质量奖励：

```bash
artifacts/venv/bin/python -m src.training.train \
  --config config/training/rotation_joint_shared_score_ppo.json \
  --output-dir artifacts/training/joint-reward1-new
```

共享评分配置包含切板基础成本、快速往返成本和预测朝向机会损失。训练奖励为
`damage * reward_scale - base_switch_penalty - return_penalty - facing_penalty`，独立评估和回放仍统计真实伤害。

顶层 `reward_scale` 是有限正数，此配置为 `0.05`。
仅训练入口在 Monitor 统计及 PPO 价值自举之前缩放伤害奖励，20 点伤害对应 1 分。
Monitor 的 `r` 和 TensorBoard 的 `rollout/ep_rew_mean` 为缩放后减去切板惩罚的净训练奖励；
`episode_damage`、`rollout/ep_damage_mean`、独立评估和回放仍为真实伤害。
直接使用原生 Gym/手动窗口仍显示原始伤害奖励。物理环境指纹和旧模型加载规则不变。
配置随检查点保存，`--resume` 会继续使用保存的缩放比例；改奖励尺度的实验应从头训练，
不直接恢复旧尺度的价值网络。新旧训练的 reward/value_loss 数字不可直接横比，应比较真实伤害。

新训练默认配置为 `config/training/rotation_joint_shared_score_ppo.json`：每回合重新生成双方位置和朝向、
2–8 米目标距离、目标方位，以及双向 1–7 rad/s 的匀速旋转。省略初始云台角度时朝向目标。
旋转从禁射预热开始，进入正式采样时不重置姿态。出生使用仿真器几何和初始可见性检查；
非零旋转还需通过每 5° 的整圈可见性采样，避免高台/障碍物只在初始相位短暂露出装甲。
更新此检查后需要重新构建 `daedalus_training` 和 `training_preview`。被拒绝的种子记录在
`reset_attempts` 中，使用现有最多 32 次出生重试；不会跳过真正的预热超时。
`environment.scene_seed`（默认 17）控制可复现的场景序列，自动 Reset 和手动 R 都取下一场景。
`environment.angular_speed_range_rad_s` 配置速度大小范围，例如 `[1, 3]`；正反方向各半，
要求 `0 < min <= max <= 7`。不要设置 `scenario.motion`，它由环境采样生成。
静止靶配置已从训练配置目录移除；已有静止靶检查点仍可继续评估。
恢复旋转靶训练从保存种子的首个场景重新开始，不接续中断前的场景随机流。
默认 PPO 训练回合为 5 秒（500 步），预热单独计时；要调整回合长度、rollout 或 minibatch，编辑配置副本对应的
`environment.episode_steps`、`ppo.n_steps` 和 `ppo.batch_size`。minibatch 必须整除 `n_steps × n_envs`。

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
artifacts/venv/bin/python -m tools.training.manual --training-config config/training/rotation_joint_shared_score_ppo.json
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
- `episodes/monitors/episodes.monitor.csv`：单环境每个完整回合的净训练奖励（缩放后减切板惩罚）、奖励分项、真实伤害、实际出膛数、长度及结束原因。
- 多环境使用 `episodes/monitors/episodes.worker-NNN.monitor.csv` 和 `environment/worker-NNN/`，避免输出冲突；逐步奖励写入对应的 `episodes/rewards/*.rewards.jsonl`。分析工具读取全部 worker 日志，按同步采样的全局样本数定位回合。
- 新训练的回合产物统一放在 `episodes/monitors/` 和 `episodes/rewards/`；读取工具兼容旧版直接放在运行目录根部的 `episodes*.monitor.csv` 文件。
- `environment/env-*/decision-clock.jsonl`（单环境）或 `environment/worker-NNN/env-*/decision-clock.jsonl`（多环境）：启用时钟时的逐次射击机会、随机间隔、原火控合法性和请求接受结果。
- `logs/progress.csv` 和 TensorBoard event：每次完整更新后的损失、KL、熵损失、采样速度和回合均值。
- `time/sampling_seconds`、`time/optimization_seconds`、`time/learn_seconds` 分别记录本次更新的采样、PPO 优化和完整 learn 耗时；`time/n_envs`、`time/rollout_samples` 记录实际规模。
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
只允许覆盖预算、设备、环境数和输出目录。`--n-envs` 可将旧单环境检查点恢复到多环境，
保持检查点的每环境 `n_steps`，不会恢复旧 rollout；所有 worker 从各自种子的首个场景重新开始。
观测规格或环境/视觉配置指纹变化时拒绝恢复。
Ctrl+C、SIGTERM 或运行故障会退出并清理进程，保留已有完整检查点，不保存半次更新。

查看训练曲线：

训练 CLI 默认在训练结束后生成并打开本地交互报告，包含完整回合奖励、最近 20 回合移动平均、
伤害、实际出膛数、PPO 指标及独立评估模型对比。前 19 回合的移动平均使用已有回合；缺失或
非有限指标显示空缺，不补零。横轴为累计训练步数，恢复训练的回合曲线加上起始偏移。
报告内嵌 Plotly，可离线打开、悬停查看数值、缩放和切换图例。原有 TensorBoard 仍可使用：

```bash
artifacts/venv/bin/tensorboard --logdir artifacts/training --port 6006
```

### 本地训练 Web 控制台

需要在训练环境之外安装可选的 Web 依赖：

```bash
artifacts/venv/bin/python -m pip install -r requirements-web.txt
artifacts/venv/bin/python -m tools.training.dashboard
```

然后访问 `http://127.0.0.1:8765`。控制台使用现有训练入口启动独立子进程，支持选择
`config/training/` 预设、新建训练、查看实时步数和指标、请求优雅停止、从完整 checkpoint
继续训练，并在 Tools 页面按需启动 `analysis`、`compare`、`diagnose`、`benchmark`、离线 `view/replay`
或本机 `manual`。这些任务复用既有 CLI，报告、JSON、CSV 和回放可从任务详情打开；manual/live viewer
仍需要本机图形桌面与原生窗口。训练预算可用总采样步数或 PPO 更新次数表示；页面会显示
`n_steps × n_envs`、预计更新数和向上取整后的实际预算。

新训练只允许覆盖预算、并行环境数、设备、seed 和 checkpoint 保存间隔。恢复训练锁定
checkpoint 中保存的环境、奖励、策略和 seed，只允许追加预算、设备、并行环境数，并始终写入新的
训练目录。任务 sidecar、进程日志和服务重启后的任务信息位于 `artifacts/training/.web-jobs/`。
默认只绑定本机并限制同时运行一个训练或工具任务；训练完成后不会自动评估，页面可手动启动分析。

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
选择 0.25、0.5、1、2 倍速，`R` 从头重播。多场景列表中 `[`/`]` 切换上一个／下一个场景，
切换从新场景第 0 帧开始；HUD 显示当前场景编号、随机种子、目标距离和目标转速。末帧会保留，关闭窗口退出。黄色线段是采样轨迹，
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

报告将缩放后的训练奖励与时间截断的价值补偿分开显示；物理伤害事件保持原单位，奖励接入检查会乘以保存的 reward_scale 并减去切板惩罚。奖励移除只在数据副本上进行，保持状态、
价值预测与回合边界不变；它验证数值传播，不代表“不发射时”的环境反事实。正优势样本也不保证
整轮更新后该动作概率一定增加；需结合全部样本、实际小批次标准化和裁剪一起判断。
解释方差的目标方差为零时显示暂无数据。未完成回合不追加尾部结算，回合末在途弹丸单独计数。

正常关闭、Ctrl+C、SIGTERM 或环境异常均清理所属进程；故障保留已完成记录并标注状态，不保存
部分更新为有效模型。首轮一致性对照失败时停止后续采样。针对性数值及更新一致性检查：

```bash
artifacts/venv/bin/python -m unittest discover -s tests -p test_diagnose.py -v
```

### 手动发射与 Gym 奖励调试

手动窗口复用真实 PPO Gym 环境。视觉自动跟踪／瞄准，用户控制连续发射与暂停，不训练模型。
先按上文重新构建 `training_preview`，然后在 `rm_vision_rl` 根目录启动：

```bash
artifacts/venv/bin/python -m tools.training.manual
# 使用工具配置中的检查点进行实时策略推理
artifacts/venv/bin/python -m tools.training.manual --config config/tools/manual.json
# 显式延长回合；窗口和运行记录会标注覆盖值
artifacts/venv/bin/python -m tools.training.manual \
  --episode-steps 1000
```

`--config PATH` 指定工具配置；手动工具配置中的 `config` 字段指定训练环境配置，`checkpoint` 字段指定模型。
命令行 `--training-config PATH` 和 `--checkpoint PATH` 可临时覆盖二者。`--checkpoint` 先验证原环境／观测
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
逐步记录动作、掩码、请求接受情况、出膛数及奖励，`episodes/rewards/*.rewards.jsonl` 记录逐步奖励，`episodes/monitors/*.monitor.csv` 记录完整回合，
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

当前唯一训练配置设定 `environment.action_mode="joint"`：每 10 ms 选择装甲板，
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

构建更新后的 bridge 与 training_preview 后再运行工具。契约检查：
`artifacts/venv/bin/python -m unittest discover -s tests -p test_joint_contract.py -v`。


### 选板质量与快速往返奖励（2026-09-26）

默认配置采用 `selection_reward`，与旧 `switch_penalty` 互斥；旧检查点仍使用原奖励。
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
  --config config/training/rotation_joint_shared_score_ppo.json \
  --n-envs 64 --timesteps 1048576 --output-dir artifacts/training/joint-selection-v3
```


### 共享评分策略

当前唯一保留的训练配置使用 `shared_score_mlp`，对四个候选装甲板使用同一评分器，
保持联合模式的九动作协议。它必须从头训练，不能恢复其他策略结构的检查点。

```bash
# 两个真实环境，各采集 1024 步；一次 PPO 更新，验证自动换场景。
artifacts/venv/bin/python -m src.training.train \
  --config config/training/rotation_joint_shared_score_ppo.json \
  --n-envs 2 --timesteps 2048 --output-dir artifacts/training/shared-score-smoke
# 加载检查点，评估后打开回放。
artifacts/venv/bin/python -m tools.training.view \
  --checkpoint artifacts/training/shared-score-smoke/final.zip
```

旧模型继续通过 `--resume` 或 `--checkpoint` 加载。单检查点评估、compare、diagnose、回放及手动
工具共用检查点配置，不需要另行切换动作协议。新结构改变策略表达及确定性解码，
短训练仅验证链路，是否改善命中率需固定场景对照验证。
