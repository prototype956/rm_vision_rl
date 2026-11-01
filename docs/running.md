# 构建与运行

本页面向 Linux 服务器与 NUC，命令使用 Bash。默认三个仓库同级放置，从 `rm_vision_rl` 根目录执行，明确标注的 `cd` 除外。先完成构建，再选择对应课程流程；部署与回放只使用已有模型。

- [依赖与构建](#依赖与构建)
- [训练、恢复和产物](#ppo-训练恢复和推理)
- [GRU MCAP 回放](#gru-原生-foxglove-回放)
- [手动调试](#手动发射与-gym-奖励调试)
- [静止靶采样](#静止靶-gym-采样)、[底层物理](#手动运行底层物理环境)、[规则闭环](#运行完整规则闭环)

历史 MLP 的推理、选优、成对比较和学习信号诊断已移至[独立文档](legacy_mlp.md)。

## 依赖与构建

- Python 3.10 或更高版本；旧规则/回调运行库使用标准库；Gym 环境需要 requirements.txt 中的 Gymnasium 和 NumPy，不要求 PyTorch。
- 仿真项目可用的 Rust/Cargo 工具链、Cargo 依赖及 `assets/`。
- 支持 C++20 的编译器、CMake、Eigen3、OpenCV（core/calib3d/imgproc）、Ceres、yaml-cpp、fmt、spdlog 和线程库。

构建仿真训练进程：

```bash
# 从 rm_vision_rl 根目录进入相邻仓库。
cd ../rm_simulator_2027
cargo build --offline --release --no-default-features --features training --bin daedalus_training
```

`--offline` 使用已有 Cargo 缓存；未缓存的依赖需要联网去掉此选项构建。当前 Bevy 仍启用动态链接，无窗口运行不意味着已经裁掉全部图形相关构建依赖。

构建桥接：

```bash
cd ../rm_vision_rl
cmake -S native/vision_bridge -B artifacts/build/vision-bridge \
  -DVISION_ROOT="$(realpath ../rm_vision_2027)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build/vision-bridge --parallel 2
```

输出分别是仿真项目的 `target/release/daedalus_training` 和本项目的 `artifacts/build/vision-bridge/rmvision-rl-bridge`。

若 Ceres 等依赖使用本工作区已有的本地安装，在上述 CMake 配置命令中增加：

```bash
-DCMAKE_PREFIX_PATH="$(realpath ../rm_vision_2027/.deps/vision/usr)"
```

## PPO 训练、恢复和推理

配置目录包含步兵、哨兵各自的 v2/v3 预设。训练 CLI 默认仍选择 v2；本节示例使用 v2。默认步兵：
`config/training/rmul_2027_infantry_curriculum_v2.json`；
哨兵：`config/training/rmul_2027_sentry_curriculum_v2.json`。

两份预设均使用 `recurrent_maskable_ppo/asymmetric_gru`、v5 单步观测、12 阶段课程和热锁奖励。
默认 `cuda:0`、128 环境、每类累计十亿步；显式请求 CUDA 而不可用时会报错。
控制周期 10 ms、射击机会 50–100 ms、回合最多 30 秒，正常计算热量及弹药消耗。
这两份配置是长期训练设置；详细阶段与运行要求见 [课程 v2](curriculum_training_v2.md)。

```bash
python -m pip install -r requirements-training.txt
python -m src.training.train --help

# 默认步兵课程。
python -m src.training.train --output-dir artifacts/training/infantry-v2
# 哨兵课程。
python -m src.training.train \
  --config config/training/rmul_2027_sentry_curriculum_v2.json \
  --output-dir artifacts/training/sentry-v2
# 恢复只完成原累计预算中剩余的部分。
python -m src.training.train \
  --resume artifacts/training/infantry-v2/latest.zip \
  --output-dir artifacts/training/infantry-v2-resumed
```

所有输出目录必须是新目录。新训练可用 `--timesteps` 覆盖累计目标、`--n-envs` 覆盖工作进程数；
课程恢复不允许改变累计目标、环境数或课程。模型、优化器、课程和随机流从检查点恢复，
各工作进程从新回合与零隐藏状态开始。采集完成当前回合才统一更新，实际步数可能超过目标。

直接训练会先进行场景预检。正式流程可先冻结场景，再进行两类 128 环境资格验证：

```bash
python -m tools.training.curriculum_v2 preflight --output artifacts/training/v2-preflight
python -m tools.training.curriculum_v2 benchmark \
  --scenarios artifacts/training/v2-preflight --output artifacts/training/v2-qualification
python -m tools.training.curriculum_v2 run \
  --scenarios artifacts/training/v2-preflight --output artifacts/training/v2-run
```

预检、资格验证和最终评估不计入正式预算。冻结源码、环境与资产及中断恢复见
[课程 v2](curriculum_training_v2.md)。通用 `tools.training.benchmark` 的默认输入也改为步兵 v2；
该工具会执行训练入口完整的课程预检和评估，不能将其短步数设置视为轻量资格验证。

奖励为实际伤害乘 `0.05`，再减去选板成本、普通/整局热锁触发成本和锁定时长成本。
真实伤害、命中、资源与净奖励分别记录。`runtime` 默认启用分块数据、有限诊断、日志轮转、
滚动检查点、里程碑 Actor 导出及磁盘不足暂停。预算、战斗门槛和热锁门槛分开报告。

主要产物：

- `run.json`：生效配置、状态、累计步数和最终各阶段结果。
- `logs/progress.csv` 和 TensorBoard：更新指标、真实伤害、资源和净回报。
- `curriculum/scenarios.json`：冻结验证/最终测试场景；逐场结果位于 `curriculum/`。
- `latest.zip`、滚动及里程碑 ZIP：可恢复的完整检查点。
- `updated-pending-validation.zip`：已更新、尚待完成该次验证的状态。
- `actor/`：TorchScript Actor 与部署清单，不包含 Critic。
- `episodes.jsonl.gz` 和有界 `rollouts/`：完整回合汇总与抽样诊断。

```bash
python -m tools.training.curriculum_report --run artifacts/training/v2-run
python -m tools.training.recurrent_deploy \
  --actor artifacts/training/infantry-v2/actor --episodes 2 \
  --output artifacts/deployment-check/infantry-v2
```

部署 API、状态重置和导出核对见 [持续记忆 GRU](persistent_gru.md)。
旧 `analysis/view/compare/diagnose` 模型工具供已有 MLP 检查点使用，不用于 GRU。
已有独立回放仍可播放。历史 MLP、静止/旋转靶实现保留，但不再提供其训练预设。

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
仍需要本机图形桌面与原生窗口。训练预算可用总采样步数或 PPO 更新次数表示；页面显示预算估计；GRU 在完整回合边界更新，`n_steps × n_envs` 只表示采集下限，实际更新步数可能更大。

新训练可覆盖预算、并行环境数、设备、seed 和 checkpoint 保存间隔。课程恢复锁定
原累计目标、环境数、课程、奖励、策略和 seed，使用相同预算与环境数并写入新目录；
页面的追加预算选项只用于旧非课程检查点。GRU 不启用旧 MLP 自动分析。任务 sidecar、进程日志和服务重启后的任务信息位于 `artifacts/training/.web-jobs/`。
默认只绑定本机并限制同时运行一个训练或工具任务；课程训练内部执行验证与最终测试；页面上的独立 MLP 分析不适用于 GRU。

v3 配置还需要传统火控校准和发布证据。其预检、显式迁移、128 环境资格验证与冻结运行命令见[课程 v3](curriculum_training_v3.md#命令)，不要将未校准预设直接当作正式发布配置。

## GRU 原生 Foxglove 回放

显式评估任务可直接录制与视觉项目同话题、同 Schema 的 MCAP，不启动训练或自动录制。
从本项目根目录运行（课程与 case 编号均从 0 开始）：

```bash
python -m tools.training.view --checkpoint artifacts/training/infantry-v2/latest.zip \
  --record-mcap --scenarios artifacts/training/infantry-v2/curriculum/scenarios.json \
  --split validation --course 0 --cases 0 1 --output-dir artifacts/evaluation --no-view
python -m tools.training.view --actor artifacts/deployment/infantry-v2 \
  --record-mcap --scenarios artifacts/training/infantry-v2/curriculum/scenarios.json \
  --course 0 --cases 0 --output-dir artifacts/evaluation --no-view
```

Checkpoint 与 Actor 互斥；目前 MCAP 模型评估支持 asymmetric_gru，Actor 使用 CPU 部署推理。
冻结清单必须通过 SHA256 检查；所选姿态只尝试一次，失败记录不会被换场掩盖。
工具固定模型副本，各场景使用新 GRU 状态，依次执行，`workers` 不影响此录制模式。
产物位于新建 `foxglove-*` 目录，每场景包含 `replay.mcap` 与 `summary.json`；
`replay_index.json` 汇总成功、失败及结算状态。现有控制台“回放 / 评估”提供相同入口和下载链接。

MCAP 包含实际 PnP、预测、控制周期和采集时刻真值。画面是标明来源的合成测量画布，
不代表真实相机渲染；检测器、灯条提取、图像角点精修和实时期限验收不适用。
物理过程按仿真时钟录制，不做墙钟限流；慢磁盘只会降低离线评估速度。
`summary.window` 保留训练窗口成绩，`summary.settlement` 按出膛窗口归属独立结算。
结算上限为 1000 个物理步，超时 `official_damage=null`，任务标记未完整完成。
回放录制完整性与结算完整性分开记录；异常文件保留，但不能作为成功产物。

使用现有 Foxglove 面板或视觉仓库 `.agents/skills/rm-vision-mcap/scripts/mcap_analyze.py`
分析文件；打开 MCAP 不需模型、训练进程或仿真器。旧 `--replay replay.json` 行为保持不变。

## 手动发射与 Gym 奖励调试

当前课程配置可在配置模式手动操作 C0 场景，不更新模型或课程；每次 R 从下一个种子生成场景：

```bash
python -m tools.training.manual --mode config \
  --training-config config/training/rmul_2027_infantry_curriculum_v2.json
```

以下检查点实时推理示例用于已有 MLP 模型；GRU 本地部署使用 `tools.training.recurrent_deploy`。


手动窗口复用真实 PPO Gym 环境。视觉自动跟踪／瞄准，用户控制连续发射与暂停，不训练模型。
先在仿真仓库构建查看器，再回到 `rm_vision_rl` 根目录启动：

```bash
(cd ../rm_simulator_2027 && cargo build --offline --release --no-default-features \
  --features training --example training_preview -j 6)
```

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
| `1`–`4` | 联合模式选择槽位；R 后需重新选择 |
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

## 静止靶 Gym 采样

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

该示例检查环境采样链路。伤害为零时，应结合实际出膛、跟踪和控制日志判断，不能单独据此认定接口失效。

旋转靶可直接通过 Gym 使用同一采样器：

```python
import gymnasium as gym
import src

with gym.make('RMRotationFire-v0', episode_steps=500,
              angular_speed_range_rad_s=[1, 7]) as env:
    obs, info = env.reset(seed=17)
    # 完成回合后 env.reset() 取下一场景；reset(seed=17) 从头复现场景序列。
```

## 手动运行底层物理环境

终端 A：

```bash
cd ../rm_simulator_2027
RL_SOCKET_DIR=$(mktemp -d /tmp/rm-rl-manual-XXXXXXXX)
echo "$RL_SOCKET_DIR/training.sock"
cargo run --offline --no-default-features --features training \
  --bin daedalus_training -- --socket "$RL_SOCKET_DIR/training.sock"
```

程序无窗口，等待客户端请求。`cargo run` 设置动态库搜索路径；还可传 `--config PATH`、`--assets PATH`、`--physics-step-us 1000`。

终端 B：将示例 socket 改为终端 A 实际打印的路径。

```bash
# 在 rm_vision_rl 根目录运行。
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

## 运行完整规则闭环

以下示例自动启动和关闭两个进程，不要提前手动启动环境。正式窗口为 3 秒，另有最多 15 秒仿真预热和最多 10 秒尾部结算；实际耗时由机器速度决定。

```bash
# 在 rm_vision_rl 根目录运行。
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

## 接入 Python 策略

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

早期实验曾记录 50 → 200 次迭代的候选参数；当前仓库未提供该候选配置文件，也不自动生成配置副本。修改实际视觉配置可能导致旧检查点的配置指纹校验失败。

## 常见问题

2026-09-13 修复了连发吞吐偏低的两个环境问题：机械冷却误用低频裁判快照，以及合成检测将经过视线的小弹丸判为整块装甲不可见。更新后重新构建 `daedalus_training`，重启已有手动窗口/环境进程；仅更新 Python 无法修复旧仿真二进制的视觉行为。手动、Gym 和规则评估共用修复后的环境。射速限制、奖励和物理命中结算未放宽。

旧检查点可以继续加载，网络维度和配置格式兼容，但修复前后的运行环境行为不同，历史分数不能视为同一实现上的复现。当时的旧配置指纹未包含源码/二进制摘要；当前 GRU 恢复另核对运行身份，比较历史结果仍须记录版本。修复本身不会让旧 PPO 权重自动学会持续发射。5 秒 Gym 奖励只计关窗前伤害，独立评估还会结算窗口内射出但尚未命中的弹丸，比较时须区分两种口径。

| 现象 | 处理方向 |
| --- | --- |
| 训练期间没有画面 | 训练过程无渲染；GRU 使用本文 MCAP 回放入口；历史 MLP 使用 `analysis` 或 `view --checkpoint` |
| 找不到 Bevy/Rust 动态库 | 使用 `cargo run` 或 `training_worker`，保留匹配工具链和 `deps/` |
| `No module named src` | 从本仓库根目录执行 |
| 缺少桥接二进制 | 完成 CMake 构建，核对输出路径 |
| 预热超时或长期无有效命令 | 检查场景、测量、跟踪及 MPC 配置；保留超时，不静默重复抽样 |
| 找不到旧验收命令 | 历史工具已删除，使用本文运行库示例 |

2026-09-23 的旋转出生可见性修复会改变部分 seed 的接纳结果。旧检查点的观测/动作和
配置指纹仍兼容，可恢复权重及优化器；修复前后的场景分布不能当作完全一致的评估条件。

显式 MCAP 回放现在默认附带可开关的双视角战斗 HUD，配置 `combat_hud=false` 或 `--no-combat-hud` 可关闭。安装独立 `requirements-replay.txt` 依赖并按 [HUD 图像面板说明](foxglove_rendered_replay.md#可开关战斗-hud) 启用对应注释、关闭同步时间戳；常规训练不录制。
