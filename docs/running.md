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
import rmvision_rl  # 注册 RMStaticFire-v0

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

## 3. PPO 训练、恢复和推理

训练依赖与基础 Gym 分开安装；在本仓库根目录执行：

```bash
artifacts/venv/bin/python -m pip install -r requirements-training.txt
artifacts/venv/bin/python -m rmvision_rl.training.train --help
```

当前固定 PyTorch 2.8.0、SB3/SB3-Contrib 2.9.0；PyTorch 的 PyPI ARM64 包支持本次 CPU
训练。默认 `device=cpu`，1 个 PyTorch 计算线程。显式指定 CUDA 而当前安装不可用时会报错，
不会悄悄回退。实际依赖版本保存在每次运行的 `run.json` 和检查点内。

开始一轮训练：

```bash
artifacts/venv/bin/python -m rmvision_rl.training.train \
  --config config/training/static_fire_ppo.json \
  --output-dir artifacts/training/static-front-v1
```

输出目录必须尚不存在；省略时自动在 `artifacts/training/` 下创建唯一目录。
配置默认 65536 个采样步、每 1024 步更新一次、minibatch 256、10 epochs、两层 128 单元
MLP，以及单环境 `DummyVecEnv`。`--timesteps` 覆盖本次预算，向上取整到完整 rollout 并在
启动时显示；`--seed` 只改变 PPO 随机种子，场景种子另由配置的 `environment.scene_seed` 控制。
配置中的相对环境路径相对于本仓库根目录解析。

每个回合都用 Gym 场景种子 17、4 m、目标方位/朝向和己方朝向均为 0 的正面静止靶，包括
向量环境自动 Reset；因此也会重复测量噪声。基础 Gym 的默认随机出生方式不变。
默认 PPO 训练回合为 5 秒（500 步），预热单独计时；要调整回合长度、rollout 或 minibatch，编辑配置副本对应的
`environment.episode_steps`、`ppo.n_steps` 和 `ppo.batch_size`。minibatch 必须整除 rollout。

运行输出包括：

- `run.json`：生效配置、模型/观测信息、配置指纹、依赖版本、预算和运行状态。
- `episodes.monitor.csv`：每个完整回合的原始奖励、伤害、实际出膛数、长度及结束原因。
- `logs/progress.csv` 和 TensorBoard event：每次完整更新后的损失、KL、熵损失、采样速度和回合均值。
- `checkpoint_<累计步数>.zip`：每 4 次完整更新保存一次；`latest.zip` 原子指向最近保存的完整状态。
- `final.zip`：正常结束时保存，同时更新 `latest.zip`。尚未结束回合的奖励不会伪装成完整回合数据。

追加训练：

```bash
artifacts/venv/bin/python -m rmvision_rl.training.train \
  --resume artifacts/training/static-front-v1/latest.zip \
  --timesteps 65536 \
  --output-dir artifacts/training/static-front-v1-resume
```

恢复预算是**追加步数**。恢复权重、优化器和累计步数，环境从新的固定场景回合开始；不恢复
原物理世界或未完成 rollout。恢复沿用检查点内配置，不同时接受 `--config` 或 `--seed`，
只允许覆盖预算、设备和输出目录。观测规格或环境/视觉配置指纹变化时拒绝恢复。
Ctrl+C、SIGTERM 或运行故障会退出并清理进程，保留已有完整检查点，不保存半次更新。

查看训练曲线：

```bash
artifacts/venv/bin/tensorboard --logdir artifacts/training --port 6006
```

加载模型运行一次带掩码推理；这仍是 Gym 采样，不是独立窗口结算评估：

```bash
artifacts/venv/bin/python - <<'PYCODE'
from pathlib import Path
from sb3_contrib.common.maskable.utils import get_action_masks
from rmvision_rl.training.environment import make_environment
from rmvision_rl.training.models import read_checkpoint_metadata, load_model

checkpoint = Path('artifacts/training/static-front-v1/final.zip')
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

训练 CLI 正常结束后，先保存 `final.zip`、关闭训练进程，再自动执行一个独立评估回合并打开
三维回放。新训练和恢复训练均如此；无人值守或只检查训练链路时添加 `--no-view`。
Python 函数 `run_training()` 只负责训练；自动查看由 CLI 衔接。

首次使用时构建回放程序：

```bash
cd /home/dgx_spark/RM_VISION_WORKSPACE/rm_simulator_2027
cargo build --offline --release --no-default-features --features training --example training_preview -j 6
```

在 `rm_vision_rl` 根目录手动查看检查点，或重新播放已有数据：

```bash
artifacts/venv/bin/python -m rmvision_rl.training.view \
  --checkpoint artifacts/training/static-front-v1/final.zip
artifacts/venv/bin/python -m rmvision_rl.training.view \
  --replay artifacts/training/static-front-v1/evaluation/eval-XXXX/replay.json
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

训练和查看命令均可用 `--viewer-binary PATH` 指定回放程序。无图形会话或程序缺失时保留数据并
打印打开命令；推理或显示失败单独报错，已完成的训练和模型不受影响。中断/失败的训练不自动
查看；评估故障不发布完整回放文件，进程日志留在该次评估目录供定位问题。

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
from rmvision_rl.transport.client import TrainingClient

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
from rmvision_rl.transport.processes import training_worker
from rmvision_rl.transport.vision_bridge import vision_worker
from rmvision_rl.environment.evaluation import EvaluationConfig, EvaluationSession

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
from rmvision_rl.policy.observations import TensorPolicy

def actor(value):
    # value 包含 features、valid、action_mask 和 version。
    return next((i for i in (1, 3, 5, 7, 0) if value['action_mask'][i]), 0)

session = EvaluationSession(
    client, bridge, EvaluationConfig(window_ms=3000),
    policy=TensorPolicy(actor), policy_mode='fire_only',
)
```

动作和数据语义见[接口说明](interfaces.md)。

## 配置工具现状

`tools/config/prepare_vision_profile.py` 仍导入已删除的 `tools.validation.validate.require`，当前不能运行，包括 `--help`。此问题不影响上面的默认配置示例；修复前不要将该工具作为运行前置条件。

`config/policy/mpc-recovery.json` 定义将单次求解上限从 50 调整为 200 的候选配置，不会自动生效。运行读取的是传给 `vision_worker` 的目录下 `src/config/modules/`。使用配置副本时，用副本根目录替换示例中的 `vision`，并记录实际配置。

## 常见问题

| 现象 | 处理方向 |
| --- | --- |
| 训练期间没有画面 | 训练过程无渲染；正常结束后默认自动评估并回放，也可用 `training.view --checkpoint` 手动查看 |
| 找不到 Bevy/Rust 动态库 | 使用 `cargo run` 或 `training_worker`，保留匹配工具链和 `deps/` |
| `No module named rmvision_rl` | 从本仓库根目录执行 |
| 缺少桥接二进制 | 完成 CMake 构建，核对输出路径 |
| 预热超时或长期无有效命令 | 检查场景、测量、跟踪及 MPC 配置；保留超时，不静默重复抽样 |
| 找不到旧验收命令 | 历史工具已删除，使用本文运行库示例 |
