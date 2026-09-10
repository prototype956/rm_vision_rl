# 构建与运行

以下使用当前 NUC 的目录布局。迁移到其他机器时修改路径并准备对应架构的依赖。示例用于运行环境和脚本策略，不启动强化学习训练。

## 1. 依赖与构建

- Python 3.10 或更高版本；当前运行库使用标准库，不要求 PyTorch 或 Gymnasium。
- 仿真项目可用的 Rust/Cargo 工具链、Cargo 依赖及 `assets/`。
- 支持 C++20 的编译器、CMake、Eigen3、OpenCV（core/calib3d/imgproc）、Ceres、yaml-cpp、fmt、spdlog 和线程库。

构建仿真训练进程：

```bash
cd /home/nuc/Workspace/bevy_robomaster_simulator
cargo build --offline --no-default-features --features training --bin daedalus_training
```

`--offline` 使用已有 Cargo 缓存；未缓存的依赖需要联网去掉此选项构建。当前 Bevy 仍启用动态链接，无窗口运行不意味着已经裁掉全部图形相关构建依赖。

构建桥接：

```bash
cd /home/nuc/Workspace/rm_vision_rl
cmake -S native/vision_bridge -B artifacts/build/vision-bridge \
  -DVISION_ROOT=/home/nuc/Workspace/rm_vision_2027 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build/vision-bridge --parallel 2
```

输出分别是仿真项目的 `target/debug/daedalus_training` 和本项目的 `artifacts/build/vision-bridge/rmvision-rl-bridge`。

## 2. 手动运行底层物理环境

终端 A：

```bash
cd /home/nuc/Workspace/bevy_robomaster_simulator
RL_SOCKET_DIR=$(mktemp -d /tmp/rm-rl-manual-XXXXXXXX)
echo "$RL_SOCKET_DIR/training.sock"
cargo run --offline --no-default-features --features training \
  --bin daedalus_training -- --socket "$RL_SOCKET_DIR/training.sock"
```

程序无窗口，等待客户端请求。`cargo run` 设置动态库搜索路径；还可传 `--config PATH`、`--assets PATH`、`--physics-step-us 1000`。

终端 B：将示例 socket 改为终端 A 实际打印的路径。

```bash
cd /home/nuc/Workspace/rm_vision_rl
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

## 3. 运行完整规则闭环

以下示例自动启动和关闭两个进程，不要提前手动启动环境。正式窗口为 3 秒，另有最多 15 秒仿真预热和最多 10 秒尾部结算；实际耗时由机器速度决定。

```bash
cd /home/nuc/Workspace/rm_vision_rl
python3 - <<'PYCODE'
import json
import tempfile
from pathlib import Path
from rmvision_rl.transport.processes import training_worker
from rmvision_rl.transport.vision_bridge import vision_worker
from rmvision_rl.environment.evaluation import EvaluationConfig, EvaluationSession

root = Path.cwd()
sim = root.parent / 'bevy_robomaster_simulator'
vision = root.parent / 'rm_vision_2027'
sim_binary = sim / 'target/debug/daedalus_training'
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

## 4. 接入 Python 策略

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
| 启动后没有画面 | 训练入口无窗口；客户端请求才推进，图像预览需另运行 Talos 渲染链路 |
| 找不到 Bevy/Rust 动态库 | 使用 `cargo run` 或 `training_worker`，保留匹配工具链和 `deps/` |
| `No module named rmvision_rl` | 从本仓库根目录执行 |
| 缺少桥接二进制 | 完成 CMake 构建，核对输出路径 |
| 预热超时或长期无有效命令 | 检查场景、测量、跟踪及 MPC 配置；保留超时，不静默重复抽样 |
| 找不到旧验收命令 | 历史工具已删除，使用本文运行库示例 |
