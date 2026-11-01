# 接口与配置

本页按通信、环境、策略、训练和评估顺序说明接口。当前课程使用 v5 单步 GRU；通信 wire v1、Actor schema v5、课程 v2/v3 是不同的版本号，不能互相替代。历史短历史数组及 MLP 网络契约见[历史接口](legacy_interfaces.md)。

## 仿真客户端

`TrainingClient` 连接本机 Unix socket。协议 v1 为 4 字节大端长度加 UTF-8 JSON，单消息最大 16 MiB，单连接串行请求。客户端维护 request_id、round_id 和 step_id。

| 方法 | 作用 |
| --- | --- |
| `reset(seed, scenario=None)` | 新回合；成功后时间和步数归零 |
| `advance(yaw_rad=0, pitch_rad=0, distance_m=4, valid=True, fire=False)` | 提交云台命令，推进 10 ms |
| `inspect()` | 读取状态，不推进、不消费新事件 |
| `end_window(max_settle_steps=1000)` | 关窗并开始或直接完成结算；本次不推进 |
| `settle()` | 结算推进 10 ms，不提交新控制命令 |
| `close()` | 关闭实例 |
| `connect()` / `retry_pending()` | 连接并重取未确认请求，不生成新动作编号 |

响应含 `ok`、关联编号、`sim_time_ns` 和 `data`。data 中 `feedback`、`self_referee`、`self_weapon`、`visual_frames` 用于控制；`evaluation`、`events`、`reward_damage` 用于独立统计。Reset 的 `previous_round` 是旧回合截断摘要，不是新回合奖励。

`self_weapon` v1 含 `version`、`valid`、`sample_ns` 和 `fire_blocks`，每物理步更新，只允许自身机械间隔（bit 5）、供弹忙（bit 6）和枪口无效（bit 7）。旧观测路径的 Python 桥接用该通道替换裁判快照的对应位，再交给 C++；生命、热锁、弹量及裁判样本时间保持原口径。武器反馈无效、来自未来或超过 10 ms 时禁止发射，格式/版本错误直接报错。不含该字段的旧仿真器沿用旧的保守处理。该组合不读取 `evaluation`，不改变网络观测维度或策略动作接口。

合成视觉的结构遮挡仍检查装甲中心和四角，但排除弹丸碰撞体。短时遮住一个采样点的小弹丸不应被等价为整块装甲消失；墙体、地形和机器人结构仍参与遮挡，弹丸物理碰撞、CCD 与伤害计算保持不变。这是合成检测近似，不是像素级遮挡模型。

角度遵循视觉 ROS 约定，单位 rad，正 pitch 向上；距离单位 m。`fire` 是脉冲电平，持续高电平不等于连续单发请求。接收命令不等于实际出膛，以 `shot_fired` 事件为准。

通信结果不明确时先重取同一未确认请求。世界执行中途出错后需要恢复或重启并 Reset，不能跳过错误继续使用部分更新状态。

## 场景和测量

场景定义位于仿真项目 `src/training/scenario.rs`。下表为旧静止/旋转任务的顶层字段；当前多车型任务通过 `duel.controlled/target` 定义双方，见[多车型环境](multirobot_environment.md)。

| 字段 | 含义 |
| --- | --- |
| `target_distance_m` | 初始水平距离，2–8 m |
| `target_bearing_rad` | 出生方位，Bevy 世界 +Y 轴绕 -Z 前方的方位角 |
| `target_yaw_rad` | 目标车体初始相位 |
| `gimbal_yaw_rad` / `gimbal_pitch_rad` | 初始云台角，俯仰受机械限位约束 |
| `motion` | 下表中的目标运动 |
| `target_hp` | 可选血量覆盖，1–1,000,000 |
| `unlimited_heat` | 布尔值，默认 false；true 禁用双方热量累积及热量锁定，机械约束仍生效 |
| `measurements` | 合成测量配置；底层 Reset 默认关闭，预热/评估会话自动启用 |

| motion 示例 | 行为 |
| --- | --- |
| `{"kind":"static"}` | 静止 |
| `{"kind":"translation","amplitude_m":0.5,"speed_m_s":2.0}` | 沿初始车体局部 +X 的正弦平移；可用 translation_heading_rad 指定世界路径轴；幅度 (0,2] m，峰值速度 [0,4] m/s |
| `{"kind":"rotation","angular_speed_rad_s":3.0}` | 匀速旋转；角速度 [-7,7] rad/s |

未指定出生/角度字段按 seed 采样。位置和路径经过场地合法性检查；显式非法场景可能被拒绝，不保证每个距离/方位组合都可放置。旧顶层场景的己方底盘固定；duel 场景支持双方脚本运动。

| measurements 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `noise_std_px` | 0.25 | 每个角点坐标独立高斯噪声标准差，范围 [0,10] px |
| `latency_ms` | 20 | 固定捕获到交付延迟，范围 [0,500] ms，控制步交付还可能增加等待 |
| `dropout_probability` | 0 | 整帧空检测概率，范围 [0,1] |
| `blackouts_ms` | `[]` | 回合时间内空检测区间 `[start,end)`，最多 16 组 |

空检测仍交付时间戳，不等于相机停帧或网络丢包。当前噪声、固定延迟和几何遮挡近似不等于已标定的完整渲染检测误差。

多车型任务不接受 `target_hp` 覆盖或无限热量。`visibility_admission="full_path_clear"` 为课程 v3 修订 3 的全路径可见性准入，字段省略保留旧行为；适用运动和证书范围见[课程 v3](curriculum_training_v3.md#修订-3原地图全路径无遮挡)。

## 当前 Gym 接口

`RMMultiRobotFire-v0` 对应 `MultiRobotFireEnv`，复用 `FireEnv` 的控制周期。课程由训练层生成阶段场景，通用环境本身不负责晋级。

| 方法 | 返回与约束 |
| --- | --- |
| `reset(seed=None, options=None)` | `(obs, info)`；完成出生准入、禁射预热并准备首个决策 |
| `step(action)` | `(obs, reward, terminated, truncated, info)`；推进 10 ms |
| `action_masks()` | 当前合法动作的布尔副本 |
| `close()` | 取消未提交周期并回收进程，可重复调用 |

课程使用 `joint/Discrete(9)`。手动 Gym 输入若被掩码禁止，射击降级为同槽位 TRACK，槽位无效则 WAIT；PPO 采样器遇到动作降级会报错。底层九动作回调严格拒绝非法动作，二者不可混淆。`obs` 含 Actor 白名单和仅训练 Critic 使用的 `privileged[3136]`，形状、归一化和重置事件见[GRU 契约](persistent_gru.md#单步与状态接口)。`info` 含场景真值、奖励分项及执行诊断，禁止整体送入 Actor。

目标击毁或己方死亡为自然终止；额度耗尽后继续处理待发请求及在途弹丸；30 s 上限为时间截断。末观测来自原回合，截断不追加评估尾部奖励。静止/旋转靶专有的默认值与出生种子规则见[历史 Gym 接口](legacy_interfaces.md)。

## 策略动作

| 编码 | 动作 |
| --- | --- |
| 0 | WAIT：不创建新请求；保持原有效槽位，无有效槽位则停止目标跟踪 |
| 1 / 2 | 跟踪槽位 0 / 跟踪槽位 0 并请求单发 |
| 3 / 4 | 跟踪槽位 1 / 跟踪槽位 1 并请求单发 |
| 5 / 6 | 跟踪槽位 2 / 跟踪槽位 2 并请求单发 |
| 7 / 8 | 跟踪槽位 3 / 跟踪槽位 3 并请求单发 |

- `rule`：不传 policy，使用传统规则。
- `nine`：回调选择九动作之一。
- `fire_only`：规则选板，回调仅在该槽位 TRACK/FIRE 间选择；无规则槽位只允许 WAIT。

回调返回 Python `int`，不接受 bool、浮点或被掩码禁止的动作。掩码 True 表示允许申请，不保证随后 MPC、发布和实际发射成功。WAIT/跟踪不撤销已接纳的请求，活动脉冲可能仍保持高电平。

C++ 在同周期发送观测并等待动作，然后完成控制计算。前置输入无效时可能不调用策略；完整 LOST 时桥接可执行禁射搜索。无策略回调不等于本步未推进，标准训练封装需要处理这一情况。

## 随机射击决策时钟

`DecisionPolicy`（含 `StaticFirePolicy`、`JointFirePolicy`）和 Gym 可启用独立决策时钟。预热完成后第 0 个控制步为首次机会；每个到期机会执行完一个 10 ms 步后，从闭区间内的离散步数等概率抽取下一间隔。抽样不接收策略动作，TRACK、物理禁射、无回调/LOST 都会消耗该机会，不排队补发。非到期步禁止新的射击请求，联合模式仍允许任意合法槽位的 TRACK/保持；已有脉冲和在途命令仍由原火控处理。计时只随成功完成的物理步前进，读取观测、暂停、目标代次/历史复位均不改变时钟。

旧短历史模式启用时，Gym 二动作观测额外含 `decision_clock: float32[5]`，依次为是否到期、距下次机会的秒数、最小间隔秒数、最大间隔秒数、每回合固定周期秒数（逐次采样模式为 0）；不含真实目标状态或命中信息。基础 8×90 特征保持不变。检查点 `action_version=2` 并记录时钟特征契约，配置指纹包括区间、种子和采样方式，不能在旧无时钟权重上静默开启。

独立 RNG 使用配置 seed 与回合流编号派生：同一环境的成功回合依次使用流 0、1、2……，即使固定场景 Reset 也不会重复同一时钟流。几何拒绝重试和预热不消耗时钟流。旧非课程环境/恢复训练从流 0 开始；课程 GRU 恢复保留 worker 回合计数，避免重用回合流；独立模型评估也从流 0 开始，使同配置候选共享相同时间表。`DecisionPolicy` 的调用方应在预热结束调用 `clock.start_episode()`，每个实际完成的控制步调用 `clock.complete_step()`；标准 Gym 与 `EvaluationSession` 已负责此流程，调用者不应重复推进。

`info.decision_clock_before` 对应刚提交的动作，`info.decision_clock` 对应返回观测；`physical_fire_legal_before` 区分原火控合法性，`clock_masked` 标识被时钟屏蔽的手动请求。环境日志 `env-*/decision-clock.jsonl` 逐次记录机会、动作、抽样间隔及接受结果。独立评估每步在 vision 结果内记录时钟前后状态，回放记录保存配置和时钟状态。原生 `rule` 无策略基线仍不加时钟；对照随机时钟下的始终发射基线应使用带同配置时钟的 `StaticFirePolicy`，选择当前合法的 FIRE。

该功能随机化决策间隔，不随机化机械冷却、命令/视觉延迟；尚不构成跨车辆时序的完整域随机化。Gym 步长、gamma/GAE、奖励口径和回合长度均保持原样。

当前 v5 始终提供单步 `decision_clock[5]`，由资源分支及开火头读取；完整输入见[持续记忆 GRU](persistent_gru.md#单步与状态接口)。v3 的传统对照使用独立 `rmvision-rule-bridge` 与共同 DecisionClock，区别于上文无外部时钟的通用 rule 会话。

## PPO 配置与检查点

训练入口为 `python -m src.training.train`，默认读取
`config/training/rmul_2027_infantry_curriculum_v2.json`；哨兵使用
`config/training/rmul_2027_sentry_curriculum_v2.json`。另有各角色的 v3 预设；v3 需要独立校准与发布流程。

当前策略为 `recurrent_maskable_ppo/asymmetric_gru`、v5 单步观测和九动作；
网络结构由 `recurrent_actor.py` 与 `recurrent_policy.py` 定义。默认单 GPU、128 环境、
每环境至少 512 步后完成当前回合，统一执行循环 PPO 更新。完整回合按 128 步序列训练，
最多 32 步预热，小批次 64 个片段。`ppo` 配置不再使用 MLP 的 `net_arch`。

`environment` 配置回合长度、场景种子、角色、观测和动作模式；
`decision_clock` 定义独立的 50–100 ms 射击机会，控制周期仍为 10 ms。
`curriculum.version=2/3` 选择相应十二阶段课程；`thermal_reward` 配置热锁惩罚；
`runtime={}` 启用长期运行默认值。详见 [课程 v2](curriculum_training_v2.md)。

`--config` 与 `--resume` 互斥。新训练可覆盖预算、设备和环境数；课程恢复保留原累计预算、
环境数、课程和随机流，从新回合零隐藏状态开始，仅完成剩余预算。预检和评估不计入训练步数。

GRU 检查点 ZIP 包含 `rmvision.json` 与 `recurrent.pt`，保存模型、优化器、训练计数、
随机流和课程状态；`latest.zip` 通过原子替换发布。NUC 使用独立导出的 TorchScript Actor，
不加载 Critic。恢复前核对观测、场景及源码/二进制指纹。

旧 MLP 检查点仍按 SB3 格式与原契约读取，用于历史实验；仓库不再提供旧训练预设。
v4 MLP 接口见 [非对称观测](asymmetric_observations.md)，当前接口见 [持续记忆 GRU](persistent_gru.md)。

## 奖励与评估分数

仿真 `reward_damage` 是本方本步实际伤害增量。未配置奖励变换的旧 Gym 将其作为逐步 reward；当前课程再乘 reward_scale 并扣除选板、热锁成本，公式见[课程 v2](curriculum_training_v2.md#奖励及物理统计)。

`WindowScore` 是独立评估计分器：只统计窗口 `[start_ns,end_ns)` 内实际出膛弹丸最终造成的本方实际伤害。窗口内出膛、窗口后命中计入；窗口后或恰好截止出膛不计入。重复事件不重复加分；结算不完整或事件异常时，不提供完整 `official_damage`。

评估关窗停止新请求，已有供弹和在途弹丸继续按物理规则结算。Reset 截断旧世界，不能替代自然评估结算。训练采用时间截断价值自举，不把评估总分复制成每一步奖励。

## 评估会话

`EvaluationSession(client, bridge, config=None, warmup_config=None, policy=None, policy_mode='rule')` 提供 `reset(seed, scenario)`、`advance()`、`info()`。它不是 Gymnasium `step(action)`，动作通过构造时传入的回调返回。

```text
idle → warming → evaluating → settling → complete
           └─ warmup_timed_out       └─ settlement_timed_out
运行异常 → fault
```

预热默认要求 5 个连续新鲜 tracking 图像帧，最大图像年龄 100 ms，期限 15 s。确认按新图像而不是控制调用次数计数。预热禁射，就绪后保持物理和估计历史连续，单独开始评估计时。

`EvaluationConfig` 默认 `window_ms=30000`、`max_settle_steps=1000`、`target_hp=100000`。这些仅控制评估会话，Gym 采样使用前述独立回合边界。

评估中己方死亡或目标被击毁时提前关窗并自然结算。`info.end_reason` 区分 `time_limit`、
`controlled_dead` 和 `target_destroyed`；真正终止优先于同时到达的时间上限。提前结束仍使用
实际结束时间作为出膛归属的右开边界。

历史 MLP 模型回放入口 `tools.training.view.evaluate_checkpoint(checkpoint, device=None, output_dir=None)` 返回
完整回放文件路径。它按保存模式复用 `DecisionPolicy` 与 Gym 的动作编码，并使用相同 Gym RNG 派生
出生种子；通过已有模型加载器检查兼容性。`tools.training.view.open_replay(path, viewer_binary=None)`
只启动显示进程，缺少图形环境或程序时返回 `False` 并保留文件。

`replay.json` 当前 `version=1`：头部包含 `model`、`fingerprints`、`render` 路径、`scene_seed`、
`reset_attempts` 和 `scenario`；`frames` 保存每个 10 ms 周期的 `time_ns`、`phase`、`metrics`、
真实姿态与事件 `data`，`summary` 保存原始伤害与最终 `WindowScore`。不推进时间的关窗事件
合并进同时间戳的末帧，不重复记录物理周期。真值仅供计分与显示，不进入模型观测。
`settlement_timed_out` 的回放可查看已有过程，但 `official_damage=null`，不能解释为完整得分。
