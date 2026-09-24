# 接口与配置

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

`self_weapon` v1 含 `version`、`valid`、`sample_ns` 和 `fire_blocks`，每物理步更新，只允许自身机械间隔（bit 5）、供弹忙（bit 6）和枪口无效（bit 7）。Python 桥接用该通道替换裁判快照的对应位，再交给 C++；生命、热锁、弹量及裁判样本时间保持原口径。武器反馈无效、来自未来或超过 10 ms 时禁止发射，格式/版本错误直接报错。不含该字段的旧仿真器沿用旧的保守处理。该组合不读取 `evaluation`，不改变网络观测维度或策略动作接口。

合成视觉的结构遮挡仍检查装甲中心和四角，但排除弹丸碰撞体。短时遮住一个采样点的小弹丸不应被等价为整块装甲消失；墙体、地形和机器人结构仍参与遮挡，弹丸物理碰撞、CCD 与伤害计算保持不变。这是合成检测近似，不是像素级遮挡模型。

角度遵循视觉 ROS 约定，单位 rad，正 pitch 向上；距离单位 m。`fire` 是脉冲电平，持续高电平不等于连续单发请求。接收命令不等于实际出膛，以 `shot_fired` 事件为准。

通信结果不明确时先重取同一未确认请求。世界执行中途出错后需要恢复或重启并 Reset，不能跳过错误继续使用部分更新状态。

## 场景和测量

场景定义位于仿真项目 `src/training/scenario.rs`：

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
| `{"kind":"translation","amplitude_m":0.5,"speed_m_s":2.0}` | 世界 X 方向正弦平移；幅度 (0,2] m，峰值速度 [0,4] m/s |
| `{"kind":"rotation","angular_speed_rad_s":3.0}` | 匀速旋转；角速度 [-7,7] rad/s |

未指定出生/角度字段按 seed 采样。位置和路径经过场地合法性检查；显式非法场景可能被拒绝，不保证每个距离/方位组合都可放置。己方底盘当前固定。

| measurements 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `noise_std_px` | 0.25 | 每个角点坐标独立高斯噪声标准差，范围 [0,10] px |
| `latency_ms` | 20 | 固定捕获到交付延迟，范围 [0,500] ms，控制步交付还可能增加等待 |
| `dropout_probability` | 0 | 整帧空检测概率，范围 [0,1] |
| `blackouts_ms` | `[]` | 回合时间内空检测区间 `[start,end)`，最多 16 组 |

空检测仍交付时间戳，不等于相机停帧或网络丢包。当前噪声、固定延迟和几何遮挡近似不等于已标定的完整渲染检测误差。

## 评估会话

`EvaluationSession(client, bridge, config=None, warmup_config=None, policy=None, policy_mode='rule')` 提供 `reset(seed, scenario)`、`advance()`、`info()`。它不是 Gymnasium `step(action)`，动作通过构造时传入的回调返回。

```text
idle → warming → evaluating → settling → complete
           └─ warmup_timed_out       └─ settlement_timed_out
运行异常 → fault
```

预热默认要求 5 个连续新鲜 tracking 图像帧，最大图像年龄 100 ms，期限 15 s。确认按新图像而不是控制调用次数计数。预热禁射，就绪后保持物理和估计历史连续，单独开始评估计时。

`EvaluationConfig` 默认 `window_ms=30000`、`max_settle_steps=1000`、`target_hp=100000`。这些仅控制评估会话，Gym 采样使用下节的独立回合边界。

评估中己方死亡或目标被击毁时提前关窗并自然结算。`info.end_reason` 区分 `time_limit`、
`controlled_dead` 和 `target_destroyed`；真正终止优先于同时到达的时间上限。提前结束仍使用
实际结束时间作为出膛归属的右开边界。

模型回放入口 `tools.training.view.evaluate_checkpoint(checkpoint, device=None, output_dir=None)` 返回
完整回放文件路径。它按保存模式复用 `DecisionPolicy` 与 Gym 的动作编码，并使用相同 Gym RNG 派生
出生种子；通过已有模型加载器检查兼容性。`tools.training.view.open_replay(path, viewer_binary=None)`
只启动显示进程，缺少图形环境或程序时返回 `False` 并保留文件。

`replay.json` 当前 `version=1`：头部包含 `model`、`fingerprints`、`render` 路径、`scene_seed`、
`reset_attempts` 和 `scenario`；`frames` 保存每个 10 ms 周期的 `time_ns`、`phase`、`metrics`、
真实姿态与事件 `data`，`summary` 保存原始伤害与最终 `WindowScore`。不推进时间的关窗事件
合并进同时间戳的末帧，不重复记录物理周期。真值仅供计分与显示，不进入模型观测。
`settlement_timed_out` 的回放可查看已有过程，但 `official_damage=null`，不能解释为完整得分。

## 静止靶 Gymnasium

安装依赖并 `import src` 后，用 `gymnasium.make('RMStaticFire-v0')` 创建环境，
也可直接导入 `src.environment.static_fire.StaticFireEnv`。
构造参数为 `simulator_root`、`vision_root`、`simulator_binary`、`bridge_binary`、
`simulator_config`、`log_dir`、`episode_steps=3000`、`warmup_config=None`、`render_mode=None`、
`decision_clock=None`、`action_mode="fire_only"`。
路径默认使用当前仓库及相邻的 `rm_simulator_2027`、`rm_vision_2027`，仿真二进制默认 Release。
无渲染模式是唯一支持的渲染设置；用 `episode_steps` 配置时限，勿额外叠加 `TimeLimit`。

| 接口 | 行为 |
| --- | --- |
| `reset(seed=None, options=None)` | 返回 `(obs, info)`；启动或复用进程，禁射预热并准备首个决策 |
| `step(action)` | 返回 `(obs, reward, terminated, truncated, info)`；恰好推进 10 ms |
| `action_masks()` | 返回当前 `bool[action_count]` 副本；需处于正式采样状态 |
| `close()` | 取消未提交周期、释放进程和 socket；可重复调用 |

默认省略 `action_mode` 时动作空间 `Discrete(2)`：0 跟踪规则槽位且不新增射击请求；1 在该槽位请求单发。
无规则槽位时动作 0 映射到底层 WAIT。动作 0 不撤销已接纳脉冲。
被掩码禁止的动作 1 执行动作 0，`info.action_masked=True`，无额外奖励或惩罚。
Python/NumPy 整数均支持；bool、浮点和越界动作在物理推进前报错。
九动作回调接口保持原来的严格检查，不会自动替换非法动作。

fire_only 的 `obs` 为 Dict：`features=float32[8,90]`、`valid=int8[8]`、`action_mask=int8[2]`。
`features` 范围 [-1,1]，后两项为 0/1；每次返回独立数组。没有策略回调的周期填零行且
valid=false，掩码为 `[1,0]`。Reset 和目标代次变化清空历史，代次仅作为内部复位元数据。
语义年龄 null 编码为对应归一化上限 1；`since_request_s=null` 表示从未请求，同样编码为 1。

`options` 仅支持 `{'scenario': {...}}`，字段沿用仿真场景接口，但 motion 必须为 static，
measurements 必须启用。缺省填入距离 4 m、HP 100000、噪声 0.25 px、延迟 20 ms、空检测概率 0。
其他出生参数仍按仿真种子采样。Gym seed 初始化其随机流，并从中派生仿真 seed；二者不等同。
几何无效出生最多尝试 32 个派生 seed，使用新请求编号；尝试和拒绝原因在 `info.reset_attempts`。
预热超时、配置错误及通信故障不会重采样掩盖，而是报错并清理进程；下次 Reset 重新启动。

正式阶段丢失目标继续采样，不重新预热。己方死亡或目标击毁返回 terminated；否则达到
episode_steps 返回 truncated，二者相遇以真正终止为准。末观测来自旧回合真实末时刻，
没有提交下一动作或推进下一物理步。结束后 Step 报错，必须 Reset。

奖励为 `float(reward_damage)`，仅计算本次推进实际发生的伤害，不包含预热及 Reset 摘要。
时间截断不执行尾部结算，Reset 会丢弃旧世界在途弹丸；训练器必须保留末观测并进行
时间截断价值自举。训练累计奖励不是窗口归属评估分数，不能用末尾结算奖励和自举重复补偿。

`info` 提供 `damage`、`episode_damage`、`actual_shots`、`episode_steps`、`episode_time_s`、
`physical_time_ns`、`end_reason`、`action`、`action_masked`、`shot_requested`、
`shot_accepted`、`reject_reason`、`reset_attempts` 和 `warmup`。请求状态对应刚执行的动作，
返回 obs/mask 对应下一决策周期。info 含实验及评估数据，禁止整体送入网络。

## 随机旋转靶 Gym

`RMRotationFire-v0` / `RotationFireEnv` 与静止靶共用 `FireEnv` 的进程、预热、观测、
动作和奖励接口。额外构造参数 `angular_speed_range_rad_s=(1, 7)` 指定角速度大小范围。
`options.scenario` 不允许包含 motion；默认不补齐位置、距离、方位或朝向，交给仿真器采样。
HP 和测量默认值与静止靶相同，旋转不会增加策略观测字段。

每回合从 Gym RNG 取一个 `episode_seed`，再通过固定域 0/1 派生角速度和出生重试子流。
角速度大小均匀采样、正反方向等概率，正号按 Bevy 世界 +Y 右手旋转。出生失败只重抽仿真种子，
不改变本回合角速度或下一回合采样结果。仿真器 revision 5 在非零旋转出生时以固定初始
相机每 5° 检查整圈可见性；地形遮挡导致某相位无完整可见装甲时按非法出生重试，
通过后仍执行原有预热确认。预热期间也持续旋转，超时直接报错。

`prepare_scene(seed=None, options=None)` 仅采样并返回 `(scenario, spawn_rng, sample)`，
不启动进程；它会消耗一次场景采样，供 Gym reset 和独立评估共用，不应在 reset 前额外调用。
显式 Gym seed 重启序列；`reset()` 继续。`info.scene` 包含从 0 开始的 `episode_index`、
`episode_seed`、`angular_speed_rad_s`、成功的 `spawn_seed` 和仿真器出生报告 `scenario`。
每次成功出生时写入环境目录 `scenes.jsonl`，包含全部 `reset_attempts`；该记录不代表预热成功。
这些真值仅用于复现与诊断，禁止送入策略网络。

检查点保存未采样的场景模板与速度范围；单模型评估总是取保存种子的首个场景，
同配置的候选检查点使用相同参考场景比较，不代表跨场景泛化成绩。
回放头部增加 `scene_sample`（场景序号、回合种子、角速度、成功出生种子），
`scenario` 仍记录实际出生报告；旧回放读取不依赖新字段。

## PPO 配置与检查点

训练入口为 `python -m src.training.train`，默认读取
`config/training/rotation_joint_ppo.json`。配置包含版本、算法/策略类型、PPO seed、device、
torch_threads、total_timesteps、checkpoint_updates、environment 和 ppo。
仅支持 `maskable_ppo/mlp`；`ppo.net_arch` 分别配置 Actor 的 pi 和 Critic 的 vf 层宽，
激活函数固定 Tanh；其余初始参数见配置。学习率和 clip 为常数，不支持分段调度。

environment 包含 episode_steps、scene_seed、scenario，并可配置 Gym 的五个路径参数
simulator_root、vision_root、simulator_binary、bridge_binary、simulator_config。
可选 `environment.decision_clock` 包含 `min_interval_ms`、`max_interval_ms`、`resample`、`seed` 四个必需字段。
区间端点为 10 ms 的整数倍，支持 10–60000 ms 且最小值不大于最大值；`resample` 为 `decision`（每次机会后采样）或 `episode`（每回合采样一个周期），seed 为独立的 32 位非负整数。省略整个对象表示关闭，保留旧动作契约。
可选 `environment.task` 为 `static_fire`（省略时保持旧行为）或 `random_rotation_fire`。
旋转任务的 `angular_speed_range_rad_s` 默认 `[1, 7]`，要求有限数值且 `0 < min <= max <= 7`；
静止任务禁止该字段，旋转任务禁止在 `scenario` 中指定 `motion`。
场景与模型 seed 独立；包装器忽略 SB3 的环境播种请求。静止任务每次 Reset 重用固定种子；
旋转任务仅首次用 scene_seed 初始化，随后持续采样。恢复训练重启场景序列，不恢复其进度。
`--config` 和 `--resume` 互斥；恢复不接受 `--seed`，只允许覆盖 device、追加 timesteps 和新输出目录。

模型 ZIP 使用 SB3 格式保存权重、优化器和累计步数，并额外嵌入 `rmvision.json`，将配置、
观测版本/形状/类型/特征 schema 指纹、对应模式动作语义、环境/视觉配置指纹、依赖版本与完整更新
次数一起保存。模型构建和加载统一经 `training.models`；加载前检查类型和观测/配置兼容性，
加载后清空旧观测，由新回合开始采样。不能把旧 MLP 检查点直接当成未来 GRU 检查点。

`run.json` 为运行配置与状态记录；`episodes.monitor.csv` 为原始回合统计，
`logs/progress.csv` 和 TensorBoard event 为完整更新日志。`latest.zip` 是最近保存的完整
检查点副本，通过同目录临时文件原子替换；失败或中断不会覆盖为部分更新模型。

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

## 观测

语义回调主要接收 `estimate`、`feedback`、`referee`、四个 `candidates`、数据年龄、上次槽位、距请求时间和 `action_mask`。实际字段以 [policy_wire.cpp](../native/vision_bridge/policy_wire.cpp) 为准。

[observations.py](../src/policy/observations.py) 显式选择字段，按 [v1.json](../config/observations/v1.json) 顺序编码，不展开整个诊断对象。

| 输出 | 约定 |
| --- | --- |
| `version` | 1 |
| `features` | 8 × 90，从旧到新；float32 精度数值，缩放并裁剪至 [-1,1] |
| `valid` | 8 个历史行有效性标记 |
| `action_mask` | 9 个布尔值 |

`TensorPolicy` 给 actor 的数据是普通 Python 列表，不是 NumPy/PyTorch 张量。每个控制步推进历史；无语义回调时该行保持零和 valid=false。回合 Reset 清空历史；桥接按目标代次变化通知历史复位，代次不进入 actor 输入。actor 接收副本，不能修改后续周期历史。Gym 再将这些数值转换为上述 NumPy 数组及二动作掩码。

## 随机射击决策时钟

`DecisionPolicy`（含 `StaticFirePolicy`、`JointFirePolicy`）和 Gym 可启用独立决策时钟。预热完成后第 0 个控制步为首次机会；每个到期机会执行完一个 10 ms 步后，从闭区间内的离散步数等概率抽取下一间隔。抽样不接收策略动作，TRACK、物理禁射、无回调/LOST 都会消耗该机会，不排队补发。非到期步禁止新的射击请求，联合模式仍允许任意合法槽位的 TRACK/保持；已有脉冲和在途命令仍由原火控处理。计时只随成功完成的物理步前进，读取观测、暂停、目标代次/历史复位均不改变时钟。

启用时，Gym 二动作观测额外含 `decision_clock: float32[5]`，依次为是否到期、距下次机会的秒数、最小间隔秒数、最大间隔秒数、每回合固定周期秒数（逐次采样模式为 0）；不含真实目标状态或命中信息。基础 8×90 特征保持不变。检查点 `action_version=2` 并记录时钟特征契约，配置指纹包括区间、种子和采样方式，不能在旧无时钟权重上静默开启。

独立 RNG 使用配置 seed 与回合流编号派生：同一环境的成功回合依次使用流 0、1、2……，即使固定场景 Reset 也不会重复同一时钟流。几何拒绝重试和预热不消耗时钟流。新环境/恢复训练从流 0 开始；独立模型评估也从流 0 开始，使同配置候选共享相同时间表。`DecisionPolicy` 的调用方应在预热结束调用 `clock.start_episode()`，每个实际完成的控制步调用 `clock.complete_step()`；标准 Gym 与 `EvaluationSession` 已负责此流程，调用者不应重复推进。

`info.decision_clock_before` 对应刚提交的动作，`info.decision_clock` 对应返回观测；`physical_fire_legal_before` 区分原火控合法性，`clock_masked` 标识被时钟屏蔽的手动请求。环境日志 `env-*/decision-clock.jsonl` 逐次记录机会、动作、抽样间隔及接受结果。独立评估每步在 vision 结果内记录时钟前后状态，回放记录保存配置和时钟状态。原生 `rule` 无策略基线仍不加时钟；对照随机时钟下的始终发射基线应使用带同配置时钟的 `StaticFirePolicy`，选择当前合法的 FIRE。

该功能随机化决策间隔，不随机化机械冷却、命令/视觉延迟；尚不构成跨车辆时序的完整域随机化。Gym 步长、gamma/GAE、奖励口径和回合长度均保持原样。

## 奖励与评估分数

仿真 `reward_damage` 是本方本步实际伤害增量。Gym 将其作为逐步 reward，并按前述规则返回 terminated/truncated。

`WindowScore` 是独立评估计分器：只统计窗口 `[start_ns,end_ns)` 内实际出膛弹丸最终造成的本方实际伤害。窗口内出膛、窗口后命中计入；窗口后或恰好截止出膛不计入。重复事件不重复加分；结算不完整或事件异常时，不提供完整 `official_damage`。

评估关窗停止新请求，已有供弹和在途弹丸继续按物理规则结算。Reset 截断旧世界，不能替代自然评估结算。训练采用时间截断价值自举，不把评估总分复制成每一步奖励。

## 联合选板与开火契约

训练配置可选 `environment.action_mode` 取 `fire_only` / `joint`，省略保持前者且不向旧配置
注入字段，保证旧指纹。Gym 构造函数同名参数默认 `fire_only`，注册名称不变。新默认训练配置为
`rotation_joint_ppo.json`，旧 `rotation_fire_ppo.json` 保留。场景采样器和动作模式互不依赖。

joint 的 `Discrete(9)`：0 保持当前槽位且不新增射击请求；1/3/5/7 跟踪槽位 0/1/2/3；
2/4/6/8 跟踪相同槽位并请求单发。无当前槽位时 0 等待。动作掩码为物理候选掩码与时钟约束的
交集，只有四个射击动作受时钟限制。非法射击降级为同槽位 TRACK；槽位无效则 WAIT，不选另一板。
PPO 采样器检测到降级即报错。合法动作仍可能因随后 MPC 等执行失败而被拒绝。

v2 的 `features=float32[8,107]`：前 90 列完整沿用 v1；按槽位 0→3 追加
`facing_now_sin/cos`、`facing_impact_sin/cos` 共 16 列，最后是
`selected_slot_age_s_normalized`，以 1 秒为上限。保留 `valid=int8[8]`、
`action_mask=int8[9]` 和可选 `decision_clock=float32[5]`。
当前朝向取预测基准加 prediction_age；命中朝向取候选保存的准确 prediction_horizon。
角度是水平投影中，从装甲外法线到“装甲指向炮口”的有符号角，世界 +Z 为正。无效候选新增值全零。
槽位时间在首次选中或切换后从零开始累计，无槽位为零；目标代次重置同时清除选板历史。

桥接原始 wire/observation 协议保持 v1，候选追加 `facing_now_rad`、`facing_impact_rad`，
观测追加 `selected_slot_age_s`；v1 编码器忽略新增字段。`begin_training` 新增可选
`policy_mode`，只接受 `fire_only`（缺省）/`nine`，评估继续使用原来的同名模式。
联合检查点记录 tensor observation version 2、action_version 3、9 动作语义与 v2 schema 指纹。
旧检查点使用原 v1 特征、动作版本和配置指纹，不接受跨模式加载，不提供权重转换。

`info` 与 `actions.jsonl` 记录请求 `action`、实际策略 `executed_action`、协议 `wire_action`、
`selected_slot`、`slot_switched`、`slot_switches`、`mask_reason`（decision_clock/fire_control）及
射击请求/接受结果。选板以控制器实际输出为准，首次获取和目标重建不计切板。
Monitor 与 PPO 日志也记录切板数；旧检查点缓存中缺失该字段的回合不参与切板均值。
这些计分/诊断字段不加入策略观测。

`evaluate_checkpoint(..., scene_override=None)` 默认评估保存场景序列的首个场景。
对照工具传入 `{scene_id, scenario, spawn_seed}` 指定已验证出生；不修改保存契约、不再更换种子，
并在回放保存覆盖参数和实际场景。比较要求两检查点的场景基础配置、时钟、视觉/物理配置一致，
回合均为 500 步。距离与转速取固定 18 组合，其他初态随机；sampling_seed=10000+scene_id。
HTML/JSON/CSV 单列失败，配对汇总仅统计双方完成的场景；命中率为总命中弹丸/总实际发射弹丸。
