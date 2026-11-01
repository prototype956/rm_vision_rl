# 历史 Gym、观测与 MLP 策略契约

本页说明仍保留的静止/旋转靶 API 和 v1–v3 短历史契约。课程版本、观测版本和动作版本独立编号；本页“v3 选板记忆”不是课程 v3。当前 v5 GRU 见[持续记忆 GRU](persistent_gru.md)，v4 非对称字段见[观测契约](asymmetric_observations.md)。

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

检查点保存未采样的场景模板与速度范围；单模型评估默认取保存种子派生的首个场景，
训练后分析会在多个共享派生场景上进行评估，
同配置的候选检查点使用相同的多场景评估集合比较，不代表更大范围的跨场景泛化成绩。
回放头部增加 `scene_sample`（场景序号、回合种子、角速度、成功出生种子），
`scenario` 仍记录实际出生报告；旧回放读取不依赖新字段。

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

## 联合选板与开火契约

训练配置可选 `environment.action_mode` 取 `fire_only` / `joint`，省略保持前者且不向旧配置
注入字段，保证旧指纹。Gym 构造函数同名参数默认 `fire_only`，注册名称不变。当前四份 v2/v3 课程预设均使用 joint；历史配置和旧模型仍可通过对应检查点加载。场景采样器和动作模式互不依赖。

joint 的 `Discrete(9)`：0 保持当前槽位且不新增射击请求；1/3/5/7 跟踪槽位 0/1/2/3；
2/4/6/8 跟踪相同槽位并请求单发。无当前槽位时 0 等待。动作掩码为物理候选掩码与时钟约束的
交集，只有四个射击动作受时钟限制。非法射击降级为同槽位 TRACK；槽位无效则 WAIT，不选另一板。
PPO 采样器检测到降级即报错。合法动作仍可能因随后 MPC 等执行失败而被拒绝。

历史观测 v2（不是课程版本）的 `features=float32[8,107]`：前 90 列完整沿用 v1；按槽位 0→3 追加
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

`evaluate_checkpoint(..., scene_override=None)` 默认评估保存场景序列的首个场景；训练后分析会通过
`scene_override` 复用多个共享场景并按平均归属伤害选优。
对照工具传入 `{scene_id, scenario, spawn_seed}` 指定已验证出生；不修改保存契约、不再更换种子，
并在回放保存覆盖参数和实际场景。比较要求两检查点的场景基础配置、时钟、视觉/物理配置一致，
回合均为 500 步。距离与转速取固定 18 组合，其他初态随机；sampling_seed=10000+scene_id。
HTML/JSON/CSV 单列失败，配对汇总仅统计双方完成的场景；命中率为总命中弹丸/总实际发射弹丸。


### 可选 v3 选板记忆及奖励契约

`environment.selection_memory=true` 仅适用于 joint；原 8×107 特征不变，
增加独立 `selection_memory: MultiBinary(5)`，顺序为 [-1,0,1,2,3] one-hot，
表示上一个不同的实际选中槽位。回合、目标代次、缺失策略回调、非 tracking 状态
或相关候选失效时清除。训练/Gym 与检查点评估共用 DecisionPolicy 维护此状态，
检查点观测契约 version=3；旧 v1/v2 契约不变。

`selection_reward` 与 `switch_penalty` 互斥，字段为 `switch_cost`、`return_cost`、
`return_window_s`、`facing_rate_per_s`、`facing_margin`，默认分别为
0.002、0.03、0.1 秒、0.1/秒、0.1。要求有限非负数，返回窗口大于零、margin 小于 1。
奖励仅改变 PPO 回报；评估伤害与裁判计分不变。
`switch_reward_context` 在新模式额外保存 `return_slot`、`tracking_reliable`、
`candidate_valid`、`facing_impact_rad`，均在执行动作前从视觉估计/控制状态复制。
每步日志和回合累计分别记录基础切板、返回、朝向成本以及总成本；
不使用机器人真值进行朝向评分。完整公式及启动命令见 running.md。


### factorized_mlp 策略契约

`policy_kind=factorized_mlp` 仅适用于 joint，保持 Gym Discrete(9) 和 v2/v3 输入契约。
共享策略主干使用 `ppo.net_arch.pi`（默认 128/128 Tanh）；选板头输出 HOLD 和四槽位。
条件开火头共享于四槽位，输入主干特征、该槽位当前候选特征、8 帧射击状态与当前时钟/
射击掩码，64 单元隐藏层后输出每槽位一个 Bernoulli logit。
所有 `referee.*`、`referee_age_*`、`since_request_*` 特征及 decision_clock、射击掩码
从选板主干排除；跟踪掩码、有效历史标志和 selection_memory 保留。
独立 critic 使用 `ppo.net_arch.vf` 和完整观测。

`P(0)=alpha[0]`，`P(1+2i)=alpha[i+1]*(1-beta[i])`，
`P(2+2i)=alpha[i+1]*beta[i]`。禁射令对应 beta=0，不改变 alpha。
禁止出现射击合法而同板跟踪非法的掩码；无合法候选时 HOLD 保持可用。
采样按 alpha 再按所选槽位 beta，确定性执行按 alpha argmax 再 beta>0.5，
平局选择不开火，绝不对九项联合概率取全局 argmax。
HOLD 不发起新射击，重复显式选择当前槽位可以射击且不算切板。
联合 log_prob 与熵 `H(alpha)+sum(alpha[i+1]*H(beta[i]))` 用于 PPO。

新检查点额外保存 version=1 的 policy_contract，包含输入路由、条件头宽度和确定性
解码语义。缺失/不匹配或策略类与 policy_kind 不一致时拒绝加载；旧 mlp 检查点元数据
不添加该字段，原指纹保持不变。诊断中的 distribution.probs 仍为九动作联合概率，
开火概率为四个射击动作概率之和；确定性动作必须调用 distribution.mode/predict。

### shared_score_mlp 策略契约

`policy_kind=shared_score_mlp` 仅适用于 joint，保持相同的 v2/v3 观测和九动作协议。
全局主干排除候选特征、四个槽位的 previous_slot 历史和原始槽位记忆；每个候选的历史、
候选相关 previous_slot、return_slot 标志和跟踪掩码送入同一个候选编码器。四个编码结果
经过掩码平均得到排列无关的候选摘要，再由同一个评分头输出四个选板分数；HOLD 使用独立
的全局分数。条件开火头仍与 `factorized_mlp` 相同并共享四槽位参数。

共享评分只改变策略结构，不要求四个槽位均匀选择，也不包含切换滞回。候选排列时，评分和
动作应随候选一起移动；无效候选仍由动作掩码排除。该策略必须从头训练，检查点契约会拒绝
与 `mlp` 或 `factorized_mlp` 互相加载。
