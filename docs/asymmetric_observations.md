# Actor / Critic 观测契约与分离训练

本轮接口是 `observation_mode=asymmetric_v1`、Actor schema **v4**、Critic schema **v1**，策略为 `asymmetric_mlp`（架构契约 v2）。旧版模型不能直接加载到新版环境，必须重新训练。四块预测装甲、九动作、8 × 10 ms 历史、原有选板奖励和决策时钟保持原定义。步兵、哨兵使用同一输入结构，分别使用 `config/training/rmul_2027_multirobot_{infantry,sentry}.json`。

## Actor 的输入与坐标

`features` 为 `[8,126] float32`，顺序见 `config/observations/v4.json`。此外输入 `valid[8]`、`action_mask[9]`、`selection_memory[5]` 和配置启用时的 `decision_clock[5]`。`valid` 表示该行控制周期具有自身观测；目标是否有效由 `target.valid` 单独表示。

视觉桥接在 PnP/EKF **之前**将 `world_t_gimbal.translation` 清零，只保留 IMU 可提供的方向以及标定的相机/枪口安装变换。估计位置的原点随云台参考点移动，估计速度是相对运动，不使用真实底盘位置或速度。网络再把位置、速度、姿态和位置协方差旋转到当前航向坐标系：X 朝当前水平航向、Y 向左、Z 向上。IMU 姿态去掉绝对航向，保留倾斜；三轴角速度在完成安装转换后的云台机体系表达，不能简单拿欧拉角导数代替。

所有标量归一化均采用 `x/(s+abs(x))`，下表给出 `s`。布尔值、one-hot、单位四元数和三角函数值不压缩。没有额度 200 发截断，也没有热量上限截断。

| 分组 | 特征与单位 | 压缩尺度 | 网络分配 |
|---|---|---|---|
| 目标，39 维 | 有效性、跟踪状态、类别、装甲种类；相对位置 m、相对速度 m/s、姿态 xyzw、两半径 m、高度差 m、装甲倾角 rad、旋转速度 rad/s、位置协方差的 6 个独立元素 m²、偏航方差 rad²、估计年龄 s | 位置/速度 10；半径/高度差 0.5；角速度 10；估计年龄 0.1；其余标量 1 | 切板、开火 |
| IMU，9 维 | 去航向姿态 xyzw、机体系角速度 xyz rad/s、有效性、反馈年龄 s | 角速度 10；年龄 0.1 | 切板、开火 |
| 选板，6 维 | 上次槽位 one-hot（无选择、0～3）、选板持续时间 s | 时间 1 | 切板、开火 |
| 四候选，每块 15 维 | 弹道有效、当前批次实际检测、预测存在、曾观测、最近观测年龄 s；偏航误差 sin/cos、俯仰误差 rad、距离 m、飞行/预测时间 s、当前/命中时刻朝向 sin/cos | 观测年龄/俯仰误差 1；距离 10；飞行/预测时间 0.5 | 切板、开火 |
| 开火专用，12 维 | 上周期本地请求、请求年龄是否存在/年龄 s；热量反馈有效性/年龄 s；当前热量、配置热量上限、基础冷却速率、剩余允许发弹量、冷却增益剩余时间 s、有效冷却速率、冷却增益是否生效 | 请求年龄 1；反馈年龄 0.1；热量/上限 400；基础/有效冷却 100；额度 200；增益时间 60 | 仅开火 |

未曾观测的候选：`seen_before=0`，年龄以 10 s 编码；无预测年龄以 1 s 编码。无开火请求：`request_age_known=0`，请求年龄以 10 s 编码。这些缺失标记与年龄同时使用。`detected` 只来自估计器接受的图像装甲关联，不来自仿真可见性真值；不等于 `ballistic_valid`。无效弹道只清除弹道特征，观测历史标记仍可存在。

热量等资源来自每 100 ms 采样一次的可靠自身反馈，控制周期保持最近样本，时间和有效性独立保留。本轮不注入丢包、乱序或随机延迟。`allowance_remaining=-1` 表示配置的无限额度（联盟赛训练配置使用有限额度）。热量上限和基础冷却来自本回合选定的上位机资源配置；不是从裁判瞬时许可或机构状态推断。冷却增益只改变有效冷却速率。

Actor 白名单没有防御增益、双方 HP、实际出弹/测速、摩擦轮、拨弹盘、物理弹仓余弹、底盘速度、升降/折叠状态、原始 RFID 或综合开火许可。弹道仍用视觉配置中的标定弹速。

## 掩码、历史和实际执行

切板编码器只能看到目标、IMU、候选、选板记忆和跟踪掩码。开火专用资源、请求历史、决策时钟、开火掩码不会进入切板编码器。五个切板概率（保持和四槽位）与四个条件开火概率组合为九动作概率。开火分支同时读取共享候选编码器的历史输出，因此可使用对应装甲的 8 步历史、上次选板和返回槽位记忆。

新模式不调用旧的 `control_referee` 机械状态合并；桥接收到的存活/原因位是固定适配值，只有采样热量、额度和有效性可以限制开火，再叠加估计有效性和本地时钟。`FireEnv` 不再用即时仿真额度提前屏蔽动作。仿真仍按真实规则拒绝请求。

PPO rollout 保留采样动作、对应的 `log_probs` 和采样时掩码；环境日志另外记录本地请求/控制器接收情况及仿真实际出弹、拒绝计数。请求绝不等于实际出弹。目标丢失时保持自身观测更新，并沿用短暂预测跟踪/禁射行为；目标代次变化仅清目标和选板历史，回合 reset 清全部历史。

## Critic 的独立通道

`privileged[3136] float32` 仅供 Critic，编码器与 Actor 无共享参数。Critic 拼接 Actor 的完整观测历史和当前这一份真值。真值的 `sample_ns` 必须等于本次决策时刻，否则立即报错。

专用 `privileged_state` 包含双方根位姿、线/角速度、实际车体姿态与旋转速度、枪口及每车四块装甲的中心/姿态/角点、HP/最大 HP、热量/上限/额度、基础/有效冷却、冷却/回合锁定、冷却与防御增益及剩余时间、实际 RFID 区域与隧道进入进度、升降/折叠及当前机构请求/稳定计时、待完成射击请求、云台响应及排队命令、在途弹丸的位置/速度/归属/剩余寿命。

姿态与位置字段明确标注 `bevy`（Y 向上）或 `ros`（Z 向上）。真值编码不依赖评估字典。根位置/速度尺度为 10，角速度 10，HP/热量 400，额度 200，冷却 100，增益时间 60/2，机构高度 0.5；枪口/装甲位置及角点尺度 10。弹丸速度尺度 30、寿命 5。排队命令和请求用相对剩余时间/年龄（尺度 0.1），不输入绝对时间戳。

固定容量为 256 枚在途弹丸、64 条待执行云台命令和 16 条待接纳射击请求，每条带有效位；越界时报错，不静默丢弃。对象 ID 仅供归属/排序，随机种子、未来运动程序、未来命中和回合结算不进入网络。没有新增摩擦轮或拨弹盘动力学。

终止和超时都先生成同一回合的最终观测，再取消尚未提交的控制周期。SB3 `DummyVecEnv/SubprocVecEnv` 保留完整 `terminal_observation`，超时自举使用其中的最终 Critic 状态。额度耗尽继续推进物理直至已有请求和弹丸处理完毕。

## 部署接口

`export_actor` 产生 `actor-XXXX.pt` 和 `.pt.json`，TorchScript 文件只含 Actor 参数。调用输入为上面列出的 Actor 字典（批量维在前），无需 `privileged`；输出为切板概率 `[B,5]`、条件开火概率 `[B,4]`、分层确定性动作 `[B]`。实车适配器负责准备语义观测、维护 8 步历史、同一份本地掩码和时钟。

IMU 数据接口：`valid, sample_ns, orientation_xyzw, gyro_rad_s[3]`，姿态顺序 xyzw；热量接口：`valid, sample_ns, sequence, heat, heat_limit, base_cooling_per_second, allowance_remaining, cooling_active, cooling_remaining_s, effective_cooling_per_second`。时间使用上位机统一的单调时钟。冷却适配器可从底层 RFID/裁判状态生成该接口，不把原始 RFID 输入网络。本轮只定义并验证这个接口，不移植串口链路和固件。

## NUC 采样、服务器更新

NUC 编译仿真和桥接后，使用同一份源码。服务器只安装 `requirements-training.txt`，不需要启动仿真和视觉程序。`src.training.split` 使用带 SHA256 配对的 checkpoint/NPZ 文件交换；NPZ 禁用 pickle，校验行为策略、版本、形状和有限数值。每次更新后才发布 ready 文件，传输时最后传 ready 文件。

NUC（以步兵为例；哨兵换配置和独立输出目录）：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /home/nuc/Workspace/rm_vision_rl
.venv-rmul/bin/python -m src.training.split collect \
  --config config/training/rmul_2027_multirobot_infantry.json \
  --output artifacts/my-infantry-run --updates 2
```

首轮生成 `rollout-0000.zip/.npz/.ready.json`。用 SSH/SFTP 将三者复制到服务器。服务器在相同源码的根目录运行：

```bash
python -m src.training.split update \
  --checkpoint INPUT/rollout-0000.zip --batch INPUT/rollout-0000.npz \
  --output OUTPUT/update-0000 --device cpu
```

把服务器产生的 `policy-0000.zip` 和 `policy-0000.ready.json` 传回 NUC 的 `artifacts/my-infantry-run/`，最后发布 ready 文件。NUC 保留原生进程、场景随机流、估计器、当前观测和历史，加载新权重后继续第二轮；不会因为等服务器而推进仿真。重复编号即可。可同时取回 Actor 导出文件，部署不需要训练 checkpoint。完整 SB3 检查点的跨机加载应保持 NumPy 大版本兼容；本轮 NUC 使用 1.24.4、服务器使用 1.26.4。持续采样回传只加载张量参数和优化器，不反序列化服务器上的环境对象。CUDA 更新要求服务器安装匹配的 CUDA PyTorch；短程验收采用 CPU。

服务器训练使用的源码快照位于 `/data/prototype956/rm_vision/rm_vision_rl/artifacts/asymmetric-runtime`，解释器为原项目 `.venv-asymmetric/bin/python`。该隔离快照用于避免覆盖服务器原有源码；长期使用时应同步并核对同一版本。

旧观测模式保留用于旧实验。新版短程模型仅用于验证采样、优化、保存加载和独立推理链路，不代表模型已收敛或完成实车验证。
