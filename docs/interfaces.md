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

响应含 `ok`、关联编号、`sim_time_ns` 和 `data`。data 中 `feedback`、`self_referee`、`visual_frames` 用于控制；`evaluation`、`events`、`reward_damage` 用于独立统计。Reset 的 `previous_round` 是旧回合截断摘要，不是新回合奖励。

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

`EvaluationConfig` 默认 `window_ms=30000`、`max_settle_steps=1000`、`target_hp=100000`。这些是评估设置，不是已经完成的 RL 终止/截断设计。

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

[observations.py](../rmvision_rl/policy/observations.py) 显式选择字段，按 [v1.json](../config/observations/v1.json) 顺序编码，不展开整个诊断对象。

| 输出 | 约定 |
| --- | --- |
| `version` | 1 |
| `features` | 8 × 90，从旧到新；float32 精度数值，缩放并裁剪至 [-1,1] |
| `valid` | 8 个历史行有效性标记 |
| `action_mask` | 9 个布尔值 |

`TensorPolicy` 给 actor 的数据是普通 Python 列表，不是 NumPy/PyTorch 张量。每个控制步推进历史；无语义回调时该行保持零和 valid=false。回合 Reset 清空历史；目标代次变化的复位衔接仍需完善。actor 接收副本，不能修改后续周期历史。

## 奖励与评估分数

仿真 `reward_damage` 是本方本步实际伤害增量。首版目标使用伤害作为基础奖励；Python 尚未实现训练器需要的逐步 reward 与 terminated/truncated 完整适配。

`WindowScore` 是独立评估计分器：只统计窗口 `[start_ns,end_ns)` 内实际出膛弹丸最终造成的本方实际伤害。窗口内出膛、窗口后命中计入；窗口后或恰好截止出膛不计入。重复事件不重复加分；结算不完整或事件异常时，不提供完整 `official_damage`。

关窗停止新请求，已有供弹和在途弹丸继续按物理规则结算。Reset 截断旧世界，不能替代自然结算。训练封装仍需决定尾部奖励与价值自举，不能把评估总分复制成每一步奖励。
