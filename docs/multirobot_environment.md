# 2027 联盟赛多车型火控环境

本页描述多车型物理环境及其独立诊断，当前训练支持持续记忆 GRU 的课程 v2/v3，CLI 默认 v2。策略继续面对四块候选装甲，只负责选甲和开火；双方底盘与机构运动由脚本指定。重装仅为敌方 450HP 大装甲近似靶，没有主动射击。

## 配置与启动

v2 的训练入口如下；相应 v3 预设和发布流程见[课程 v3](curriculum_training_v3.md)。

- 步兵：`config/training/rmul_2027_infantry_curriculum_v2.json`，也是默认训练配置。
- 哨兵：`config/training/rmul_2027_sentry_curriculum_v2.json`。

两份配置使用持续记忆 GRU、v5 观测、九动作、热锁奖励及 12 阶段课程。
默认单 GPU、128 环境、每类累计十亿步。预检、资格验证和恢复见 [课程 v2](curriculum_training_v2.md)。

```bash
python -m src.training.train --output-dir artifacts/training/infantry-v2
python -m src.training.train \
  --config config/training/rmul_2027_sentry_curriculum_v2.json \
  --output-dir artifacts/training/sentry-v2
# 单独检查物理环境，不更新模型。
python -m tools.training.multirobot_smoke --output artifacts/multirobot-check
```

Gym 注册名为 `RMMultiRobotFire-v0`，类为 `MultiRobotFireEnv`。下文介绍的是底层通用场景
采样能力；课程 v2 另有自己的固定阶段和采样规则，未启用全部地形/机构任务。
源配置不保存机器专用绝对路径；仿真根目录默认识别同级 `rm_simulator_2027`。

## 场景选择

比赛资源预设 `preset` 与几何 `profile` 独立。四种己方车型为 `standard`、`wheel_leg`、`low`、`folding`；敌方额外支持 `heavy_target`。重装不指定步兵/哨兵 preset。

省略车型时，每回合独立采样己方四种底盘；敌方从步兵四底盘、哨兵四底盘及一个重装靶中等概率采样。默认步兵使用血量＋冷却，也可明确选择另外三种步兵参数。每回合记录实际车型、预设、种子、路径及拒绝原因。

运动模板有静止、旋转、平移、平移加旋转、轮腿升降、上下高地和隧道往返。随机采样只选择车型支持的动作；高地和隧道采用固定合法起点及观察位，观察位避开工作台。默认只安排一方执行复杂模板，另一方作为静止观察者，不引入课程调度。

可在训练配置的 `environment.scenario` 中指定固定组合，例如：

```json
{
  "duel": {
    "controlled": {
      "profile": "wheel_leg",
      "preset": "sentry",
      "motion": {"kind": "lift"}
    },
    "target": {
      "profile": "standard",
      "preset": "infantry_health_cooling",
      "motion": {"kind": "combined", "amplitude_m": 0.2, "speed_m_s": 0.3}
    }
  },
  "unlimited_heat": false
}
```

显式几何/路径无效会报出原因，不偷偷移动机器人；普通随机出生最多尝试 64 次。地面、高地和桥面分别检查，不能将隧道顶面误作桥下地面。运行中允许自然遮挡、折叠造成的视野变化与重新跟踪。

## 资源与接口

步兵采用交战状态 200 发、哨兵 750 发，热量/冷却/血量按预设执行；这是训练交战起点，不是正式比赛的 0 发开局。没有补弹、补血、复活或经济流程。

目标击毁、己方死亡为自然终止；弹药耗尽后禁射，继续普通 10ms 控制步结算在途弹，清空后终止。达到 30s 时按时间截断，沿用原来的末观测与价值自举语义，不追加独立评估尾部奖励。新任务限制回合最多 3000 步、预热最多 15s，与联合路径检查的 45s 范围对应。

敌方车型、机构状态、真实姿态及未来路径仅用于场景生成、评估和日志。策略仍从现有视觉估计与己方反馈获取信息，网络输入没有增加这些真值。

Talos v7 布局及训练事务操作保持原状；折叠禁射使用现有枪口不可用位，详细原因保留在诊断信息。旧静止/旋转训练及原实验配置兼容；新任务不接受高血量覆盖和无限热量。

## 历史诊断记录与验证范围

以下数值沿用原文的 NUC/视觉桥接记录。原文未给出运行目录、源码指纹及完整日志入口，本次文档整理未重新执行；因此仅保留为历史观察，不能当作当前版本已验收的证据。

NUC 自动检查已通过 72 种基础角色/车型组合、四种步兵预设、双方高地/隧道模板、同种子重现、旧训练入口及 6 项战斗规则回归。

真实视觉桥接记录：普通步兵结算 350HP、重装结算 450HP、低矮哨兵结算 400HP；己方升降配合敌方组合运动产生 220 点伤害；己方折叠穿隧道运行 900 步、产生 180 点伤害，并确认丢失后恢复。动作均来自现有合法动作掩码，没有读取敌方真值来控制射击。

单发弹量结算诊断首次因临时配置重复 `[combat]` 表而未启动，修复已在本机完成，尚未重新同步、运行。额外的无窗口 GUI 相机与车型重置检查也尚未运行。原记录未提供上述两项补充检查的完成结果，当前无法据此确认是否通过。

NoMachine 驾驶、按键、视角、机构动作和实际视觉手感留待人工验收；原记录未包含远程桌面操作或键鼠验收。机器人结构与动力学近似见[仿真仓库说明](../../rm_simulator_2027/docs/MULTIROBOT_FIRE_CONTROL.md)。
