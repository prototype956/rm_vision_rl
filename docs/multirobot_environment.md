# 2027 联盟赛多车型火控环境

本轮接入仿真与训练环境，不更新模型，不改网络结构、奖励或策略观测/动作维度。策略继续面对四块候选装甲，只负责选甲和开火；双方底盘与机构运动由脚本指定。重装仅为敌方 450HP 大装甲近似靶，没有主动射击。

## 配置与启动

- 步兵：`config/training/rmul_2027_multirobot_infantry.json`
- 哨兵：`config/training/rmul_2027_multirobot_sentry.json`
- Gym 注册名：`RMMultiRobotFire-v0`，类为 `MultiRobotFireEnv`。

两份配置使用现有 shared-score MLP、选甲九动作、选择记忆和射击时钟，CPU、单环境，默认预算为一个 1024 步 rollout。本轮没有执行模型更新或长训练。

NUC 源码位置为 `/home/nuc/Workspace/rm_vision_rl`，独立 Python 环境为 `.venv-rmul`。仿真根目录自动识别同级 `rm_simulator_2027` 或 `bevy_robomaster_simulator`，也可通过环境构造参数显式覆盖；源配置不保存机器专用绝对路径。

```sh
cd /home/nuc/Workspace/rm_vision_rl
export PATH="$HOME/.cargo/bin:$PATH"
# 仅验收环境：不会训练或更新模型，不需要远程桌面。
.venv-rmul/bin/python -m tools.training.multirobot_smoke \
  --output artifacts/multirobot-check
```

如后续手动进行短训练：

```sh
.venv-rmul/bin/python -m src.training.train \
  --config config/training/rmul_2027_multirobot_infantry.json \
  --output-dir artifacts/training/rmul-infantry-check
```

哨兵替换配置文件即可。该命令只是使用说明，本轮没有自动执行。

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

## 已执行记录与待验项目

NUC 自动检查已通过 72 种基础角色/车型组合、四种步兵预设、双方高地/隧道模板、同种子重现、旧训练入口及 6 项战斗规则回归。

真实视觉桥接记录：普通步兵结算 350HP、重装结算 450HP、低矮哨兵结算 400HP；己方升降配合敌方组合运动产生 220 点伤害；己方折叠穿隧道运行 900 步、产生 180 点伤害，并确认丢失后恢复。动作均来自现有合法动作掩码，没有读取敌方真值来控制射击。

单发弹量结算诊断首次因临时配置重复 `[combat]` 表而未启动，修复已在本机完成，尚未重新同步、运行。额外的无窗口 GUI 相机与车型重置检查也尚未运行。继续远端写入及启动补充检查被自动审批阻止，需新的明确授权后才能完成这两项。

NoMachine 驾驶、按键、视角、机构动作和实际视觉手感留待人工验收；本轮没有操作远程桌面或注入键鼠。机器人结构与动力学近似见仿真仓库 `docs/MULTIROBOT_FIRE_CONTROL.md`。
