# 辅助工具

此目录维护训练分析和人工验证工具。工具依赖 `src` 的运行接口，
运行库和训练入口不反向依赖工具。所有命令从仓库根目录执行，完整参数见[构建与运行](../docs/running.md)。

## 训练分析与人工验证

| 文件 | 职责 |
| --- | --- |
| [training/analysis.py](training/analysis.py) | 独立评估已完成训练的检查点、选择最佳模型、组织报告和回放 |
| [training/report.py](training/report.py) | 生成训练曲线和模型对比 HTML 报告 |
| [training/view.py](training/view.py) | 单模型评估、回放记录和原生窗口显示 |
| [training/manual.py](training/manual.py) | 人工操作实际 Gym 环境，记录逐步奖励和事件 |
| [training/diagnose.py](training/diagnose.py) | 从检查点执行短期 PPO 更新并采集学习信号 |
| [training/diagnostic_report.py](training/diagnostic_report.py) | 生成学习信号诊断报告 |

```bash
python -m tools.training.analysis --run-dir <训练目录> --no-view
python -m tools.training.view --checkpoint <模型.zip>
python -m tools.training.manual --config config/training/rotation_fire_ppo.json
python -m tools.training.diagnose --checkpoint <模型.zip> --rollouts 3
```

训练使用 `python -m src.training.train`，完成后按需运行上述工具。
分析、回放、人工操作和诊断入口统一位于 `tools.training`，运行库位于 `src`。
`report.py` 和 `diagnostic_report.py` 由对应工具调用。

历史 `tools.validation` 等验收命令不再使用。

默认手动环境每回合随机生成原地旋转靶，R 切换下一场景。静止靶仍可显式选择
`config/training/static_fire_ppo.json`。检查点评估使用保存场景序列的首个参考场景，
不将单场景分数解释为泛化表现。
