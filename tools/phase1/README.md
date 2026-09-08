# 阶段一验收工具

生产代码位于视觉项目；本目录保留规则差分及有界 Foxglove JSON 采样工具。
`legacy/` 是 53c3f00084a01af24cee51355de3adf0d949359d 基线的控制线程去 I/O
适配副本和预测外推提取，仅用于比较，不可作为训练生产实现。原始算法来自该提交，
会话适配不等同于原实时线程的调度行为；完整渲染闭环单独验收。

在 `artifacts/phase1/baseline` 准备该提交的源码（原始快照哈希见
`artifacts/phase1/snapshot.json`），然后：

```bash
cmake -S tools/phase1 -B artifacts/phase1/build-baseline \
  -DVISION_ROOT="$PWD/artifacts/phase1/baseline" -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/phase1/build-baseline --parallel 4
./artifacts/phase1/build-baseline/phase1-reference artifacts/phase1/baseline/src/config/modules > artifacts/phase1/baseline.csv
./artifacts/phase1/build-baseline/phase1-session-reference artifacts/phase1/baseline/src/config/modules > artifacts/phase1/baseline-session.csv
```

新实现两个 reference 程序的输出分别用 `compare_reference.py` 比较（会话程序加 `--session`）。
固定 8 个场景，每个 600 周期：静止、横移、旋转、组合运动、短丢失、过期预测、
外部使能变化/发布失败、高不确定度。新实现注入新鲜可发射裁判观测，以隔离规则等价验证；
新裁判约束另由 `mv-control-acceptance` 验收。整数/布尔精确比较，浮点容差 1e-9/1e-7。

```bash
python3 tools/phase1/compare_reference.py BASELINE.csv CANDIDATE.csv --output REPORT.json
python3 tools/phase1/compare_reference.py BASELINE_SESSION.csv CANDIDATE_SESSION.csv --session --output SESSION_REPORT.json
python3 tools/phase1/collect_control.py --seconds 30 --output artifacts/phase1/control.jsonl
```

JSON 采样只读且有界，不启用 MCAP。`run_closed_loop.py` 运行正式项目启动脚本，
会备份录制配置、临时关闭录制，并在结束时停止子进程和恢复原文件。若用户在运行期间
修改了配置，则保留用户改动并指出备份位置；该操作需要视觉目录写权限和本机图形权限。
`summarize_control.py` 将有效跟踪与全部采样分别统计；`combat_snapshot.py` 只读提取
v7 当前稳定帧中的战斗计数，不向策略提供这些真值，也不代表可靠的逐事件训练账本。
