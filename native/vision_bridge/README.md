# 视觉火控桥接

把合成二维测量送入真实 PnP、预测器和火控，并将 Python 策略动作接入同一次控制计算。通过标准输入/输出传递 JSONL，本程序不独立启动仿真或训练。

完整示例见[构建与运行](../../docs/running.md)，数据说明见[接口与配置](../../docs/interfaces.md)。从本仓库根目录构建：

```bash
cmake -S native/vision_bridge -B artifacts/build/vision-bridge \
  -DVISION_ROOT=/home/dgx_spark/RM_VISION_WORKSPACE/rm_vision_2027 -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build/vision-bridge --parallel 2
```

生成 `artifacts/build/vision-bridge/rmvision-rl-bridge`：

```text
rmvision-rl-bridge CONFIG_MODULE_DIR LOGGER_YAML [DIAGNOSTICS_JSONL]
```

`CONFIG_MODULE_DIR` 包含视觉模块 YAML，`LOGGER_YAML` 指定日志配置；可选诊断文件不进入策略观测。推荐由 Python `vision_worker` 管理进程和日志。桥接不依赖 OpenVINO 检测推理，C++ 依赖以 [CMakeLists.txt](CMakeLists.txt) 为准。

Gym 在预热确认后使用 `begin_training(round_id, start_ns)` 开启无截止时间的只开火策略模式，
时间截断由 Python 环境管理。`step_policy` 返回带 token 的当周期 `policy_observation` 并暂停；
`policy_action(token, action)` 恢复相同控制计算。命令提交给仿真后仍必须 `ack(success)`。

尚未返回动作时可发送 `policy_cancel(token)`，取消已计算但未发布的命令则发送 `cancel`。
二者都不发布命令或推进世界，并要求下一轮从 Reset 开始。Python 提供 `prepare/submit/cancel`
对应这些操作，原有 `step(response, policy)` 同步接口和 `begin_evaluation` 有界评估保持可用。
