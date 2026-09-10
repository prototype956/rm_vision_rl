# 视觉火控桥接

把合成二维测量送入真实 PnP、预测器和火控，并将 Python 策略动作接入同一次控制计算。通过标准输入/输出传递 JSONL，本程序不独立启动仿真或训练。

完整示例见[构建与运行](../../docs/running.md)，数据说明见[接口与配置](../../docs/interfaces.md)。从本仓库根目录构建：

```bash
cmake -S native/vision_bridge -B artifacts/build/vision-bridge \
  -DVISION_ROOT=/home/nuc/Workspace/rm_vision_2027 -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/build/vision-bridge --parallel 2
```

生成 `artifacts/build/vision-bridge/rmvision-rl-bridge`：

```text
rmvision-rl-bridge CONFIG_MODULE_DIR LOGGER_YAML [DIAGNOSTICS_JSONL]
```

`CONFIG_MODULE_DIR` 包含视觉模块 YAML，`LOGGER_YAML` 指定日志配置；可选诊断文件不进入策略观测。推荐由 Python `vision_worker` 管理进程和日志。桥接不依赖 OpenVINO 检测推理，C++ 依赖以 [CMakeLists.txt](CMakeLists.txt) 为准。
