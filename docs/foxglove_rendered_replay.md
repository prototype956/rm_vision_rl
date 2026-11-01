# 带第一人称与第三人称画面的 MCAP

NUC 上的 `training_preview` 是物理状态回放器：读取已记录的位置、姿态和弹丸状态后重新绘制场景。原 GRU MCAP 入口使用无渲染锁步仿真，因此 `/vision/camera/image` 原本是标有“合成测量”的画布；图像检测器、灯条提取和角点精修没有运行。

`tools.training.render_mcap` 将一份完整的原始 MCAP 离线重绘为**新的文件**，不重新运行策略、检测器或物理仿真。常规训练不会启用渲染。仅在用户显式要求时导出有限案例。

## 构建和使用

在带 Vulkan GPU 的 Linux 环境构建现有查看器扩展，无须桌面、X11 或可见窗口：

```bash
cd rm_simulator_2027
cargo build --release --example training_preview --features training
```

为离线工具单独准备 Python 依赖，避免修改正式训练的冻结环境：

```bash
cd rm_vision_rl
python3 -m venv .venv-replay
.venv-replay/bin/python -m pip install -r requirements-replay.txt
.venv-replay/bin/python -m tools.training.render_mcap \
  --source /absolute/path/original.mcap \
  --work /absolute/path/new-render-work \
  --output /absolute/path/rendered.mcap \
  --renderer /absolute/path/rm_simulator_2027/target/release/examples/training_preview \
  --config /absolute/path/original-simulator-config.toml \
  --assets /absolute/path/original-assets
```

使用与源录制相同的冻结配置和资产；配置文件 SHA256 必须与源文件一致。动态链接构建按项目运行环境设置 `LD_LIBRARY_PATH`，包含 `target/release/deps` 和 Rust 运行库目录。无桌面服务器可为本任务设置私有 `XDG_RUNTIME_DIR`。工具拒绝覆盖源文件、输出文件或工作目录；失败产物保留供诊断。

`--prepare-only --limit 2` 可用于少量帧的渲染器开发检查；不完整渲染不能组装成完整 MCAP。

## Foxglove 话题

|话题|内容与用途|
|---|---|
|`/vision/camera/image`|第一人称重建画面，保持源图像采集时间戳和相机 frame_id，沿用现有 Image 面板标注|
|`/replay/render/third_person`|固定轨道第三人称画面，观察双方运动、选板与弹丸，不叠加第一人称像素标注|
|`/replay/render/measurement_canvas`|保留原始合成测量画布，方便核对|
|`/replay/info.render`|源文件及渲染器哈希、重建方法、图像话题和模型身份|

在 Foxglove 添加两个 Image 面板，分别选择第一人称与第三人称话题。原有 PnP、预测、控制、仿真与裁判话题保留原始消息内容和时间戳；新增图像不会改变发射、命中、击毁或结算成绩。源测量仍是 `synthetic_measurements`，`real_camera=false`，新增 `rendered_camera=true`；这不是渲染图像经过真实检测网络后的性能测试。

## 时间、几何与验收口径

- 每个源视觉帧生成两张图像，并等待 GPU 截图回读确认后才推进下一帧。图像按仿真采集时间写入，磁盘慢时导出变慢，不丢帧。
- 从采集真值的三维角点和无噪声理想投影恢复绝对光学相机姿态。不能使用控制周期 TF 代替采集姿态，也不使用 PnP 估计或带噪测量拟合相机。
- 初始平面位姿经所有可用真值角点优化，最大重投影误差不超过 0.02 px；光学原点必须位于己方机器人安装包络内。
- 双方位置和车体航向使用采集时刻真值；机构姿态、己方云台外观和弹丸位置仅在相邻完整 10 ms 物理样本间插值。这些都是展示数据，不进入策略观测或控制。
- 渲染器再次调用仿真共用的真实资产几何计算，对实际加载的装甲网格进行独立投影检查；任一角点偏差超过 0.5 px 就停止，不通过放宽门槛掩盖错位。
- 当前支持原录制中的居中、方形像素、无畸变相机和单回合。缺少采集真值、可用投影角点或完整物理采样时明确失败，不补造图像。
- 新文件重新读取并检查 CRC、双视角帧数；除原图像和新增渲染元数据外，其他诊断消息逐字节一致。原始文件哈希在导出前后保持一致。

工作目录保留 `prepare.json`、`images/complete.json`、`images/adapter.json`、`validation.json`。输出 MCAP 包含同样的渲染来源说明。新文件可继续使用仓库的 MCAP 只读分析工具执行 `inspect`、PnP／预测／控制预设与有限叠加截图。

## 可开关战斗 HUD

显式 MCAP 回放导出（`tools.training.view --record-mcap`）、有限课程对照导出和渲染导出默认 `combat_hud=true`。可用 `--no-combat-hud` 关闭；回放 JSON 配置也可设置 `"combat_hud": false`。训练控制台回放表单提供“附带可开关的战斗 HUD（仅 MCAP）”选项。常规训练、旧 JSON 播放均不启用录制。

安装 `requirements-replay.txt` 中的离线依赖后，可以为已有录制追加 HUD，输出必须是新文件：

```bash
python -m tools.training.combat_hud --input /path/original.mcap --output /path/new.hud.mcap
```

该工具复用 `WindowScore`，核对源结算与物理计数器，保留原消息的内容、采集/发布时间、序号、Schema 和渲染来源元数据，并生成同名 `.hud.json` 校验收据。重复处理含 HUD 的文件时仅替换这两个派生注释话题。缺少回放身份、物理账本或图像时明确失败；存在第三人称图像时才生成第三人称 HUD。原始合成测量画布同样支持第一人称 HUD，无须重新渲染。

普通录制先保留 `replay.raw.mcap`，再生成供索引和下载使用的 `replay.mcap`；渲染组装先保留 `*.render-only.mcap`，再生成指定输出文件。HUD 写入或核对失败会使显式任务失败，原始产物保留。HUD 不进入策略、物理模拟或奖励。

### 两路 Image 面板

不需要插件，也不需要替换已有布局。在各 Image 面板的设置中选择：

|视角|图像主题|启用的“图像注释”|同步时间戳|
|---|---|---|---|
|第一人称|`/vision/camera/image`|`/replay/hud/first_person`|关|
|第三人称|`/replay/render/third_person`|`/replay/hud/third_person`|关|

每个面板只打开对应的 HUD 话题，避免两份相同文字叠在一起。第三人称不应启用第一人称的 PnP／预测像素标注。通过话题旁的眼睛图标即可关闭 HUD，原图像字节完全没有变化。图像面板缩放会同时缩放文字；1280×720 时字号22、行距28，五行位于左上角。

### 统计和时间

- 身份行显示控制器、已有模型步数、课程、车型、运动类型及案例。
- 命中效率按评估窗口内**实际出膛**的弹丸计数，正有效伤害弹丸去重计命中；未发射显示 `—（0/0）`。射击请求不是发射次数。
- 预热不计分；评估及尾部结算显示暂计成绩。在途命中归属原出膛窗口。只有源录制确认完整结算且重建账本一致时显示绿色“最终成绩”。超时或失败显示黄色的不完整状态，缺失数据写“不可用”。
- HP、当前热量来自该时刻以前的物理诊断；普通热锁次数及累计锁定时间从接管基线开始计算，结算期间继续更新。热锁为黄色，不按单例命中率套用课程晋级门槛。
- 每条注释是完整快照，正常图像只使用采集时刻以前的事件，不依赖客户端播放历史。倒放、跳转或切换文件不会通过客户端累计统计。
- 结算期间图像停止时，HUD 继续按物理采样更新，写明“定格 Xs／统计 Ys”；两者均相对策略接管时刻。关闭“同步时间戳”才能让较新的注释覆盖最后图像。画面定格不代表统计已完成。

标准横移 GRU 验收参考：39发、18次有效命中、350伤害、46.15%；最终己方HP350、目标HP0、热量82/88、普通热锁2次、锁定7.39秒。最后图像在接管后12.90秒，完整统计在12.93秒。
