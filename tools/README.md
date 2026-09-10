# 辅助工具

当前只保留 `config/prepare_vision_profile.py`。历史基线、验收和诊断脚本已删除，不再使用旧命令。环境启动和策略示例见[构建与运行](../docs/running.md)。

## 配置副本生成器

[prepare_vision_profile.py](config/prepare_vision_profile.py) 复制视觉项目的 `src/config/modules/`，按 [mpc-recovery.json](../config/policy/mpc-recovery.json) 调整 MPC 单次求解迭代上限。副本保持原目录层级，另生成 `profile.json` 记录来源和配置指纹，不修改默认配置，也不自动选择运行时配置。

**当前不可直接运行：** 仍从已删除的 `tools.validation.validate` 导入 `require`，包括 `--help` 都会报 `ModuleNotFoundError`。后续应解除此依赖，无需恢复整套验收工具。本次文档重建未修改脚本。

设计参数为 `--vision-root`、`--profile`、`--output`。输出必须是视觉项目之外的新目录，源迭代上限须匹配 profile 的 `source_max_iterations`。当前 profile 指定 50 → 200，不代表已验证的通用配置。
