# 文档示意图

这些图说明实现结构、状态规则和发布条件，不表示已完成实验或通过验收。Markdown 默认嵌入 SVG，支持普通图片的阅读器即可显示；同名 `.mmd` 文件供 Mermaid 编辑器使用。

| 图 | 阅读位置 | 对应实现 |
| --- | --- | --- |
| [闭环数据流](control_loop.svg) | [架构](../architecture.md#当前闭环) | `transport/vision_bridge.py`、`training/recurrent_policy.py`、`training/recurrent_ppo.py` |
| [GRU 状态生命周期](gru_lifecycle.svg) | [持续记忆 GRU](../persistent_gru.md#状态生命周期) | `policy/recurrent_observations.py`、`policy/recurrent_actor.py`、`training/recurrent_policy.py` |
| [v2 冻结训练](curriculum_v2_release.svg) | [课程 v2](../curriculum_training_v2.md#发布与训练流程) | `tools/training/curriculum_v2.py`、`curriculum_v2_pipeline.py` |
| [v3 修订 3 发布](curriculum_v3_release.svg) | [课程 v3](../curriculum_training_v3.md#发布流程) | `tools/training/curriculum_v3.py`、`curriculum_v3_release.py` |

从仓库根目录重新生成全部图：

```bash
python docs/figures/generate.py
```

[generate.py](generate.py) 仅使用 Python 标准库，从同一份节点与边定义生成 SVG 和 Mermaid。修改时以生成脚本为准，再生成两种文件，避免图意分叉。SVG 布局由脚本中的坐标控制，Mermaid 编辑器使用自己的自动布局。

核验时对照代码检查数据入口、状态重置条件、流程顺序及失败分支；在浏览器渲染 SVG，检查中文字体、文字边界、连线和箭头。默认字体依次为 Microsoft YaHei、Noto Sans CJK SC 和系统 sans-serif。
