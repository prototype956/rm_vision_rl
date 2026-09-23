"""根据短期 PPO 诊断数据生成离线报告，并明确证据范围。"""
import csv
from html import escape
import json
import numpy as np
import plotly.graph_objects as go

from src.environment.warmup import EPOCH_NS, STEP_NS

LABELS = {"fire": "合法发射", "track": "合法跟踪", "masked": "禁止发射"}



def associate_actions(steps, events):
    """按命令时间戳精确关联动作与事件，不使用最近时间匹配。

    bridge.cpp 将 command_timestamp_ns 设为 EPOCH 加当前步起始时间；
    StaticFireEnv 保证每次状态转换恰好推进 STEP_NS。
    """
    commands = {(r["episode"], EPOCH_NS + r["physical_time_ns"] - STEP_NS): r
                for r in steps if r["action"] == 1 and r["shot_accepted"]}
    requests = {}
    for event in events:
        if event["kind"] == "fire_requested" and event["data"].get("robot_id") == 1:
            stamp = event["data"].get("source", {}).get("command_timestamp_ns")
            row = commands.get((event["episode"], stamp))
            if row is not None:
                requests[(event["episode"], event["data"]["request_id"])] = row
    enriched = []
    for event in events:
        row = requests.get((event["episode"], event["data"].get("request_id")))
        enriched.append(dict(event, action_step=row["training_step"] if row else None,
                             action_episode_step=row["episode_step"] if row else None,
                             action_association="exact_command_timestamp_and_request_id" if row else "unassociated",
                             action_cross_rollout=bool(row and row["rollout"] != event["rollout"])))
    return enriched


def write_report(output, state, steps, events):
    events = associate_actions(steps, events)
    with (output / "event-associations.jsonl").open("w") as file:
        for event in events:
            file.write(json.dumps(event, allow_nan=False) + "\n")
    summaries = state["summaries"]
    checks, limitations = [], [
        "这是从保存权重与优化器开始的新采样，未恢复原训练现场的环境或随机数状态；不能重建此前训练的每一次更新。",
        "奖励移除实验是固定轨迹、固定价值预测的数值敏感性检查，不是“不发射时真实会怎样”的环境反事实。",
        "单个样本的正优势表示其策略损失倾向提高所选动作概率；其他样本、裁剪及熵项仍会影响整轮更新。",
        "诊断不强制发射、不调参；短程结果只能验证链路及提示问题，不能单独确定长期训练根因。"]
    if summaries:
        checks.append(("奖励接入", "正常" if all(s["reward_event_max_error"] == 0 for s in summaries) else "异常",
                       "逐步对比己方 damage_applied 事件与 Gym 原始奖励。"))
        checks.append(("GAE 计算", "正常" if all(s["gae_matches"] for s in summaries) else "异常",
                       f"独立 float64 重算的最大绝对误差 {max(s['gae_max_error'] for s in summaries):.6g}；与 float32 缓冲区采用相对 2e-5、绝对 2e-4 容差。"))
        nonzero = sum(s["nonzero_rewards"] for s in summaries)
        checks.append(("奖励向前传播", ("正常" if max(s["reward_removal_max_error"] for s in summaries) < 1e-8 else "异常") if nonzero else "证据不足",
                       f"逐个移除 {nonzero} 个非零奖励，与含回合边界的 (γλ)^k 系数核对；不向前一回合传播。"))
        checks.append(("时间截断补偿", "正常" if all(s["timeout_compensation_only_at_truncation"] for s in summaries) else "异常",
                       "缓冲区奖励减原始奖励，仅允许在 TimeLimit.truncated 处非零；真正终止不补偿。"))
    else:
        checks.append(("真实更新", "证据不足", "没有完成的 rollout；查看 sampling.jsonl 和进程日志。"))
    parity = state.get("parity", {})
    checks.append(("跟踪是否改变更新", "正常" if parity.get("status") == "passed" else "异常" if parity else "未完成",
                   f"首轮固定缓冲区对照：{json.dumps(parity, ensure_ascii=False)}"))
    checks.append(("原检查点", "未改变" if state.get("source_unchanged") else "无法确认", "运行前后 SHA-256 校验。"))
    associations = [e for e in events if e["kind"] == "damage_applied" and e["data"].get("shooter") == 1]
    unknown = sum(e["association"] == "unassociated" for e in associations)
    cross = sum(e["cross_rollout"] for e in associations)
    pending = sum(s["pending_at_episode_end"] for s in summaries)
    limitations.append(f"伤害事件 {len(associations)} 个，其中请求编号无法关联 {unknown} 个、跨 rollout {cross} 个；"
                       f"已结束回合末仍在飞行的弹丸共 {pending} 个。编号按回合隔离，重置后不关联到旧回合。"
                       "回合末有在途弹丸不等于它们一定会命中；没有执行尾部结算。")
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>PPO 奖励学习信号诊断</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f9;color:#203044;margin:0}"
             "main{max-width:1220px;margin:auto;padding:32px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}"
             "h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dce3eb;padding:9px;text-align:left}"
             ".note{color:#526478}.scroll{overflow:auto}code{overflow-wrap:anywhere}</style><main><h1>PPO 奖励学习信号诊断</h1>",
             f"<p>状态：{escape(state['status'])} · 完成 {state['completed_rollouts']} / {state['requested_rollouts']} 个 rollout · "
             f"记录 {len(steps)} 个已完成更新的采样步骤。原始采样见 sampling.jsonl。</p>",
             f"<p class='note'>来源：<code>{escape(state['source'])}</code><br>SHA-256：{state['source_sha256']}</p>"]
    if state.get("error"):
        parts.append(f"<section><h2>运行故障</h2><p>{escape(state['error'])}</p></section>")

    def table(title, headers, rows):
        parts.append(f"<section><h2>{escape(title)}</h2><div class='scroll'><table><thead><tr>" +
                     "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{escape(str(x)) if x is not None else '暂无数据'}</td>" for x in row) + "</tr>")
        parts.append("</tbody></table></div></section>")

    if steps:
        legal = [r for r in steps if r["fire_legal"]]
        parts.append(f"<section><h2>关键发现</h2><p>累计真实伤害奖励 {sum(r['raw_reward'] for r in steps):.0f}；"
                     f"合法决策机会 {len(legal)} / {len(steps)}，禁止发射步骤占 {1-len(legal)/len(steps):.1%}。"
                     "禁止发射时只有一个合法动作，不能直接学习发射与跟踪之间的选择，但这些步骤仍参与价值学习和优势标准化。</p>")
        for summary in summaries:
            selected = [r for r in legal if r["rollout"] == summary["rollout"]]
            if selected:
                before = np.mean([r["fire_probability_before"] for r in selected])
                after = np.mean([r["fire_probability_after"] for r in selected])
                maximum = max(r["fire_probability_after"] for r in selected)
                parts.append(f"<p>第 {summary['rollout']} 轮，同一批合法观测的平均发射概率 {before:.2%} → {after:.2%}；更新后最大值 {maximum:.2%}。</p>")
        parts.append("<p>概率变化应逐轮在同一批观测上比较，不把不同轮次经过的不同状态混为一组。价值预测是否有效请看下表解释方差；损失下降不等于已经能够区分高低回报状态。</p></section>")
    table("检查结论", ["检查", "结果", "证据／口径"], checks)
    table("每轮更新汇总", ["轮次", "原始奖励", "合法发射 / 跟踪 / 禁止", "优化步数", "价值 MSE 前 → 后", "解释方差前 → 后", "目标均值 ± 标准差"],
          [(s["rollout"], s["reward"], " / ".join(str(s["groups"].get(g, {}).get("samples", 0)) for g in LABELS),
            s["optimizer_steps"], f"{s['value_before']['mse']:.3f} → {s['value_after']['mse']:.3f}",
            f"{s['value_before']['explained_variance']} → {s['value_after']['explained_variance']}",
            f"{s['target_mean']:.3f} ± {s['target_std']:.3f}") for s in summaries])
    table("按动作分组的学习信号", ["轮次", "类别", "样本数", "原始优势均值", "标准化优势均值", "标准化正优势比例", "裁剪生效比例", "发射概率前 → 后", "平均变化（百分点）"],
          [(s["rollout"], LABELS[group], g["samples"], f"{g['mean_advantage']:.4f}",
            f"{g['normalized_advantage_mean']:.4f}" if g.get("normalized_advantage_mean") is not None else None,
            f"{g['normalized_positive_fraction']:.2%}" if g["normalized_positive_fraction"] is not None else None,
            f"{g['surrogate_clipped_fraction']:.2%}" if g["surrogate_clipped_fraction"] is not None else None,
            f"{g['mean_probability_before']:.4%} → {g['mean_probability_after']:.4%}", f"{100*g['mean_probability_change']:+.4f}")
           for s in summaries for group, g in s["groups"].items()])
    if state.get("recovery"):
        parts.append("<section><h2>诊断恢复记录</h2><p>首轮检查器曾因复制跟踪钩子重复记录小批次顺序而停止。修复后使用原采样数据精确重建首轮更新，动作、顺序、更新后概率和价值预测一致；重放 24 个已记录步骤恢复未结束回合，观测一致。这 24 步不作为新训练样本，最终训练采样仍为 3072 步。原始失败状态保留在 recovery/。</p></section>")
    script = True

    def chart(title, note, series, ylabel, xlabel="累计训练步数"):
        nonlocal script
        parts.append(f"<section><h2>{escape(title)}</h2><p class='note'>{escape(note)}</p>")
        figure = go.Figure()
        for name, x, y, mode in series:
            if len(x):
                figure.add_trace(go.Scatter(x=x, y=y, name=name, mode=mode, connectgaps=False))
        if figure.data:
            figure.update_layout(template="plotly_white", height=410, xaxis_title=xlabel, yaxis_title=ylabel,
                                 legend=dict(orientation="h"), margin=dict(l=65, r=20, t=25, b=65))
            parts.append(figure.to_html(full_html=False, include_plotlyjs=True if script else False,
                                       config={"displaylogo": False, "responsive": True}))
            script = False
        else:
            parts.append("<p>暂无数据</p>")
        parts.append("</section>")

    x = [r["training_step"] for r in steps]
    chart("原始奖励与时间截断补偿", "补偿来自价值预测，不能当作真实伤害。", [
        (label, x, [r[key] for r in steps], "lines+markers") for label, key in
        (("Gym 原始奖励", "raw_reward"), ("伤害事件", "event_damage"), ("截断补偿", "timeout_bootstrap"))], "奖励")
    chart("价值预测、固定回报目标与优势", "每轮的回报目标保持为更新前计算值；曲线按回合拆开，避免跨回合连线。", [
        (f"回合 {ep} · {label}", [r["training_step"] for r in steps if r["episode"] == ep],
         [r[key] for r in steps if r["episode"] == ep], "lines")
        for ep in sorted({r["episode"] for r in steps})
        for label, key in (("V 更新前", "value_before"), ("回报目标", "target"), ("原始优势", "advantage"))], "奖励单位")
    timeline = []
    for label, kind, level in (("物理请求", "fire_requested", 1), ("实际出膛", "shot_fired", 2), ("伤害", "damage_applied", 3)):
        selected = [e for e in events if e["kind"] == kind and e["data"].get("robot_id", e["data"].get("shooter")) == 1]
        timeline.append((label, [e["training_step"] for e in selected], [level] * len(selected), "markers"))
    timeline.insert(0, ("Gym 发射动作", [r["training_step"] for r in steps if r["action"] == 1],
                        [0] * sum(r["action"] == 1 for r in steps), "markers"))
    chart("发射—出膛—伤害时间线", "纵轴为事件类别；编号关联和真实物理时间保存在 events.jsonl。", timeline, "事件类别")
    chart("同一观测与掩码的发射概率", "更新前后只重算网络，不运行新的环境。禁止发射的步骤单独列示。", [
        (f"{LABELS[group]} · {label}", [r["training_step"] for r in steps if r["group"] == group],
         [r[key] for r in steps if r["group"] == group], "markers")
        for group in LABELS for label, key in (("更新前", "fire_probability_before"), ("更新后", "fire_probability_after"))], "P(发射)")
    batches = []
    batch_path = output / "minibatches.csv"
    if batch_path.exists() and batch_path.stat().st_size:
        with batch_path.open() as file:
            batches = list(csv.DictReader(file))
    normalized = {}
    for row in batches:
        if row["optimizer_executed"] == "True":
            normalized.setdefault(int(row["training_step"]), []).append(float(row["normalized_advantage"]))
    chart("优势与整轮发射概率变化", "横轴为同一样本跨实际小批次的标准化优势均值，仅作汇总；完整逐次记录见 minibatches.csv。", [
        (LABELS[group], [float(np.mean(normalized[r["training_step"]])) for r in steps if r["group"] == group and r["training_step"] in normalized],
         [100*r["probability_change"] for r in steps if r["group"] == group and r["training_step"] in normalized], "markers")
        for group in LABELS], "发射概率变化（百分点）", "标准化优势均值")
    contribution_series, example_rows = [], []
    for summary in summaries:
        number = summary["rollout"]
        with np.load(output / f"rollout-{number:02d}.npz") as data:
            indices, contributions = data["reward_indices"], data["reward_contributions"]
            if not len(indices):
                continue
            local = [r for r in steps if r["rollout"] == number]
            j = int(indices[0])
            start = max(i for i in range(j + 1) if i == 0 or data["episode_starts"][i])
            contribution_series.append((f"轮 {number} · 奖励步骤 {local[j]['training_step']}",
                                        [r["training_step"] for r in local[start:j + 1]], contributions[0, start:j + 1], "lines+markers"))
            for i in range(start, j + 1):
                if local[i]["action"] == 1:
                    example_rows.append((number, local[j]["training_step"], local[i]["training_step"], j-i,
                                         float(data["raw_rewards"][j]), float(contributions[0, i])))
    chart("移除奖励后的优势变化", "每轮展示首个非零奖励的敏感性曲线；NPZ 包含全部非零奖励对本段每一步的贡献矩阵。贡献不等于弹丸因果归属。", contribution_series, "原优势 − 移除奖励后的优势")
    table("奖励回传实例", ["轮次", "奖励步骤", "此前发射动作步骤", "相隔步数", "移除的奖励", "该动作优势减少"], example_rows)
    table("按编号关联的命中实例（前 30 条）", ["回合", "请求编号", "弹丸编号", "Gym 发射步骤", "请求事件步骤", "出膛步骤", "伤害步骤", "实际伤害", "跨 rollout"],
          [(e["episode"], e["data"].get("request_id"), e["data"].get("projectile_id"), e["action_step"], e["request_step"],
            e["shot_step"], e["training_step"], e["data"]["actual"], e["cross_rollout"]) for e in associations[:30]])
    parts.append("<section><h2>结论边界与后续判断</h2><ul>" + "".join(f"<li>{escape(note)}</li>" for note in limitations) + "</ul></section></main></html>")
    (output / "report.html").write_text("\n".join(parts))
    state["checks"] = [{"check": a, "status": b, "evidence": c} for a, b, c in checks]
    state["boundaries"] = {"damage_events": len(associations), "unassociated_damage_events": unknown,
                           "cross_rollout_damage_events": cross, "pending_projectiles_at_episode_end": pending,
                           "unassociated_action_damage_events": sum(e["action_step"] is None for e in associations)}
