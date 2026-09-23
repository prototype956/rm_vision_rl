"""根据完整回合的 Monitor 日志和逐次更新的 CSV 日志生成离线 Plotly 报告。

奖励、伤害、发射数及 PPO 指标按训练步数展示；检查点评估伤害作为离散评估点展示。
保留原始回合曲线并叠加最近 20 回合的均值。用蓝色、金色及不同线型和标记区分
Final/Best，表格提供精确数值。
"""
import csv
from html import escape
import math
import os
from pathlib import Path
import tempfile


BLUE, GOLD, MUTED = "#2463a6", "#ac7617", "#a6b8cb"
STATUS = {"complete": "完成", "partial": "部分候选失败", "failed": "失败", "incomplete": "结算不完整",
          "duplicate": "同一步数，已合并", "interrupted": "已中断", "pending": "待评估", "evaluating": "评估中"}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _read_csv(path, warnings):
    if not path.is_file():
        warnings.append(f"缺少 {path.name}，对应曲线暂无数据。")
        return []
    try:
        with path.open(newline="") as file:
            return list(csv.DictReader(line for line in file if not line.startswith("#")))
    except (OSError, UnicodeError, csv.Error) as error:
        warnings.append(f"无法读取 {path.name}：{error}")
        return []


def read_curves(output, run):
    """缺失或非有限数值显示为断点；回合长度无效时停止定位后续累计步数。"""
    output = Path(output)
    warnings, episodes = [], []
    step = run.get("start_timesteps")
    if step is None:
        if run.get("resume_from"):
            warnings.append("恢复训练缺少起始步数，无法定位回合曲线；PPO 曲线仍使用日志中的累计步数。")
        else:
            step = 0
    for row in _read_csv(output / "episodes.monitor.csv", warnings):
        length = _number(row.get("l"))
        if step is None:
            break
        if length is None or length <= 0 or not length.is_integer():
            warnings.append("回合长度缺失或无效：此行及后续回合不绘制，避免累计步数错位。")
            break
        step += int(length)
        if step > run.get("last_completed_timesteps", math.inf):
            warnings.append("回合步数超出训练完成位置，后续回合不绘制。")
            break
        episodes.append({"step": step, **{key: _number(row.get(key))
                                         for key in ("r", "episode_damage", "actual_shots")}})
    updates = _read_csv(output / "logs/progress.csv", warnings)
    return episodes, updates, warnings


def _format(value, *, rate=False):
    if value is None:
        return "—"
    if rate:
        return f"{value:.1%}"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.4f}"


def write_report(output, state, run):
    import plotly.graph_objects as go

    output = Path(output)
    episodes, updates, warnings = read_curves(output, run)
    best, final = state["best"], state["final"]
    parts = []
    script_included = False

    def chart(title, note, series, unit, *, evaluation=False):
        nonlocal script_included
        figure = go.Figure()
        for item in series:
            if not any(x is not None and y is not None for x, y in zip(item["x"], item["y"])):
                continue
            figure.add_trace(go.Scatter(
                x=item["x"], y=item["y"], name=item["name"], connectgaps=False,
                mode=item.get("mode", "lines+markers"),
                line={"color": item.get("color", BLUE), "width": 2, "dash": item.get("dash", "solid")},
                marker={"color": item.get("color", BLUE), "size": item.get("size", 5),
                        "symbol": item.get("symbol", "circle")},
                hovertemplate="累计步数 %{x:,.0f}<br>%{y:,.4g}<extra>%{fullData.name}</extra>"))
        parts.append(f"<section><h2>{escape(title)}</h2><p class='note'>{escape(note)}</p>")
        if not figure.data:
            parts.append("<p class='empty'>暂无数据</p></section>")
            return
        # 评估图用标签展示检查点步数和得分；训练曲线仅以竖线标出检查点，
        # 避免将训练奖励误读为最佳模型的选取依据。
        if not evaluation:
            for item, color, dash in ((best, GOLD, "dash"), (final, BLUE, "dot")):
                if item and item.get("num_timesteps") is not None:
                    figure.add_vline(x=item["num_timesteps"], line_color=color, line_dash=dash, line_width=1)
        figure.update_layout(
            template="plotly_white", height=350, margin={"l": 65, "r": 25, "t": 55, "b": 60},
            font={"family": "system-ui, sans-serif", "size": 13, "color": "#233247"},
            paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
            legend={"orientation": "h", "y": 1.15, "x": 0},
            xaxis={"title": "累计训练步数", "tickformat": ",.0f", "gridcolor": "#eef1f5"},
            yaxis={"title": unit, "gridcolor": "#eef1f5", "zerolinecolor": "#ccd4df"})
        if evaluation or unit in ("奖励", "伤害", "发"):
            figure.update_yaxes(rangemode="tozero")
        parts.append(figure.to_html(full_html=False, include_plotlyjs=not script_included,
                                   config={"responsive": True, "displaylogo": False, "scrollZoom": True}))
        script_included = True
        parts.append("</section>")

    x = [row["step"] for row in episodes]
    rewards = [row["r"] for row in episodes]
    smooth = []
    for index in range(len(rewards)):
        window = rewards[max(0, index - 19):index + 1]
        smooth.append(sum(window) / len(window) if all(value is not None for value in window) else None)
    chart("每回合奖励", "完整回合原始奖励与最近 20 回合移动平均；前 19 回合使用已有回合。虚线：Best（金）／Final（蓝）。",
          [{"x": x, "y": rewards, "name": "原始奖励", "color": MUTED},
           {"x": x, "y": smooth, "name": "最近 20 回合移动平均", "dash": "dash"}], "奖励")
    chart("每回合伤害", "训练窗口内实际伤害；不包含独立评估的尾部结算。",
          [{"x": x, "y": [row["episode_damage"] for row in episodes], "name": "实际伤害"}], "伤害")
    chart("每回合实际出膛数", "按完整训练回合记录；并非策略请求开火次数。",
          [{"x": x, "y": [row["actual_shots"] for row in episodes], "name": "实际出膛"}], "发")
    items = [item for item in state["candidates"] if item["status"] != "duplicate"
             and item["num_timesteps"] is not None]
    series = [{"x": [item["num_timesteps"] for item in items],
               "y": [item["official_damage"] for item in items], "name": "完整评估伤害",
               "dash": "dot"}]
    for role, item, color, symbol in (("Best", best, GOLD, "diamond-open"),
                                     ("Final", final, BLUE, "circle-open")):
        if item and item["official_damage"] is not None:
            series.append({"x": [item["num_timesteps"]], "y": [item["official_damage"]],
                           "name": role, "color": color, "mode": "markers", "size": 14, "symbol": symbol})
    chart("独立评估伤害与模型位置", "每个点为一个已保存模型的一次确定性评估；同场景、同种子、同窗口。缺失点不参与选优，连线只辅助定位。",
          series, "完整窗口归属伤害", evaluation=True)
    for metric, title in (("policy_gradient_loss", "PPO 策略损失"), ("value_loss", "PPO 价值损失"),
                          ("entropy_loss", "PPO 熵损失"), ("approx_kl", "PPO 近似 KL"),
                          ("explained_variance", "PPO 解释方差")):
        chart(title, "每次完整参数更新后的日志值；缺失或非有限数值显示为空缺。",
              [{"x": [_number(row.get("time/total_timesteps")) for row in updates],
                "y": [_number(row.get("train/" + metric)) for row in updates], "name": title}], "数值")

    def cells(item):
        if item is None:
            return "<td colspan='5'>暂无有效结果</td>"
        return "".join(f"<td>{escape(str(value))}</td>" for value in (
            item["name"], _format(item["num_timesteps"]), _format(item["official_damage"]),
            _format(item["hit_rate"], rate=True), _format(item["shots"])))

    comparison = "".join(f"<tr><th>{role}</th>{cells(item)}</tr>" for role, item in (("Final 最终", final), ("Best 最佳", best)))
    ranking = []
    for item in sorted(state["candidates"], key=lambda item: (item.get("rank", math.inf), item["num_timesteps"] or 0)):
        reason = item.get("error", "")
        if item.get("duplicate_of"):
            reason = f"由 {item['duplicate_of']} 代表本步数"
        ranking.append(f"<tr><td>{item.get('rank', '—')}</td>{cells(item)}"
                       f"<td>{escape(STATUS.get(item['status'], item['status']))}</td><td>{escape(reason)}</td></tr>")
    same = best and final and best["sha256"] == final["sha256"]
    outcome = ("有效候选中的最佳" if state["status"] == "partial" else "最佳已保存模型") if best else "本次分析暂无最佳模型"
    warnings.extend([state["error"]] if state.get("error") else [])
    warnings_html = "".join(f"<p class='warning'>{escape(warning)}</p>" for warning in warnings)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练评估 · {escape(Path(state['run_dir']).name)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f9;color:#233247;font:15px/1.65 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:36px 24px}}header{{margin-bottom:28px}}h1{{font-size:30px;margin:6px 0}}
h2{{font-size:20px;margin:0 0 8px}}p{{margin:8px 0}}.eyebrow{{color:#2463a6;font-weight:650;letter-spacing:1px}}
.note,footer{{color:#617084;font-size:13px}}section{{background:white;padding:22px;margin:20px 0;border-radius:10px}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #e7ecf2;white-space:nowrap}}
td:last-child{{white-space:normal;min-width:100px}}thead{{background:#f4f6f9}}.warning{{color:#795717;background:#fff6df;padding:12px;border-radius:6px}}
.empty{{padding:35px;text-align:center;color:#617084}}code{{overflow-wrap:anywhere}}a{{color:#2463a6}}
@media(max-width:600px){{main{{padding:16px 10px}}section{{padding:14px 8px}}h1{{font-size:24px}}}}
@media print{{body{{background:white}}section{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">RM VISION RL · 训练后分析</div><h1>训练奖励与模型评估</h1>
<p>{escape(outcome)} · {escape(STATUS.get(state['status'], state['status']))}{' · Final 与 Best 为同一模型' if same else ''}</p>
<p class="note">训练目录：<code>{escape(state['run_dir'])}</code><br>分析时间：{escape(state['created_at'])}</p></header>
{warnings_html}
<section><h2>最终与最佳模型</h2><p class="note">按完整窗口归属伤害选优，同分优先累计步数较大的模型。命中率＝造成伤害的弹丸数／窗口内实际出膛数；无出膛为“—”。</p>
<div class="table-wrap"><table><thead><tr><th>角色</th><th>来源模型</th><th>累计步数</th><th>归属伤害</th><th>命中率</th><th>出膛数</th></tr></thead><tbody>{comparison}</tbody></table></div></section>
<section><h2>全部候选与排名</h2><div class="table-wrap"><table><thead><tr><th>排名</th><th>来源模型</th><th>累计步数</th><th>归属伤害</th><th>命中率</th><th>出膛数</th><th>状态</th><th>说明</th></tr></thead><tbody>{''.join(ranking)}</tbody></table></div></section>
{''.join(parts)}
<footer><p>范围：仅本次训练目录的已保存检查点；固定场景的结果不代表跨场景泛化表现。训练奖励与独立窗口结算伤害口径不同。</p>
<p>数据来源：本分析目录中的 episodes.monitor.csv、logs/progress.csv 和 evaluation/ 回放；模型、摘要及状态见 <a href="analysis.json">analysis.json</a>。
累计步数包含恢复训练的起始偏移，仅绘制完整回合。未记录的指标不补零。</p>
<p>操作：悬停查看数值，拖动框选缩放，双击复位，点击图例隐藏／显示曲线。报告内嵌 Plotly，可离线打开。</p></footer>
</main></body></html>"""
    destination = output / "report.html"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, prefix=".report-", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(document)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
