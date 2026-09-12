# -*- coding: utf-8 -*-
"""Q1 经济分析图：基准对比 + 容量/功率敏感性（各1张逻辑图）"""
import sys
import io
import os
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "utils"))
try:  # 优先用 math-modeling skill 的出版级工具链
    from setup_style import setup_style
    from export_figure import export_figure
except ImportError:  # skill 不在本机时，用项目自带降级实现
    from fig_export import setup_style, export_figure

FIG = BASE / "figures"
PREV = BASE / "figures_previews"
setup_style(journal="general", lang="zh", serif_for_zh=True)

with open(BASE / r"results\q1_analysis.json", encoding="utf-8") as f:
    A = json.load(f)

CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def save(fig, name, size):
    export_figure(fig, basename=str(FIG / name), formats=["svg", "png"],
                  size_inches=size, dpi=300, grayscale_preview=True)
    gray = FIG / f"{name}_grayscale.png"
    if gray.exists():
        gray.replace(PREV / gray.name)
    plt.close(fig)
    print("导出", name)


# ---- 基准对比：费用与购电量 ----
fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0))
c0, c1 = A["baseline_no_storage"]["cost"], A["with_storage"]["cost"]
g0, g1 = A["baseline_no_storage"]["purchase_kwh"], A["with_storage"]["purchase_kwh"]
b = axes[0].bar(["无储能", "有储能"], [c0, c1], color=[CB[3], CB[0]], width=0.55)
axes[0].set_ylabel("全天购电费 / 元")
axes[0].set_ylim(0, c0 * 1.18)
for r, v in zip(b, [c0, c1]):
    axes[0].text(r.get_x() + r.get_width() / 2, v + 600, f"{v:,.0f}", ha="center", fontsize=8)
axes[0].set_title(f"节省 {A['saving_yuan']:,.0f} 元（{A['saving_rate']*100:.1f}%）", fontsize=8.5)
b = axes[1].bar(["无储能", "有储能"], [g0 / 1000, g1 / 1000], color=[CB[3], CB[0]], width=0.55)
axes[1].set_ylabel("全天购电量 / MWh")
axes[1].set_ylim(0, g0 / 1000 * 1.18)
for r, v in zip(b, [g0 / 1000, g1 / 1000]):
    axes[1].text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v:,.1f}", ha="center", fontsize=8)
axes[1].set_title(f"购电量 {A['purchase_delta_kwh']/1000:+.2f} MWh（{A['purchase_delta_kwh']/g0*100:+.1f}%）", fontsize=8.5)
for ax in axes:
    ax.grid(alpha=0.3, linewidth=0.5, axis="y")
save(fig, "result_q1_baseline_compare", (5.8, 3.0))

# ---- 容量/功率敏感性 ----
cap = A["capacity_sensitivity"]
pow_ = A["power_sensitivity"]
fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
wx = [r["window_kwh"] / 1000 for r in cap]
wy = [r["cost"] for r in cap]
axes[0].plot(wx, wy, "o-", color=CB[0], ms=3.5, lw=1.2)
axes[0].axvline(9.6, color="0.4", lw=0.8, ls="--")
axes[0].set_xlabel("可用容量窗口 / MWh（固定下限1200 kWh）")
axes[0].set_ylabel("全天购电费 / 元")
mv = A["capacity_mv_right_diff"]["h=480"]["mv"]
axes[0].set_title(f"扩容右侧差分 ≈ {mv:.3f} 元/(kWh·日)", fontsize=8.5)
px = [r["p_kw"] / 1000 for r in pow_]
py = [r["cost"] for r in pow_]
axes[1].plot(px, py, "s-", color=CB[3], ms=3.5, lw=1.2)
axes[1].axvline(5.0, color="0.4", lw=0.8, ls="--")
axes[1].set_xlabel("充放电功率上限 / MW")
axes[1].set_title("功率翻倍仅降费 0.47%", fontsize=8.5)
for ax in axes:
    ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "result_q1_sensitivity", (6.0, 3.0))

print("全部完成")
