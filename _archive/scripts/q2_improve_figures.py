# -*- coding: utf-8 -*-
"""Q2 策略对比图（result 类，3张）"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SKILL_ROOT = r"C:\Users\fzz17\.workbuddy\skills\math-modeling"
sys.path.insert(0, os.path.join(SKILL_ROOT, "tools", "figure", "scripts"))
from setup_style import setup_style
from export_figure import export_figure

BASE = Path(r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23")
FIG = BASE / "figures"
PREV = BASE / "figures_previews"
setup_style(journal="general", lang="zh", serif_for_zh=True)
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

with open(BASE / r"results\q2_strategies.json", encoding="utf-8") as f:
    S = json.load(f)
T = S["table"]
names = ["A", "B", "B48", "C", "D", "PI"]
labels = ["A 基线", "B 校准", "B48 跨日", "C 风险预留", "D 执行优化", "PI 下界"]

# 载入逐日序列（月度费用曲线用）
import openpyxl
def load_daily(name):
    z = np.load(BASE / rf"results\q2_strategy_{name}.npz")
    return z
sys.path.insert(0, str(BASE / "scripts"))
from q2_improve import load_inputs, EMERG_MULT
price, net_fc, dates, load_act, pv_act, net_act = load_inputs()
eval_idx = np.where(dates >= pd.Timestamp("2025-02-01"))[0]
de = pd.to_datetime(pd.Series(dates[eval_idx]))
mm = de.dt.month

def save(fig, name, size):
    export_figure(fig, basename=str(FIG / name), formats=["svg", "png"],
                  size_inches=size, dpi=300, grayscale_preview=True)
    gray = FIG / f"{name}_grayscale.png"
    if gray.exists():
        gray.replace(PREV / gray.name)
    plt.close(fig)
    print("导出", name)

# ---- 图1: 策略阶梯（堆叠柱: 计划+紧急, 总费用标注） ----
fig, ax = plt.subplots(figsize=(6.0, 3.4))
x = np.arange(len(names))
pc = [T[n]["计划费用(万元)"] for n in names]
ec = [T[n]["紧急费用(万元)"] for n in names]
ax.bar(x, pc, width=0.55, color=CB[0], label="计划购电费用")
ax.bar(x, ec, width=0.55, bottom=pc, color=CB[3], label="紧急购电费用")
for i, n in enumerate(names):
    ax.text(i, pc[i] + ec[i] + 40, f"{pc[i]+ec[i]:.0f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("评价期总费用 / 万元")
ax.set_ylim(0, 3100)
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5, axis="y")
save(fig, "result_q2_strategy_ladder", (6.0, 3.4))

# ---- 图2: 月度总费用对比（A/C/D/PI 折线） ----
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for name, color, ls in (("A", CB[3], "-"), ("C", CB[1], "-"), ("D", CB[0], "-"), ("PI", CB[2], "--")):
    z = load_daily(name)
    tot = (z["B"][eval_idx] * price).sum(axis=1) + EMERG_MULT * (z["EM"][eval_idx] * price).sum(axis=1)
    ms = pd.Series(tot / 1e4).groupby(mm.values).sum()
    ax.plot([f"{m}月" for m in ms.index], ms.values, ls, color=color, lw=1.2,
            marker="o", ms=3, label={"A": "A 基线", "C": "C 风险预留", "D": "D 执行优化", "PI": "PI 下界"}[name])
ax.set_ylabel("月总费用 / 万元")
ax.legend(frameon=False, fontsize=7.5, ncol=2)
ax.tick_params(axis="x", labelsize=7)
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "result_q2_monthly_compare", (6.0, 3.2))

# ---- 图3: β 预热期选择曲线 + C/D策略紧急费用结构 ----
fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), gridspec_kw={"width_ratios": [1, 1]})
bs = S["beta_scores"]
bx = sorted(float(k) for k in bs)
by = [bs[str(b) if str(b) in bs else b] for b in bx]
# json 键为字符串形式
by = [bs[k] for k in sorted(bs, key=lambda s: float(s))]
bx = sorted(float(k) for k in bs)
axes[0].plot(bx, by, "o-", color=CB[0], lw=1.2, ms=4)
axes[0].axvline(S["beta_star"], color=CB[3], ls="--", lw=0.9)
axes[0].set_xlabel("风险预留分位数 β")
axes[0].set_ylabel("预热期(1月)总费用 / 万元")
axes[0].set_title(f"β* = {S['beta_star']}（预热期选定后冻结）", fontsize=8.5)
axes[0].grid(alpha=0.3, linewidth=0.5)
em_kwh = [T[n]["紧急电量(kWh)"] / 1e4 for n in ["A", "B", "C", "D"]]
em_cost = [T[n]["紧急费用(万元)"] for n in ["A", "B", "C", "D"]]
x2 = np.arange(4)
axes[1].bar(x2 - 0.19, em_cost, width=0.38, color=CB[3], label="紧急费用/万元")
axes[1].bar(x2 + 0.19, em_kwh, width=0.38, color=CB[4], label="紧急电量/万kWh")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(["A", "B", "C", "D"], fontsize=8)
axes[1].set_title("D: 紧急电量↑但费用↓", fontsize=8.5)
axes[1].legend(frameon=False, fontsize=7)
axes[1].grid(alpha=0.3, linewidth=0.5, axis="y")
save(fig, "result_q2_beta_and_D", (6.2, 3.0))

print("全部完成")
