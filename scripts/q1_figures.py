# -*- coding: utf-8 -*-
"""Q1 三类图（raw/process/result 各3张），出版级导出 SVG+PNG300dpi+灰度预览"""
import sys
import io
import os
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
FIG.mkdir(exist_ok=True)
PREV.mkdir(exist_ok=True)

setup_style(journal="general", lang="zh", serif_for_zh=True)

DT = 1.0 / 6.0
data = np.load(BASE / r"results\q1_solution.npz")
g, c, d, w, E = data["g"], data["c"], data["d"], data["w"], data["E"]
price, load_kw, pv_kw = data["price"], data["load_kw"], data["pv_kw"]
cost = float(data["cost"])

t = (np.arange(144) + 1) * DT        # 区间末端时刻 h（1/6..24）
tE = np.arange(145) * DT             # SOC 时刻 0..24
hours = np.arange(0, 25, 4)

CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def style_ax(ax):
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h}:00" for h in hours])
    ax.set_xlim(0, 24)
    ax.grid(alpha=0.3, linewidth=0.5)


def save(fig, name, size):
    export_figure(fig, basename=str(FIG / name), formats=["svg", "png"],
                  size_inches=size, dpi=300, grayscale_preview=True)
    gray = FIG / f"{name}_grayscale.png"
    if gray.exists():
        gray.replace(PREV / gray.name)  # 灰度预览归入 figures_previews，保持 figures 目录审计纯净
    plt.close(fig)
    print("导出", name)


# ---------- raw 1: 日内电价/负载/光伏（上下共享x，避免双Y轴） ----------
fig, axes = plt.subplots(2, 1, figsize=(5.8, 4.4), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2]})
axes[0].plot(t, load_kw, color=CB[0], lw=1.2, label="小区负载")
axes[0].plot(t, pv_kw, color=CB[1], lw=1.2, label="光伏预测功率")
axes[0].set_ylabel("功率 / kW")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
style_ax(axes[0])
axes[1].step(t, price, color=CB[3], lw=1.2, where="post")
axes[1].set_ylabel("电价 / (元·kWh$^{-1}$)")
axes[1].set_xlabel("时刻")
style_ax(axes[1])
fig.align_ylabels(axes)
save(fig, "raw_q1_price_load_pv", (5.8, 4.4))

# ---------- raw 2: 负载与光伏功率分布直方图 ----------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(load_kw, bins=24, alpha=0.75, color=CB[0], label="小区负载")
ax.hist(pv_kw, bins=24, alpha=0.65, color=CB[1], label="光伏预测功率")
ax.set_xlabel("功率 / kW")
ax.set_ylabel("区间数 / 个")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "raw_q1_load_pv_hist", (5.2, 3.2))

# ---------- raw 3: 净负载持续曲线 + 电价排序散点 ----------
net = load_kw - pv_kw
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(np.linspace(0, 100, 144), np.sort(net)[::-1], color=CB[2], lw=1.2)
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.set_xlabel("累计区间比例 / %")
ax.set_ylabel("净负载 / kW")
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "raw_q1_netload_duration", (5.2, 3.2))

# ---------- process 1: SOC 轨迹 + 充放电功率 ----------
fig, axes = plt.subplots(2, 1, figsize=(5.8, 4.4), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2]})
axes[0].plot(tE, E, color=CB[2], lw=1.2)
axes[0].axhline(10800, color=CB[3], lw=0.8, ls="--")
axes[0].axhline(1200, color=CB[3], lw=0.8, ls="--")
axes[0].axhline(6000, color="0.4", lw=0.8, ls=":")
axes[0].set_ylabel("储能电量 / kWh")
style_ax(axes[0])
axes[1].bar(t - DT / 2, c / DT, width=DT, color=CB[0], label="充电功率")
axes[1].bar(t - DT / 2, -d / DT, width=DT, color=CB[3], label="放电功率")
axes[1].set_ylabel("功率 / kW")
axes[1].set_xlabel("时刻")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
style_ax(axes[1])
fig.align_ylabels(axes)
save(fig, "process_q1_soc_trajectory", (5.8, 4.4))

# ---------- process 2: 充放电量10分钟分布（散点，按时段着色） ----------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
seg = np.digitize(t, [4, 8, 12, 16, 20])  # 0..5 -> 6个时段
labels = ["0-4时", "4-8时", "8-12时", "12-16时", "16-20时", "20-24时"]
for s in range(6):
    m = seg == s
    ax.scatter(c[m] - d[m], price[m], s=12, color=CB[s], label=labels[s], alpha=0.85)
ax.axvline(0, color="0.4", lw=0.8, ls="--")
ax.set_xlabel("净充电量(充−放) / kWh")
ax.set_ylabel("电价 / (元·kWh$^{-1}$)")
ax.legend(frameon=False, fontsize=6.5, ncol=2)
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "process_q1_cd_vs_price", (5.2, 3.2))

# ---------- process 3: 供能构成堆积面积 ----------
fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.stackplot(t, g / DT, pv_kw, d / DT,
             labels=["计划购电", "光伏", "储能放电"],
             colors=[CB[0], CB[1], CB[3]], alpha=0.85)
ax.plot(t, load_kw, color="black", lw=1.0, label="负载")
ax.set_ylabel("功率 / kW")
ax.set_xlabel("时刻")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
style_ax(ax)
save(fig, "process_q1_supply_stack", (5.8, 3.4))

# ---------- result 1: 计划购电策略（柱状+电价背景） ----------
fig, ax1 = plt.subplots(figsize=(5.8, 3.4))
ax1.bar(t - DT / 2, g, width=DT, color=CB[0], label="计划购电量")
ax1.set_ylabel("购电量 / kWh")
ax1.set_xlabel("时刻")
ax2 = ax1.twinx()
ax2.step(t, price, color=CB[3], lw=1.0, where="post", alpha=0.7)
ax2.set_ylabel("电价 / (元·kWh$^{-1}$)", color=CB[3])
ax2.tick_params(axis="y", colors=CB[3])
ax2.grid(visible=False)
style_ax(ax1)
save(fig, "result_q1_purchase_plan", (5.8, 3.4))

# ---------- result 2: 4小时块充放电分组柱状（表2） ----------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
blocks = [f"{b*4}-{(b+1)*4}时" for b in range(6)]
cs = [c[b*24:(b+1)*24].sum() for b in range(6)]
ds = [d[b*24:(b+1)*24].sum() for b in range(6)]
x = np.arange(6)
ax.bar(x - 0.19, cs, width=0.38, color=CB[0], label="充电量")
ax.bar(x + 0.19, ds, width=0.38, color=CB[3], label="放电量")
ax.set_xticks(x)
ax.set_xticklabels(blocks, fontsize=7.5)
ax.set_ylabel("电量 / kWh")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5, axis="y")
save(fig, "result_q1_block_bar", (5.2, 3.2))

# ---------- result 3: 累计购电费用曲线 + 分时段费用柱状 ----------
fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0),
                         gridspec_kw={"width_ratios": [3, 2]})
axes[0].plot(t, np.cumsum(price * g), color=CB[4], lw=1.2)
axes[0].set_ylabel("累计购电费 / 元")
axes[0].set_xlabel("时刻")
style_ax(axes[0])
seg_cost = [float((price * g)[b*24:(b+1)*24].sum()) for b in range(6)]
axes[1].bar(blocks, seg_cost, color=CB[4])
axes[1].set_ylabel("购电费 / 元")
axes[1].tick_params(axis="x", labelsize=6.5, rotation=30)
axes[1].grid(alpha=0.3, linewidth=0.5, axis="y")
fig.suptitle(f"全天购电费 {cost:.2f} 元", fontsize=9)
save(fig, "result_q1_cost", (5.8, 3.0))

print("全部完成")
