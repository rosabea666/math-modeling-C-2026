# -*- coding: utf-8 -*-
"""Q2 三类图（raw/process/result 各3张），出版级导出 SVG+PNG300dpi+灰度预览"""
import sys
import io
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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
FIG.mkdir(exist_ok=True)
PREV.mkdir(exist_ok=True)
setup_style(journal="general", lang="zh", serif_for_zh=True)

DT = 1.0 / 6.0
TOL = 1e-6
z = np.load(BASE / r"results\q2_solution.npz")
B, C, D, EM, W = z["B"], z["C"], z["D"], z["EM"], z["W"]
SOC0, SOC1 = z["SOC0"], z["SOC1"]
price, load_fc, pv_fc = z["price"], z["load_fc"], z["pv_fc"]
load_act, pv_act = z["load_act"], z["pv_act"]
dates = pd.to_datetime(z["dates"])
eval_mask = z["eval_mask"]
plan_cost_day, emerg_cost_day = z["plan_cost_day"], z["emerg_cost_day"]

CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
months = pd.Series(dates.month)


def save(fig, name, size):
    export_figure(fig, basename=str(FIG / name), formats=["svg", "png"],
                  size_inches=size, dpi=300, grayscale_preview=True)
    gray = FIG / f"{name}_grayscale.png"
    if gray.exists():
        gray.replace(PREV / gray.name)
    plt.close(fig)
    print("导出", name)


# ---------- raw 1: 全年实测负载/光伏日能量（折线） ----------
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(dates, load_act.sum(axis=1) / 1000, color=CB[0], lw=0.9, label="小区日用电量")
ax.plot(dates, pv_act.sum(axis=1) / 1000, color=CB[1], lw=0.9, label="光伏日发电量")
ax.axhline(load_fc.sum() / 1000, color=CB[0], lw=0.9, ls="--", alpha=0.6, label="代表日用电(附件1)")
ax.axhline(pv_fc.sum() / 1000, color=CB[1], lw=0.9, ls="--", alpha=0.6, label="代表日光伏预测(附件1)")
ax.set_ylabel("日能量 / MWh")
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "raw_q2_year_energy", (6.0, 3.2))

# ---------- raw 2: 实际净负载分位带 vs 代表日预测 ----------
net_act = (load_act - pv_act) / DT          # kW
net_fc = (load_fc - pv_fc) / DT
t = (np.arange(144) + 1) * DT
q = np.percentile(net_act, [10, 50, 90], axis=0)
fig, ax = plt.subplots(figsize=(5.8, 3.2))
ax.fill_between(t, q[0], q[2], color=CB[0], alpha=0.2, label="实际净负载10-90分位")
ax.plot(t, q[1], color=CB[0], lw=1.2, label="实际净负载中位数")
ax.plot(t, net_fc, color=CB[3], lw=1.2, ls="--", label="代表日净负载(计划依据)")
ax.set_xlabel("时刻")
ax.set_ylabel("净负载 / kW")
ax.set_xticks(range(0, 25, 4))
ax.set_xlim(0, 24)
ax.legend(frameon=False, fontsize=7.5)
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "raw_q2_netload_band", (5.8, 3.2))

# ---------- raw 3: 预测残差直方图（晚高峰区间） ----------
resid = (net_act - net_fc)                  # kW, 正=实际超预测
peak = resid[:, 108:126].flatten()          # 18:00-21:00
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(peak, bins=60, color=CB[4], alpha=0.8)
ax.axvline(0, color="0.3", lw=0.9, ls="--")
ax.axvline(np.percentile(peak, 90), color=CB[3], lw=1.0, ls="-.",
           label=f"90分位 {np.percentile(peak, 90):.0f} kW")
ax.set_xlabel("晚高峰净负载预测残差 / kW")
ax.set_ylabel("频数 / 个")
ax.legend(frameon=False, fontsize=7.5)
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "raw_q2_residual_hist", (5.2, 3.2))

# ---------- process 1: 全年 SOC(日初/日末) 轨迹 ----------
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(dates, SOC0 / 1000, color=CB[2], lw=0.8, label="日初储电量")
ax.plot(dates, SOC1 / 1000, color=CB[3], lw=0.8, label="日末储电量")
ax.axhline(10.8, color="0.4", lw=0.8, ls="--")
ax.axhline(1.2, color="0.4", lw=0.8, ls="--")
ax.set_ylabel("储电量 / MWh")
ax.legend(frameon=False, fontsize=7.5, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "process_q2_soc_year", (6.0, 3.0))

# ---------- process 2: 日费用时间序列（计划+紧急堆叠面积） ----------
fig, ax = plt.subplots(figsize=(6.0, 3.2))
de = dates[eval_mask]
ax.stackplot(de, plan_cost_day[eval_mask] / 1e4, emerg_cost_day[eval_mask] / 1e4,
             labels=["计划购电费用", "紧急购电费用"], colors=[CB[0], CB[3]], alpha=0.85)
ax.set_ylabel("日费用 / 万元")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
ax.grid(alpha=0.3, linewidth=0.5)
save(fig, "process_q2_daily_cost", (6.0, 3.2))

# ---------- process 3: 紧急购电 日×时段 热力图 ----------
fig, ax = plt.subplots(figsize=(6.0, 3.6))
em_eval = EM[eval_mask].T / DT              # kW 便于色阶
im = ax.imshow(em_eval, aspect="auto", origin="lower", cmap="magma",
               extent=[0, em_eval.shape[1], 0, 24])
ax.set_yticks(range(0, 25, 4))
ax.set_ylabel("时刻")
xt = np.linspace(0, em_eval.shape[1] - 1, 11)
ax.set_xticks(xt)
ax.set_xticklabels([de[int(i)].strftime("%m月") for i in xt])
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("紧急购电功率 / kW", fontsize=8)
save(fig, "process_q2_emerg_heatmap", (6.0, 3.6))

# ---------- result 1: 指定4日执行结果 2x2 ----------
spec = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True)
for k, ds in enumerate(spec):
    ax = axes[k // 2][k % 2]
    day = int(np.where(dates == pd.Timestamp(ds))[0][0])
    ax.plot(t, load_act[day] / DT / 1000, color="black", lw=0.9, label="实际负载")
    ax.plot(t, (B[day] + pv_act[day]) / DT / 1000, color=CB[0], lw=0.9, label="计划购电+光伏")
    ax.fill_between(t, 0, EM[day] / DT / 1000, color=CB[3], alpha=0.6, label="紧急购电")
    ax.set_title(ds, fontsize=8)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.grid(alpha=0.3, linewidth=0.5)
axes[0][0].legend(frameon=False, fontsize=6.5, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel("时刻")
for ax in axes[:, 0]:
    ax.set_ylabel("功率 / MW")
save(fig, "result_q2_spec_days", (6.3, 4.6))

# ---------- result 2: 月度费用分解分组柱状 ----------
fig, ax = plt.subplots(figsize=(6.0, 3.2))
mm = months[eval_mask]
pc = pd.Series(plan_cost_day[eval_mask] / 1e4).groupby(mm).sum()
ec = pd.Series(emerg_cost_day[eval_mask] / 1e4).groupby(mm).sum()
x = np.arange(len(pc))
ax.bar(x - 0.19, pc.values, width=0.38, color=CB[0], label="计划购电费用")
ax.bar(x + 0.19, ec.values, width=0.38, color=CB[3], label="紧急购电费用")
ax.set_xticks(x)
ax.set_xticklabels([f"{m}月" for m in pc.index], fontsize=7.5)
ax.set_ylabel("月费用 / 万元")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5, axis="y")
save(fig, "result_q2_month_cost", (6.0, 3.2))

# ---------- result 3: 紧急事件日电量分布 + 月度事件日数 ----------
fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0),
                         gridspec_kw={"width_ratios": [3, 2]})
em_day = EM[eval_mask].sum(axis=1)
axes[0].hist(em_day[em_day > TOL] / 1000, bins=40, color=CB[4])
axes[0].set_xlabel("紧急事件日总电量 / MWh")
axes[0].set_ylabel("日数 / 天")
axes[0].grid(alpha=0.3, linewidth=0.5)
ev_cnt = pd.Series((em_day > TOL).astype(int)).groupby(mm).sum()
axes[1].bar([f"{m}月" for m in ev_cnt.index], ev_cnt.values, color=CB[3])
axes[1].set_ylabel("紧急事件日数 / 天")
axes[1].tick_params(axis="x", labelsize=6.5, rotation=45)
axes[1].grid(alpha=0.3, linewidth=0.5, axis="y")
save(fig, "result_q2_emerg_stats", (6.0, 3.0))

print("全部完成")
