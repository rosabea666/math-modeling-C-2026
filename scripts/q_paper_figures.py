# -*- coding: utf-8 -*-
"""生成竞赛论文用图（中文、PDF 矢量），输出到 CUMCM2026-Template/figures/。

数据来源（均为已落盘的最终结果，不改动任何模型逻辑）：
  Q1: results/q1_solution.npz, results/q1_analysis.json
  Q2: results/q2_V_dow_N_profile.json + 主方案 E 一次性回放（仅内存，不写 Excel）
"""
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "CUMCM2026-Template" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
})
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
DT = 1.0 / 6.0


def save(fig, name):
    path = FIG / (name + ".pdf")
    fig.savefig(path)
    plt.close(fig)
    print("导出", path.name)


# ================= Q1 =================
z = np.load(BASE / "results" / "q1_solution.npz")
g, c1, d1, w1, E1 = z["g"], z["c"], z["d"], z["w"], z["E"]
price, load_kw, pv_kw = z["price"], z["load_kw"], z["pv_kw"]
t = (np.arange(144) + 1) * DT
tE = np.arange(145) * DT

# 图 1：代表日输入特征
fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.4), sharex=True)
ax = axes[0]
ax.plot(t, load_kw, color=CB[0], lw=1.2, label="小区负载")
ax.plot(t, pv_kw, color=CB[1], lw=1.2, label="光伏预测功率")
ax.set_ylabel("功率 (kW)")
ax.legend(ncol=2, loc="upper left")
ax = axes[1]
ax.step(t, price, where="post", color=CB[3], lw=1.2)
ax.set_ylabel("电价 (元/kWh)")
ax.set_xlabel("时刻 (h)")
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 4))
fig.tight_layout()
save(fig, "fig_q1_input")

# 图 2：最优购电与储能运行策略
fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.8), sharex=True)
ax = axes[0]
ax.bar(t, g, width=DT * 0.9, color=CB[0], label="购电量")
ax.bar(t, c1, width=DT * 0.9, color=CB[2], label="充电量")
ax.bar(t, -d1, width=DT * 0.9, color=CB[3], label="放电量")
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("区间电量 (kWh)")
ax.legend(ncol=3, loc="upper left")
ax = axes[1]
ax.plot(tE, E1, color=CB[4], lw=1.4)
ax.axhline(10800, color="gray", ls="--", lw=0.8)
ax.axhline(1200, color="gray", ls="--", lw=0.8)
ax.text(24.1, 10800, "上限", va="center", fontsize=7, color="gray")
ax.text(24.1, 1200, "下限", va="center", fontsize=7, color="gray")
ax.set_ylabel("储电量 (kWh)")
ax.set_xlabel("时刻 (h)")
ax.set_xlim(0, 24)
ax.set_ylim(0, 11500)
ax.set_xticks(range(0, 25, 4))
fig.tight_layout()
save(fig, "fig_q1_strategy")

# 图 3：容量—功率敏感性
ana = json.load(open(BASE / "results" / "q1_analysis.json", encoding="utf-8"))
cap = ana["capacity_sensitivity"]
pw = ana["power_sensitivity"]
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
ax = axes[0]
ax.plot([x["window_kwh"] for x in cap], [x["cost"] for x in cap],
        "o-", color=CB[0], lw=1.2, ms=3.5)
ax.set_xlabel("可用储电窗口 (kWh)")
ax.set_ylabel("全天购电费 (元)")
ax.set_title("(a) 抬升储能上限（下限固定 1200）")
ax = axes[1]
ax.plot([x["p_kw"] for x in pw], [x["cost"] for x in pw],
        "s-", color=CB[1], lw=1.2, ms=3.5)
ax.axvline(5000, color="gray", ls="--", lw=0.8)
ax.text(5000, ax.get_ylim()[0], " 当前 5000 kW", fontsize=7, color="gray")
ax.set_xlabel("充放电功率上限 (kW)")
ax.set_title("(b) 功率上限扫描")
fig.tight_layout()
save(fig, "fig_q1_sensitivity")

# ================= Q2 =================
prof = json.load(open(BASE / "results" / "q2_V_dow_N_profile.json", encoding="utf-8"))
mon = prof["月度"]
ms = sorted(mon.keys(), key=int)
plan_v = [mon[m]["plan"] for m in ms]
emer_v = [mon[m]["emerg"] for m in ms]

# 图 4：月度费用构成
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.bar(ms, plan_v, color=CB[0], label="计划费用")
ax.bar(ms, emer_v, bottom=plan_v, color=CB[3], label="紧急费用")
ax.set_xlabel("月份（2025 年）")
ax.set_ylabel("费用 (万元)")
ax.legend(ncol=2)
fig.tight_layout()
save(fig, "fig_q2_month")

# 图 5：策略对照（数值来自正文表：对照策略与主方案）
labels = ["A 固定代表日", "B 代表日+校准", "B48 跨日窗口", "C 分位裕量",
          "D 执行优化", "E 主方案", "完美信息参照"]
vals = [2825.28, 2035.24, 2024.98, 1775.91, 1739.64, 1395.70, 1229.10]
colors = [CB[5]] * 5 + [CB[0], "gray"]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
ypos = np.arange(len(labels))[::-1]
ax.barh(ypos, vals, color=colors, height=0.62)
for y, v in zip(ypos, vals):
    ax.text(v + 25, y, f"{v:.2f}", va="center", fontsize=8)
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
ax.set_xlabel("评价期总费用 (万元)")
ax.set_xlim(0, 3150)
fig.tight_layout()
save(fig, "fig_q2_ladder")

# 图 6：全年日初 SOC 与日费用（主方案 E 一次性回放，仅内存）
sys.path.insert(0, str(BASE / "scripts"))
from q2_model import (T, MULT, E_FEB1, WARMUP_DAY,  # noqa: E402
                      load_data, make_plan, execute, Policy)


def _col_quantile(X, qs):
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        out[j] = np.quantile(X[:, j], qs[j])
    return out


def dow_index(dates):
    return np.array([pd.Timestamp(dd).dayofweek for dd in dates])


def m_recent(dd, X, W):
    hist = X[max(0, dd - W):dd]
    if hist.shape[0] < 7:
        if dd >= 30:
            return np.median(X[max(0, dd - 30):dd], axis=0)
        return np.zeros(X.shape[1])
    return np.median(hist, axis=0)


def delta_dow(dd, X, dow_idx, n_dow_hist, W_level):
    w_d = dow_idx[dd]
    cand = [s for s in range(max(0, dd - 90), dd) if dow_idx[s] == w_d]
    cand = cand[-n_dow_hist:]
    if not cand:
        return np.zeros(X.shape[1])
    deltas = np.zeros((len(cand), X.shape[1]))
    for i, s in enumerate(cand):
        deltas[i] = X[s] - m_recent(s, X, W_level)
    return deltas.mean(axis=0)


def forecast_dow_N(dd, D, dow_idx, gamma, W_level, beta, n_dow_hist):
    if dd < 30:
        return D["net_ref"].copy()
    m = m_recent(dd, D["net"], W_level)
    dlt = delta_dow(dd, D["net"], dow_idx, n_dow_hist, W_level)
    base = m + gamma * dlt
    net_hist = D["net"][max(0, dd - W_level):dd]
    if net_hist.shape[0] < 7:
        return D["net_ref"].copy()
    resid = net_hist - np.median(net_hist, axis=0)
    margin = _col_quantile(resid, np.full(T, beta))
    return base + margin


SCHED = {m: (0.70, 7, 1.0, 4) for m in (2, 3)}
for m in range(4, 13):
    SCHED[m] = (0.75, 7, 1.0, 4)

D = load_data()
dow_idx = dow_index(D["dates"])
months = np.array([dd.astype("datetime64[M]").astype(int) % 12 + 1 for dd in D["dates"]])
price_y, net = D["price"], D["net"]
nd = len(D["dates"])
POL = Policy("E", forecast="cquant", W=7, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

SOC0 = np.zeros(nd)
day_cost = np.zeros(nd)
day_emerg = np.zeros(nd)
Ecur = float(E_FEB1)
for dd in range(WARMUP_DAY, nd):
    beta, W_level, gamma, n_dow = SCHED[months[dd]]
    N_hat = forecast_dow_N(dd, D, dow_idx, gamma, W_level, beta, n_dow)
    b, c_plan = make_plan(POL, price_y, N_hat, Ecur)
    cc, ddc, ee, ww, Etraj = execute(POL, b, c_plan, N_hat, net[dd], price_y, Ecur)
    SOC0[dd] = Etraj[0]
    day_cost[dd] = float((b * price_y).sum()) + MULT * float((ee * price_y).sum())
    day_emerg[dd] = float((ee * price_y).sum())
    Ecur = Etraj[-1]

ev = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]
dts = pd.to_datetime(D["dates"][ev])

fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.8), sharex=True)
ax = axes[0]
ax.plot(dts, SOC0[ev] / 1e4, color=CB[4], lw=0.9)
ax.axhline(1.08, color="gray", ls="--", lw=0.8)
ax.axhline(0.12, color="gray", ls="--", lw=0.8)
ax.set_ylabel("日初储电量 (万kWh)")
ax = axes[1]
ax.bar(dts, day_cost[ev] / 1e4, width=1.0, color=CB[0], label="日总费用")
ax.bar(dts, day_emerg[ev] / 1e4, width=1.0, color=CB[3], label="其中紧急费用")
ax.set_ylabel("日费用 (万元)")
ax.set_xlabel("日期（2025 年）")
ax.legend(ncol=2, loc="upper left")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
fig.tight_layout()
save(fig, "fig_q2_year")

print("全部论文图已生成 ->", FIG)
