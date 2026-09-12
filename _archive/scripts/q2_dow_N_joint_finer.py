# -*- coding: utf-8 -*-
r"""
§38 V_dow_N 联合网格细化（只读，不改模型）
================================================================================
§37 在 (β, W=7, γ=1.0, N_DOW_HIST∈{4,6,10}) 网格下选 N=4 ⇒ V_dow_N = 1395.70。
但 β 只测了 {0.70, 0.75}、N 只测了 {4, 6, 10}。**§30 单扫描在 (β=0.75, W=7, γ=1.0)
固定下 N=10 ⇒ 1394 万元** ——比 §37 的 N=4 还低 1.7 万元。

关键：§30 是"评价期端点比较"，§37 是"训练期月末 walk-forward 选择"——两种准则。
§38 用 §37 的 walk-forward 准则，扩 (β, N) 网格寻找更优：

  β ∈ {0.65, 0.70, 0.75, 0.80}     (4)
  N ∈ {2, 3, 4, 5, 6, 8, 10}        (7)
  W = 7, γ = 1.0                    (§21/§32 已验最优)
  候选 = 4 × 7 = 28

协议与 §37 完全一致：month-end cum_j 最小准则；must/max；期末 geq2400。
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_FEB1, WARMUP_DAY, E_INIT,  # noqa: E402
                      Policy, load_data, make_plan, execute)

POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

# 网格（28 候选）
BETA_GRID = [0.65, 0.70, 0.75, 0.80]
N_DOW_HIST_GRID = [2, 3, 4, 5, 6, 8, 10]
W_LEVEL = 7
GAMMA = 1.0

V_DOW_TOTAL = 1412.0619        # §30
V_DOW_N_TOTAL = 1395.6995      # §37
V0_TOTAL = 1719.7084


# ============================ 工具函数 ============================
def _col_quantile(X, qs):
    n, m = X.shape
    vs = np.sort(X, axis=0)
    pos = np.asarray(qs, dtype=float) * (n - 1.0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    cols = np.arange(m)
    return vs[lo, cols] * (1.0 - frac) + vs[hi, cols] * frac


def dow_index(dates):
    return np.array([pd.Timestamp(d).dayofweek for d in dates])


def m_recent(d, X, W):
    hist = X[max(0, d - W):d]
    if hist.shape[0] < 7:
        if d >= 30:
            return np.median(X[max(0, d - 30):d], axis=0)
        return np.zeros(X.shape[1])
    return np.median(hist, axis=0)


def delta_dow(d, X, dow_idx, n_dow_hist, W_level):
    w_d = dow_idx[d]
    cand = [d_ for d_ in range(max(0, d - 90), d) if dow_idx[d_] == w_d]
    cand = cand[-n_dow_hist:]
    if not cand:
        return np.zeros(X.shape[1])
    deltas = np.zeros((len(cand), X.shape[1]))
    for i, d_ in enumerate(cand):
        m_ = m_recent(d_, X, W_level)
        deltas[i] = X[d_] - m_
    return deltas.mean(axis=0)


def forecast_dow_N(d, D, dow_idx, gamma, W_level, beta, n_dow_hist):
    if d < 30:
        return D["net_ref"].copy()
    m = m_recent(d, D["net"], W_level)
    dlt = delta_dow(d, D["net"], dow_idx, n_dow_hist, W_level)
    base = m + gamma * dlt
    net_hist = D["net"][max(0, d - W_level):d]
    if net_hist.shape[0] < 7:
        return D["net_ref"].copy()
    resid = net_hist - np.median(net_hist, axis=0)
    margin = _col_quantile(resid, np.full(T, beta))
    return base + margin


def replay_full_year(D, dow_idx, beta, n_dow_hist):
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_INIT)
    cum = np.zeros(nd)
    for d in range(nd):
        N_hat = forecast_dow_N(d, D, dow_idx, GAMMA, W_LEVEL, beta, n_dow_hist)
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        day_cost = float((b * price).sum()) + float(MULT * (e * price).sum())
        cum[d] = day_cost if d == 0 else cum[d - 1] + day_cost
        E = Etraj[-1]
    return cum


def main():
    t_all = time.perf_counter()
    D = load_data()
    D["_dow"] = dow_index(D["dates"])
    months_arr = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    D["_month_end"] = {m: int(np.max(np.where(months_arr == m)[0])) for m in range(1, 13)}
    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]

    print("=" * 100)
    print("§38 V_dow_N 联合网格细化（W=7, γ=1.0 固定；扩展 β、N）")
    print("=" * 100)
    print(f"  β ∈ {BETA_GRID} | N ∈ {N_DOW_HIST_GRID} | W={W_LEVEL}, γ={GAMMA}")
    ncand = len(BETA_GRID) * len(N_DOW_HIST_GRID)
    print(f"  候选数 = {ncand}")

    # ---- 0. 自检：(β=0.70, N=6) ⇒ V0 = 1719.71 ----
    print(f"\n[0] 自检 β=0.70 + N=6 ⇒ V0 ≈ 1719.71")
    cum0 = replay_full_year(D, D["_dow"], 0.70, 6)
    cV0 = float(cum0[idx[-1]] - cum0[idx[0] - 1]) / 1e4
    print(f"  β=0.70, N=6 评价期 = {cV0:.4f} 万元（应 ≈ 1719.7084）")

    # ---- 1. 28 候选回放 ----
    print(f"\n[1] 全年回放 {ncand} 个 (β, N) 候选")
    cands = []
    cum_all = {}
    k = 0
    for beta in BETA_GRID:
        for n_dow_hist in N_DOW_HIST_GRID:
            cands.append((beta, n_dow_hist))
            cum_j = replay_full_year(D, D["_dow"], beta, n_dow_hist)
            cum_all[k] = cum_j
            k += 1
        print(f"    β={beta:.2f} 完成（{time.perf_counter()-t_all:.0f}s）", flush=True)

    # ---- 2. 联合 walk-forward（与 §37 同协议：month-end cum_j）----
    print(f"\n[2] (β, N) 联合 walk-forward")
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k_star = min(range(len(cands)), key=lambda i: cum_all[i][te])
        sched[m] = cands[k_star]

    print(f"  {'月':>4}{'β*':>7}{'N*':>5}")
    for m in range(2, 13):
        b_, n_ = sched[m]
        print(f"  {m:>4}{b_:>7.2f}{n_:>5d}")

    # ---- 3. 评价期回放 ----
    print(f"\n[3] 评价期回放")
    dp = np.zeros(len(D["dates"])); de = np.zeros(len(D["dates"]))
    E = float(E_FEB1)
    for d in range(WARMUP_DAY, len(D["dates"])):
        beta, n_dow_hist = sched[months_arr[d]]
        N_hat = forecast_dow_N(d, D, D["_dow"], GAMMA, W_LEVEL, beta, n_dow_hist)
        b, c_plan = make_plan(POL, D["price"], N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, D["net"][d], D["price"], E)
        dp[d] = float((b * D["price"]).sum()); de[d] = float(MULT * (e * D["price"]).sum())
        E = Etraj[-1]
    tot38 = float((dp[idx] + de[idx]).sum()) / 1e4
    print(f"  V_dow_N_38 (β, N 扩展联合) = {tot38:.4f} 万元")
    print(f"  Δ vs V_dow_N(§37) = {tot38 - V_DOW_N_TOTAL:+.4f} 万元")
    print(f"  Δ vs V_dow        = {tot38 - V_DOW_TOTAL:+.4f} 万元")
    print(f"  Δ vs V0           = {tot38 - V0_TOTAL:+.4f} 万元")

    # ---- 4. N 分布 ----
    print(f"\n[4] 逐月 N* 分布")
    n_dist = {n: 0 for n in N_DOW_HIST_GRID}
    for m in range(2, 13):
        n_dist[sched[m][1]] += 1
    for n in N_DOW_HIST_GRID:
        print(f"  N={n:>2}: {n_dist[n]}/11")

    # ---- 5. β 分布 ----
    print(f"\n[5] 逐月 β* 分布")
    b_dist = {b: 0 for b in BETA_GRID}
    for m in range(2, 13):
        b_dist[sched[m][0]] += 1
    for b in BETA_GRID:
        print(f"  β={b:.2f}: {b_dist[b]}/11")

    # ---- 6. 与 §37 单点比较 ----
    print(f"\n[6] 与 §37 单点 (β=0.75, N=4) 评价期比较")
    cum_one = replay_full_year(D, D["_dow"], 0.75, 4)
    cV_one = float(cum_one[idx[-1]] - cum_one[idx[0] - 1]) / 1e4
    print(f"  (β=0.75, N=4) 固定全年评价期 = {cV_one:.4f} 万元")

    # ---- 7. 收口 ----
    print(f"\n[7] 与 V_dow_N 比较")
    print(f"  V_dow_N(§37, 36候选) = {V_DOW_N_TOTAL:.4f}")
    print(f"  V_dow_N_38(§38, 28候选，固定 W=7,γ=1.0) = {tot38:.4f}")
    if tot38 < V_DOW_N_TOTAL - 0.001:
        print(f"  ⇒ 网格细化可再降费 = {V_DOW_N_TOTAL - tot38:.4f} 万元")
    elif tot38 > V_DOW_N_TOTAL + 0.001:
        print(f"  ⇒ 网格细化反而恶化 = {tot38 - V_DOW_N_TOTAL:+.4f} 万元（§37 N=4 已内点最优）")
    else:
        print(f"  ⇒ §37 N=4 已内点最优")

    # ---- 落盘 ----
    out = dict(
        口径=dict(beta_grid=BETA_GRID, N_DOW_HIST_grid=N_DOW_HIST_GRID,
                  W=W_LEVEL, gamma=GAMMA, 候选数=len(cands)),
        自检_V0=cV0,
        V_dow_N_38=tot38,
        delta_vs_V_dow_N37=tot38 - V_DOW_N_TOTAL,
        delta_vs_V_dow=tot38 - V_DOW_TOTAL,
        联合日程={str(m): {"beta": float(sched[m][0]), "N_DOW_HIST": int(sched[m][1])}
                  for m in range(2, 13)},
        N分布=dict(n_dist),
        β分布=dict(b_dist),
        对比=dict(V0=V0_TOTAL, V_dow=V_DOW_TOTAL,
                  V_dow_N=V_DOW_N_TOTAL, V_dow_N_38=tot38,
                  V_single_β_75_N_4=cV_one),
    )
    (BASE / "results" / "q2_dow_N_joint_finer.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_dow_N_joint_finer.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
