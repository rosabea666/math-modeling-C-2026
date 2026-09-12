# -*- coding: utf-8 -*-
r"""
§37 N_DOW_HIST 联合 walk-forward（只读，不改模型）
================================================================================
§30 已验 N_DOW_HIST 单变量（N=1→1448, 2→1417, 4→1400, 6→1399, 10→1394，N=6 单调饱和）。
但 §30 是固定 γ=1.0, W=7, β=0.75。**当 γ/W/β 都随月变时，N_DOW_HIST 是否仍单调饱和？**

§37 测试 (β, W, γ, N_DOW_HIST) 四参数联合 walk-forward：
  β ∈ {0.70, 0.75}          (2)
  W ∈ {7, 14}                (2)
  γ ∈ {0, 0.5, 1.0}          (3)
  N_DOW_HIST ∈ {4, 6, 10}    (3)
  候选 = 2×2×3×3 = 36
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

# 网格（36 候选）
BETA_GRID = [0.70, 0.75]
W_GRID = [7, 14]
GAMMA_GRID = [0.0, 0.5, 1.0]
N_DOW_HIST_GRID = [4, 6, 10]

V_DOW_TOTAL = 1412.0619
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


def replay_full_year(D, dow_idx, params_fn):
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_INIT)
    cum = np.zeros(nd)
    for d in range(nd):
        W_level, beta, gamma, n_dow_hist = params_fn(d)
        N_hat = forecast_dow_N(d, D, dow_idx, gamma, W_level, beta, n_dow_hist)
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

    print("=" * 102)
    print("§37 N_DOW_HIST 联合 walk-forward（与 β/W/γ 同协议）")
    print("=" * 102)
    print(f"  β ∈ {BETA_GRID} | W ∈ {W_GRID} | γ ∈ {GAMMA_GRID} | N ∈ {N_DOW_HIST_GRID}")
    ncand = len(BETA_GRID) * len(W_GRID) * len(GAMMA_GRID) * len(N_DOW_HIST_GRID)
    print(f"  候选数 = {ncand}")

    # ---- 0. 自检：γ=0 + N=6 + β=0.70 + W=14 ⇒ V0 ----
    print(f"\n[0] 自检 γ=0 + N=6 + β=0.70 + W=14 + cycle ⇒ V0 ≈ 1719.71")
    cum0 = replay_full_year(D, D["_dow"], lambda d, W=14, beta=0.70: (W, beta, 0.0, 6))
    cV0 = float(cum0[idx[-1]] - cum0[idx[0] - 1]) / 1e4
    print(f"  γ=0, N=6, β=0.70, W=14 评价期 = {cV0:.4f} 万元（应 ≈ 1719.7084）")

    # ---- 1. 候选回放 ----
    print(f"\n[1] 全年回放 {ncand} 个候选")
    cands = []
    cum_all = {}
    k = 0
    for beta in BETA_GRID:
        for W in W_GRID:
            for gamma in GAMMA_GRID:
                for N in N_DOW_HIST_GRID:
                    cands.append((beta, W, gamma, N))
                    cum_j = replay_full_year(
                        D, D["_dow"],
                        lambda d, beta=beta, W=W, gamma=gamma, N=N: (W, beta, gamma, N))
                    cum_all[k] = cum_j
                    k += 1
        print(f"    β={beta:.2f} 完成（{time.perf_counter()-t_all:.0f}s）", flush=True)

    # ---- 2. 联合 walk-forward ----
    print(f"\n[2] (β, W, γ, N) 联合 walk-forward")
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k_star = min(range(len(cands)), key=lambda i: cum_all[i][te])
        sched[m] = cands[k_star]
    print(f"  {'月':>4}{'β*':>6}{'W*':>5}{'γ*':>5}{'N*':>4}")
    for m in range(2, 13):
        b_, w_, g_, n_ = sched[m]
        print(f"  {m:>4}{b_:>6.2f}{w_:>5d}{g_:>5.2f}{n_:>4d}")

    # ---- 3. 评价期回放 ----
    print(f"\n[3] 评价期回放")
    cum_v = cum_all[0].copy()  # 借用长度
    cum_v_eval = np.zeros(idx[-1] - idx[0] + 1)
    E = float(E_FEB1)
    months = months_arr
    dp = np.zeros(len(D["dates"])); de = np.zeros(len(D["dates"]))
    for d in range(WARMUP_DAY, len(D["dates"])):
        beta, W_level, gamma, n_dow_hist = sched[months[d]]
        N_hat = forecast_dow_N(d, D, D["_dow"], gamma, W_level, beta, n_dow_hist)
        b, c_plan = make_plan(POL, D["price"], N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, D["net"][d], D["price"], E)
        dp[d] = float((b * D["price"]).sum()); de[d] = float(MULT * (e * D["price"]).sum())
        E = Etraj[-1]
    tot37 = float((dp[idx] + de[idx]).sum()) / 1e4
    print(f"  V_dow_N (β, W, γ, N 联合日程) = {tot37:.4f} 万元")
    print(f"  Δ vs V_dow = {tot37 - V_DOW_TOTAL:+.4f} 万元")
    print(f"  Δ vs V0    = {tot37 - V0_TOTAL:+.4f} 万元")

    # ---- 4. N 分布 ----
    print(f"\n[4] 逐月 N* 分布")
    n4 = sum(1 for m in range(2, 13) if sched[m][3] == 4)
    n6 = sum(1 for m in range(2, 13) if sched[m][3] == 6)
    n10 = sum(1 for m in range(2, 13) if sched[m][3] == 10)
    print(f"  N=4  : {n4}/11")
    print(f"  N=6  : {n6}/11")
    print(f"  N=10 : {n10}/11")

    # ---- 5. 收口 ----
    print(f"\n[5] 与基线对比")
    print(f"  V_dow（§30, N=6 固定）= {V_DOW_TOTAL:.4f} 万元")
    print(f"  V_dow_N（§37, N 联合）= {tot37:.4f} 万元（Δ = {tot37 - V_DOW_TOTAL:+.4f}）")
    if tot37 < V_DOW_TOTAL - 0.001:
        print(f"  ⇒ N 联合可降费 = {V_DOW_TOTAL - tot37:.4f} 万元")
    elif tot37 > V_DOW_TOTAL + 0.001:
        print(f"  ⇒ N 联合反而恶化 = {tot37 - V_DOW_TOTAL:+.4f} 万元")
    else:
        print(f"  ⇒ N=6 联合仍最优，N 维度已饱和")

    # ---- 落盘 ----
    out = dict(
        口径=dict(beta_grid=BETA_GRID, W_grid=W_GRID, gamma_grid=GAMMA_GRID,
                  N_DOW_HIST_grid=N_DOW_HIST_GRID, 候选数=len(cands)),
        自检_V0=cV0,
        V_dow_N=tot37,
        delta_vs_V_dow=tot37 - V_DOW_TOTAL,
        联合日程={str(m): {"beta": float(sched[m][0]), "W": int(sched[m][1]),
                           "gamma": float(sched[m][2]), "N_DOW_HIST": int(sched[m][3])}
                  for m in range(2, 13)},
        N分布=dict(N4=n4, N6=n6, N10=n10),
        对比=dict(V0=V0_TOTAL, V_dow=V_DOW_TOTAL, V_dow_N=tot37),
    )
    (BASE / "results" / "q2_dow_N_joint.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_dow_N_joint.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
