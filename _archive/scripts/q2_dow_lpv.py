# -*- coding: utf-8 -*-
r"""
§34 L/PV 分别建模 + 分别加 dow（只读，不改模型）
================================================================================
用户 2026-09-12 意见的"Layer 1"剩余部分：把 §30 在净负载层做的 dow 分解
拆成"负载侧 dow 生效、光伏侧 dow 不生效"两条独立预测。

  L̂_{d,t} = m^近期_L(d,t) + γ_L · Δ^星期_L(d,t)
  Ĝ_{d,t} = m^近期_PV(d,t)            ← 不加 dow（用户原话："光伏另建预测，不强行加入星期效应"）
  N̂_{d,t} = L̂_{d,t} - Ĝ_{d,t} + margin(t)

协议（与 §30 一致）：
  · 候选 = 7 β × 3 W × 5 γ_L = 105
  · γ_PV = 0 固定（按用户原方案）
  · 其它不动：执行 must/max、期末 geq2400、采购层直接 LP

测试目的是看 §30 的 V_dow（1412.06）是 L+PV 分离后还是不是更优。若分离后
更优 ⇒ L/PV 各自独立的 dow 信号在 §30 净负载分解下被"稀释"。
若分离后差不多 ⇒ §30 已捕获全部 dow 信号，分离不会进一步降费。
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

# 网格
GAMMA_L_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
BETA_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
W_GRID = [7, 14, 21]
N_DOW_HIST = 6

V0_TOTAL = 1719.7084
V_DOW_TOTAL = 1412.0619
V6_TOTAL = 1581.6655


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


def m_recent_X(d, X, W):
    """X ∈ (n_days, T) 的最近 W 天中位数（≥7 天回退到 30 天中位数；不足 30 天回退 0）"""
    hist = X[max(0, d - W):d]
    if hist.shape[0] < 7:
        if d >= 30:
            return np.median(X[max(0, d - 30):d], axis=0)
        return np.zeros(X.shape[1])
    return np.median(hist, axis=0)


def delta_dow_X(d, X, dow_idx, n_dow_hist=N_DOW_HIST, W_level=14):
    """Δ_dow_X(d, t)：取 d 之前最近的 n_dow_hist 个同星期日，对每个 d' 算它
    自己当时的 m_recent_X(d', W_level)，再求 X[d'] - m_recent 在 slot t 上的均值。"""
    w_d = dow_idx[d]
    cand = [d_ for d_ in range(max(0, d - 90), d) if dow_idx[d_] == w_d]
    cand = cand[-n_dow_hist:]
    if not cand:
        return np.zeros(X.shape[1])
    deltas = np.zeros((len(cand), X.shape[1]))
    for i, d_ in enumerate(cand):
        m_ = m_recent_X(d_, X, W_level)
        deltas[i] = X[d_] - m_
    return deltas.mean(axis=0)


def forecast_lpv(d, D, dow_idx, gamma_L, W_level, beta_quantile):
    """L/PV 分别建模预测：L̂ = m_L + γ·Δ_dow_L，Ĝ = m_PV，N̂ = L̂ - Ĝ + margin
    margin 用**净负载**残差（与 §30 一致）。"""
    m_L = m_recent_X(d, D["load"], W_level)
    m_PV = m_recent_X(d, D["pv"], W_level)
    dlt_L = delta_dow_X(d, D["load"], dow_idx, N_DOW_HIST, W_level)
    L_hat = m_L + gamma_L * dlt_L
    G_hat = m_PV
    base = L_hat - G_hat
    # margin：与 §30 同口径（净负载残差上尾分位）
    net_hist = D["net"][max(0, d - W_level):d]
    if net_hist.shape[0] < 7:
        return D["net_ref"].copy()
    resid = net_hist - np.median(net_hist, axis=0)
    margin = np.maximum(_col_quantile(resid, np.full(T, beta_quantile)), 0.0)
    return base + margin


def replay_full_year(D, dow_idx, params_fn):
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_INIT)
    dp = np.zeros(nd); de = np.zeros(nd)
    for d in range(nd):
        W_level, beta, gamma_L = params_fn(d)
        N_hat = forecast_lpv(d, D, dow_idx, gamma_L, W_level, beta)
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        E = Etraj[-1]
    return np.cumsum(dp + de)


def replay_eval(D, dow_idx, sched, months, keep=False):
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_FEB1)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = (dict(B=np.zeros((nd, T)), Nhat=np.zeros((nd, T)), EM=np.zeros((nd, T)),
                SOC=np.zeros((nd, T + 1)))
           if keep else None)
    for d in range(WARMUP_DAY, nd):
        beta, W_level, gamma_L = sched[months[d]]
        N_hat = forecast_lpv(d, D, dow_idx, gamma_L, W_level, beta)
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        if keep:
            det["B"][d] = b; det["Nhat"][d] = N_hat; det["EM"][d] = e; det["SOC"][d] = Etraj
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, det=det)


def main():
    t_all = time.perf_counter()
    D = load_data()
    D["_dow"] = dow_index(D["dates"])
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§34 L/PV 分别建模 + 分别加 dow（用户原方案 Layer 1 剩余）")
    print("=" * 104)
    print(f"  公式: L̂ = m_L + γ_L·Δ_dow_L，Ĝ = m_PV，N̂ = L̂ - Ĝ + margin")
    print(f"  γ_PV = 0 固定；其它网格 γ_L ∈ {GAMMA_L_GRID}；β ∈ {BETA_GRID}；W ∈ {W_GRID}")
    print(f"  N_DOW_HIST = {N_DOW_HIST}；其它不动：执行 must/max、期末 geq2400")
    ncand = len(BETA_GRID) * len(W_GRID) * len(GAMMA_L_GRID)
    print(f"  候选数 = {ncand}")

    # ---- 0. 自检：γ_L=0 + β=0.70 + W=14 + cycle ⇒ V0 = 1719.7084 ----
    print(f"\n[0] 自检 γ_L=0 + β=0.70 + W=14 + cycle ⇒ V0 = 1719.7084")
    bi0 = BETA_GRID.index(0.70); wi0 = W_GRID.index(14)
    cum0 = replay_full_year(D, D["_dow"], lambda d, bi0=bi0, wi0=wi0: (W_GRID[wi0], 0.70, 0.0))
    cV0 = float(cum0[idx[-1]] - cum0[idx[0] - 1]) / 1e4
    print(f"  γ_L=0, β=0.70, W=14 评价期 = {cV0:.4f} 万元（应 ≈ 1719.7084）")

    # ---- 1. 候选回放 ----
    print(f"\n[1] 全年回放 {ncand} 个候选")
    cands = []
    cum_all = {}
    k = 0
    for bi, beta in enumerate(BETA_GRID):
        for wi, W in enumerate(W_GRID):
            for gi, g in enumerate(GAMMA_L_GRID):
                cands.append((beta, W, g))
                cum_j = replay_full_year(
                    D, D["_dow"], lambda d, beta=beta, W=W, g=g: (W, beta, g))
                cum_all[k] = cum_j
                k += 1
        print(f"    β={beta:.2f} 完成（累计 {time.perf_counter()-t_all:.0f}s）", flush=True)

    # ---- 2. (β, W, γ_L) 联合 walk-forward ----
    print(f"\n[2] (β, W, γ_L) 联合 walk-forward")
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k_star = min(range(len(cands)), key=lambda i: cum_all[i][te])
        sched[m] = cands[k_star]
    print(f"  {'月':>4}{'β*':>6}{'W*':>5}{'γ_L*':>7}")
    for m in range(2, 13):
        b_, w_, g_ = sched[m]
        print(f"  {m:>4}{b_:>6.2f}{w_:>5d}{g_:>7.2f}")

    # ---- 3. 评价期回放 ----
    print(f"\n[3] 评价期回放")
    rV34 = replay_eval(D, D["_dow"], sched, months, keep=True)
    tot34 = float((rV34["dp"][idx] + rV34["de"][idx]).sum()) / 1e4
    print(f"  V_lpv (β, W, γ_L 联合日程) = {tot34:.4f} 万元")
    print(f"  Δ vs V_dow (γ=1.0 on net) = {tot34 - V_DOW_TOTAL:+.4f} 万元")
    print(f"  Δ vs V0 = {tot34 - V0_TOTAL:+.4f} 万元")

    # ---- 4. γ_L 分布 ----
    print(f"\n[4] 逐月 γ_L* 分布")
    above_1 = sum(1 for m in range(2, 13) if sched[m][2] > 1.0)
    at_1 = sum(1 for m in range(2, 13) if sched[m][2] == 1.0)
    below_1 = sum(1 for m in range(2, 13) if sched[m][2] < 1.0)
    print(f"  γ_L > 1.0 的月份: {above_1}/11")
    print(f"  γ_L = 1.0 的月份: {at_1}/11")
    print(f"  γ_L < 1.0 的月份: {below_1}/11")

    # ---- 5. 风险 ----
    print(f"\n[5] 风险（评价期）")
    p = float(rV34["dp"][idx].sum()); e = float(rV34["de"][idx].sum())
    soc = rV34["det"]["SOC"][idx]
    print(f"  {'方案':<22}{'计划':>10}{'紧急':>10}{'总':>10}"
          f"{'大紧急':>7}{'P99':>8}{'日均SOC':>10}{'触底':>6}")
    print(f"  {'V_lpv':<22}{p/1e4:>10.4f}{e/1e4:>10.4f}{(p+e)/1e4:>10.4f}"
          f"{int((rV34['dk'][idx] > 1000).sum()):>7d}"
          f"{float(np.quantile(rV34['de'][idx]/1e4, 0.99)):>8.3f}"
          f"{float(soc[:, 0].mean()):>10.0f}"
          f"{int((soc.min(axis=1) <= 1200+1e-3).sum()):>6d}")

    # ---- 6. 与基线对比 ----
    print(f"\n[6] 与基线对比")
    print(f"  V0              = {V0_TOTAL:.4f} 万元")
    print(f"  V_dow (γ on net)= {V_DOW_TOTAL:.4f} 万元")
    print(f"  V_lpv (γ on L)  = {tot34:.4f} 万元  (Δ vs V_dow = {tot34 - V_DOW_TOTAL:+.4f})")
    print(f"  ⇒ γ_L on load 是否优于 γ on net: "
          f"{'✔（L/PV 分离建模更优）' if tot34 < V_DOW_TOTAL else '✘（净负载 dow 分解已捕获）'}")

    # ---- 落盘 ----
    out = dict(
        口径=dict(gamma_L_grid=GAMMA_L_GRID, beta_grid=BETA_GRID, W_grid=W_GRID,
                  gamma_PV=0.0, N_DOW_HIST=N_DOW_HIST, 候选数=len(cands)),
        自检_V0=cV0,
        V_lpv=tot34,
        delta_vs_V_dow=tot34 - V_DOW_TOTAL,
        联合日程={str(m): {"beta": float(sched[m][0]), "W": int(sched[m][1]),
                            "gamma_L": float(sched[m][2])}
                  for m in range(2, 13)},
        gamma_L分布=dict(above_1=above_1, at_1=at_1, below_1=below_1),
        对比=dict(V0=V0_TOTAL, V_dow=V_DOW_TOTAL, V_lpv=tot34),
    )
    (BASE / "results" / "q2_dow_lpv.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_dow_lpv.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()