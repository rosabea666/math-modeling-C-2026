# -*- coding: utf-8 -*-
r"""
§35 联合 (γ_L, γ_PV) walk-forward（只读，不改模型）
================================================================================
§34 的反向结论："光伏另建预测、强制 γ_PV=0"是把 dow 信号拆到 L 侧 = 系统性
高估净 dow。本节测**两条 dow 信号都可调**时的因果选参，看能否：
  · 完全复现 §30 的 V_dow = 1412.06（说明 dow 信号可分解为 (γ_L, γ_PV)）
  · 或者超过 V_dow（说明 §30 的 γ-on-net 还有冗余）
  · 或者输给 V_dow（说明 §30 是 γ 方向的最优点，不可分解）

公式（在 §34 基础上）：
  L̂_{d,t} = m^L + γ_L · Δ_dow^L
  Ĝ_{d,t} = m^PV + γ_PV · Δ_dow^PV
  N̂_{d,t} = L̂ − Ĝ + margin(t)

候选（60 个）：
  γ_L    ∈ {0.0, 0.5, 1.0}        (3)
  γ_PV   ∈ {-1.0, -0.5, 0.0, 0.5, 1.0}   (5)  ← 含负号让 PV 反向 dow 也可试
  β      ∈ {0.70, 0.75}           (2)  ← §30 验证过的最优区间
  W      ∈ {7, 14}                (2)  ← §21 验证 W=7 极佳

协议（与 §30/§34 一致）：
  · 候选 walk-forward 选参
  · 必须/max 执行、期末 geq2400
  · γ_L=γ_PV=0 自检 ⇒ 与 §34 的 γ_L=0 一致（应 ≈ 1677.31）

预期：用 60 候选（≈ 132s）跑完一轮，看最优 (γ_L*, γ_PV*) 月度分布。
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

# 网格（60 候选）
GAMMA_L_GRID = [0.0, 0.5, 1.0]
GAMMA_PV_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
BETA_GRID = [0.70, 0.75]
W_GRID = [7, 14]
N_DOW_HIST = 6

V0_TOTAL = 1719.7084
V_DOW_TOTAL = 1412.0619
V_LPV34_TOTAL = 1548.5700


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
    自己当时的 m_recent_X(d', W_level)，再求 X[d'] - m_ 在 slot t 上的均值。"""
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


def forecast_joint(d, D, dow_idx, gamma_L, gamma_PV, W_level, beta_quantile):
    """联合 dow：L̂ = m_L + γ_L·Δ_dow_L；Ĝ = m_PV + γ_PV·Δ_dow_PV；
    N̂ = L̂ − Ĝ + margin（margin 用净负载残差上尾分位，与 §30 同口径）"""
    m_L = m_recent_X(d, D["load"], W_level)
    m_PV = m_recent_X(d, D["pv"], W_level)
    dlt_L = delta_dow_X(d, D["load"], dow_idx, N_DOW_HIST, W_level)
    dlt_PV = delta_dow_X(d, D["pv"], dow_idx, N_DOW_HIST, W_level)
    L_hat = m_L + gamma_L * dlt_L
    G_hat = m_PV + gamma_PV * dlt_PV
    base = L_hat - G_hat
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
        W_level, beta, gamma_L, gamma_PV = params_fn(d)
        N_hat = forecast_joint(d, D, dow_idx, gamma_L, gamma_PV, W_level, beta)
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
        beta, W_level, gamma_L, gamma_PV = sched[months[d]]
        N_hat = forecast_joint(d, D, dow_idx, gamma_L, gamma_PV, W_level, beta)
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

    print("=" * 110)
    print("§35 联合 (γ_L, γ_PV) walk-forward（用 §30 的净 dow 信号分解为两条独立信号）")
    print("=" * 110)
    print(f"  公式: L̂ = m_L + γ_L·Δ_dow_L；Ĝ = m_PV + γ_PV·Δ_dow_PV；N̂ = L̂ − Ĝ + margin")
    print(f"  γ_L ∈ {GAMMA_L_GRID} | γ_PV ∈ {GAMMA_PV_GRID} | β ∈ {BETA_GRID} | W ∈ {W_GRID}")
    ncand = len(BETA_GRID) * len(W_GRID) * len(GAMMA_L_GRID) * len(GAMMA_PV_GRID)
    print(f"  候选数 = {ncand}")

    # ---- 0. 自检：γ_L=0, γ_PV=0, β=0.70, W=14 ⇒ 与 §34 的 γ_L=0 一致（应 ≈ 1677.31）----
    print(f"\n[0] 自检 γ_L=0, γ_PV=0, β=0.70, W=14 cycle ⇒ 应 ≈ 1677.31 (§34 自检)")
    cum0 = replay_full_year(D, D["_dow"], lambda d: (14, 0.70, 0.0, 0.0))
    cV0 = float(cum0[idx[-1]] - cum0[idx[0] - 1]) / 1e4
    print(f"  γ_L=0, γ_PV=0, β=0.70, W=14 评价期 = {cV0:.4f} 万元")

    # ---- 1. 候选回放 ----
    print(f"\n[1] 全年回放 {ncand} 个候选")
    cands = []  # [(beta, W, gamma_L, gamma_PV), ...]
    cum_all = {}
    k = 0
    for bi, beta in enumerate(BETA_GRID):
        for wi, W in enumerate(W_GRID):
            for gli, gamma_L in enumerate(GAMMA_L_GRID):
                for gpi, gamma_PV in enumerate(GAMMA_PV_GRID):
                    cands.append((beta, W, gamma_L, gamma_PV))
                    cum_j = replay_full_year(
                        D, D["_dow"],
                        lambda d, beta=beta, W=W, gamma_L=gamma_L, gamma_PV=gamma_PV:
                            (W, beta, gamma_L, gamma_PV))
                    cum_all[k] = cum_j
                    k += 1
        print(f"    β={beta:.2f} 完成（累计 {time.perf_counter()-t_all:.0f}s）", flush=True)

    # ---- 2. (β, W, γ_L, γ_PV) 联合 walk-forward ----
    print(f"\n[2] (β, W, γ_L, γ_PV) 联合 walk-forward")
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k_star = min(range(len(cands)), key=lambda i: cum_all[i][te])
        sched[m] = cands[k_star]
    print(f"  {'月':>4}{'β*':>6}{'W*':>5}{'γ_L*':>7}{'γ_PV*':>8}")
    for m in range(2, 13):
        b_, w_, gl_, gp_ = sched[m]
        print(f"  {m:>4}{b_:>6.2f}{w_:>5d}{gl_:>7.2f}{gp_:>8.2f}")

    # ---- 3. 评价期回放 ----
    print(f"\n[3] 评价期回放")
    rV35 = replay_eval(D, D["_dow"], sched, months, keep=True)
    tot35 = float((rV35["dp"][idx] + rV35["de"][idx]).sum()) / 1e4
    print(f"  V_lpv_joint (β, W, γ_L, γ_PV 联合日程) = {tot35:.4f} 万元")
    print(f"  Δ vs V_dow        = {tot35 - V_DOW_TOTAL:+.4f} 万元")
    print(f"  Δ vs V_lpv(§34)   = {tot35 - V_LPV34_TOTAL:+.4f} 万元")
    print(f"  Δ vs V0           = {tot35 - V0_TOTAL:+.4f} 万元")

    # ---- 4. γ 信号分布 ----
    print(f"\n[4] 逐月 γ* 分布")
    gl_pos = sum(1 for m in range(2, 13) if sched[m][2] > 0)
    gl_zero = sum(1 for m in range(2, 13) if sched[m][2] == 0.0)
    gl_neg = sum(1 for m in range(2, 13) if sched[m][2] < 0)
    gp_pos = sum(1 for m in range(2, 13) if sched[m][3] > 0)
    gp_zero = sum(1 for m in range(2, 13) if sched[m][3] == 0.0)
    gp_neg = sum(1 for m in range(2, 13) if sched[m][3] < 0)
    print(f"  γ_L > 0: {gl_pos}/11 | γ_L = 0: {gl_zero}/11 | γ_L < 0: {gl_neg}/11")
    print(f"  γ_PV > 0: {gp_pos}/11 | γ_PV = 0: {gp_zero}/11 | γ_PV < 0: {gp_neg}/11")

    # ---- 5. 风险 ----
    print(f"\n[5] 风险（评价期）")
    p = float(rV35["dp"][idx].sum()); e_ = float(rV35["de"][idx].sum())
    soc = rV35["det"]["SOC"][idx]
    print(f"  {'方案':<22}{'计划':>10}{'紧急':>10}{'总':>10}"
          f"{'大紧急':>7}{'P99':>8}{'日均SOC':>10}{'触底':>6}")
    print(f"  {'V_lpv_joint':<22}{p/1e4:>10.4f}{e_/1e4:>10.4f}{(p+e_)/1e4:>10.4f}"
          f"{int((rV35['dk'][idx] > 1000).sum()):>7d}"
          f"{float(np.quantile(rV35['de'][idx]/1e4, 0.99)):>8.3f}"
          f"{float(soc[:, 0].mean()):>10.0f}"
          f"{int((soc.min(axis=1) <= 1200+1e-3).sum()):>6d}")

    # ---- 6. 与基线对比 ----
    print(f"\n[6] 与基线对比")
    print(f"  V0              = {V0_TOTAL:.4f} 万元")
    print(f"  V_dow (γ on net)= {V_DOW_TOTAL:.4f} 万元")
    print(f"  V_lpv (§34)     = {V_LPV34_TOTAL:.4f} 万元")
    print(f"  V_lpv_joint(§35)= {tot35:.4f} 万元  "
          f"(Δ vs V_dow = {tot35 - V_DOW_TOTAL:+.4f})")
    if tot35 < V_DOW_TOTAL:
        print(f"  ⇒ 联合可调超过 V_dow：dow 信号**可分解**为独立 (γ_L, γ_PV)")
    elif tot35 == V_DOW_TOTAL:
        print(f"  ⇒ 联合 = V_dow：dow 信号**精确**等价于 γ-on-net")
    else:
        print(f"  ⇒ 联合仍劣于 V_dow：γ-on-net 是最优 dow 形式，不可被分拆替代")

    # 找候选中是否选中 γ_PV != 0
    picks_with_gp = sum(1 for m in range(2, 13) if abs(sched[m][3]) > 1e-9)
    print(f"  ⇒ 11 个月中选 γ_PV ≠ 0 的次数: {picks_with_gp}/11")

    # ---- 落盘 ----
    out = dict(
        口径=dict(gamma_L_grid=GAMMA_L_GRID, gamma_PV_grid=GAMMA_PV_GRID,
                  beta_grid=BETA_GRID, W_grid=W_GRID,
                  N_DOW_HIST=N_DOW_HIST, 候选数=len(cands)),
        自检_γ都0=cV0,
        V_lpv_joint=tot35,
        delta_vs_V_dow=tot35 - V_DOW_TOTAL,
        delta_vs_V_lpv34=tot35 - V_LPV34_TOTAL,
        联合日程={str(m): {"beta": float(sched[m][0]), "W": int(sched[m][1]),
                           "gamma_L": float(sched[m][2]), "gamma_PV": float(sched[m][3])}
                  for m in range(2, 13)},
        γ分布=dict(gamma_L=dict(pos=gl_pos, zero=gl_zero, neg=gl_neg),
                   gamma_PV=dict(pos=gp_pos, zero=gp_zero, neg=gp_neg)),
        对比=dict(V0=V0_TOTAL, V_dow=V_DOW_TOTAL,
                  V_lpv34=V_LPV34_TOTAL, V_lpv_joint=tot35),
    )
    (BASE / "results" / "q2_dow_lpv_joint.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_dow_lpv_joint.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
