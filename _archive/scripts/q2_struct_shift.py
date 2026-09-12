# -*- coding: utf-8 -*-
r"""
§36 结构性采购微调 walk-forward（在 V_dow 的 b_base 上做基底修正）（只读，不改模型）
================================================================================
用户原"三层重构"的 Layer 2 剩余：V_dow 的 LP 已经最优（N̂ 完美已知）。
本节是"轻量诊断"：是否任何基底修正都比 b_base 更省钱？

  b_shift = clip(b_base + alpha · phi_k, 0, P_MAX)

5 基函数 × 3 α：
  phi_1 = I(t in 18:00-21:00)         # 峰时降购
  phi_2 = I(t in 22:00-06:00)         # 谷时加购
  phi_3 = I(t in 14:00-17:00)         # 峰前加购
  phi_4 = I(t in 10:00-14:00)         # 光伏谷加购
  phi_5 = I(t in 06:00-10:00)         # 早高峰加购
  alpha ∈ {-100, 0, +100} kWh/slot   # P_MAX=833, 取 100 可看效应不破约束

不动：V_dow 预测（用 §30 json 的逐月联合日程）、must/max、期末 geq2400。
15 候选 × 365 天 ≈ 45s。
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
                      P_MAX, Policy, load_data, make_plan, execute)

POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

# 5 基函数 × 3 α = 15 候选（每 slot 10 分钟，144 slot/天）
BASIS = {
    "peak_18-21":     lambda t: (t >= 108) & (t < 126),  # 18:00-21:00
    "off_22-06":      lambda t: (t >= 132) | (t < 36),
    "pre_peak_14-17": lambda t: (t >= 84) & (t < 108),
    "solar_10-14":    lambda t: (t >= 60) & (t < 84),
    "morning_06-10":  lambda t: (t >= 36) & (t < 60),
}
ALPHA_GRID = [-100.0, 0.0, 100.0]  # kWh/slot

V0_TOTAL = 1719.7084
V_DOW_TOTAL = 1412.0619


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


def delta_dow(d, X, dow_idx, n_dow_hist=6, W_level=14):
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


def forecast_dow(d, D, dow_idx, gamma, W_level, beta):
    """§30 的预测实现（与 q2_forecast_dow.forecast_dow 逐位等价）"""
    if d < 30:
        return D["net_ref"].copy()
    m = m_recent(d, D["net"], W_level)
    dlt = delta_dow(d, D["net"], dow_idx, 6, W_level)
    base = m + gamma * dlt
    net_hist = D["net"][max(0, d - W_level):d]
    if net_hist.shape[0] < 7:
        return D["net_ref"].copy()
    resid = net_hist - np.median(net_hist, axis=0)
    margin = _col_quantile(resid, np.full(T, beta))
    return base + margin


def replay_full_year_shift(D, dow_idx, sched, basis_name, alpha):
    """V_dow 联合日程 + (basis, α) 微调 b"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    E = float(E_INIT)
    dp = np.zeros(nd)
    de = np.zeros(nd)
    cum = np.zeros(nd)
    phi = BASIS[basis_name](np.arange(T)).astype(float)
    for d in range(nd):
        beta, W_level, gamma = sched.get(months[d], (0.70, 14, 0.0))  # 1月回退 V0 默认
        N_hat = forecast_dow(d, D, dow_idx, gamma, W_level, beta)
        b, c_plan = make_plan(POL, price, N_hat, E)
        if alpha != 0.0:
            b_shift = np.clip(b + alpha * phi, 0.0, P_MAX)
        else:
            b_shift = b
        c, dd, e, w, Etraj = execute(POL, b_shift, c_plan, N_hat, net[d], price, E)
        day_cost = float((b_shift * price).sum()) + float(MULT * (e * price).sum())
        dp[d] = float((b_shift * price).sum()); de[d] = float(MULT * (e * price).sum())
        E = Etraj[-1]
        cum[d] = day_cost if d == 0 else cum[d - 1] + day_cost
    return cum


def main():
    t_all = time.perf_counter()
    D = load_data()
    D["_dow"] = dow_index(D["dates"])
    months_arr = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    D["_month_end"] = {m: int(np.max(np.where(months_arr == m)[0])) for m in range(1, 13)}
    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]

    print("=" * 102)
    print("§36 结构性采购微调（在 V_dow 的 b_base 上做基底修正）— 用户 Layer 2 剩余")
    print("=" * 102)
    print(f"  b_shift = clip(b_base + α·φ_k, 0, P_MAX)")
    print(f"  5 基函数 × 3 α ∈ {ALPHA_GRID} = 15 候选")
    print(f"  其它：V_dow 联合日程 / must/max / 期末 geq2400")
    ncand = len(BASIS) * len(ALPHA_GRID)
    print(f"  候选数 = {ncand}")

    # ---- 0. 加载 §30 的 V_dow 联合日程 ----
    print(f"\n[0] 加载 §30 的 V_dow 联合日程")
    json_30 = json.loads((BASE / "results" / "q2_forecast_dow.json").read_text(encoding="utf-8"))
    sched_raw = json_30["联合日程"]
    sched = {int(k): (v["beta"], v["W"], v["gamma"]) for k, v in sched_raw.items()}
    print(f"  日程 {len(sched)} 月：")
    for m in range(2, 13):
        b_, w_, g_ = sched[m]
        print(f"    月{m}: β={b_:.2f} W={w_} γ={g_:.2f}")

    # ---- 1. α=0 (V_dow b) ⇒ 应 ≈ 1412.06 ----
    print(f"\n[1] α=0 + V_dow b 基线 replay")
    cum_dow = replay_full_year_shift(D, D["_dow"], sched, "peak_18-21", 0.0)
    cV_dow = float(cum_dow[idx[-1]] - cum_dow[idx[0] - 1]) / 1e4
    print(f"  V_dow b_base（α=0） 评价期 = {cV_dow:.4f} 万元（应 ≈ 1412.06）")

    # ---- 2. 15 候选回放 ----
    print(f"\n[2] 全年回放 15 个 (basis, α) 候选")
    results = {}
    for bn in BASIS:
        for alpha in ALPHA_GRID:
            cum_j = replay_full_year_shift(D, D["_dow"], sched, bn, alpha)
            results[(bn, alpha)] = cum_j
        print(f"    {bn} 完成（{time.perf_counter()-t_all:.0f}s）", flush=True)

    # ---- 3. 评价期总费用表 ----
    print(f"\n[3] 评价期总费用（按 α 与 basis）")
    print(f"  {'basis':<14}{'α=−100':>12}{'α=0':>12}{'α=+100':>12}{'min−α0':>12}")
    deltas = {}
    for bn in BASIS:
        ws = []
        for alpha in ALPHA_GRID:
            cV = float(results[(bn, alpha)][idx[-1]] - results[(bn, alpha)][idx[0] - 1]) / 1e4
            ws.append(cV)
        delta = min(ws) - ws[1]
        deltas[bn] = (ws, delta)
        print(f"  {bn:<14}{ws[0]:>12.4f}{ws[1]:>12.4f}{ws[2]:>12.4f}{delta:>+12.4f}")

    # ---- 4. 全年最优 (basis, α) ----
    print(f"\n[4] 全年最优点")
    best = min(results, key=lambda k: float(results[k][idx[-1]] - results[k][idx[0] - 1]) / 1e4)
    bn_best, a_best = best
    cV_best = float(results[best][idx[-1]] - results[best][idx[0] - 1]) / 1e4
    print(f"  最优 (basis, α) = ({bn_best}, {a_best:+.0f})")
    print(f"  评价期 = {cV_best:.4f} 万元（V_dow = {V_DOW_TOTAL:.4f}, Δ = {cV_best - V_DOW_TOTAL:+.4f}）")

    # ---- 5. 收口 ----
    if cV_best < V_DOW_TOTAL - 0.0001:
        print(f"\n  ⇒ 单基底单 α 修正可降费 = {V_DOW_TOTAL - cV_best:.4f} 万元")
        print(f"  ⇒ 进一步做 (basis, α) 月度 walk-forward 可有更多收益")
    else:
        print(f"\n  ⇒ 单基底单 α 修正**未超过** V_dow ⇒ LP 在 N̂ 上已最优")
        print(f"  ⇒ 任何 b 微调都会被 must/max 浪费或触发更多紧急费")

    # ---- 落盘 ----
    out = dict(
        口径=dict(basis=list(BASIS.keys()), alpha_grid=ALPHA_GRID,
                  P_MAX_kWh=P_MAX, prediction="V_dow",
                  execution="must/max", terminal="geq2400"),
        V_dow_baseline=cV_dow,
        全年最优=dict(basis=bn_best, alpha=a_best, total=cV_best),
        ΔvsV_dow=cV_best - V_DOW_TOTAL,
        逐basis_评价期={bn: {"α-100": deltas[bn][0][0],
                             "α0": deltas[bn][0][1],
                             "α+100": deltas[bn][0][2],
                             "Δ_最优_minus_0": deltas[bn][1]}
                        for bn in BASIS},
    )
    (BASE / "results" / "q2_struct_shift.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_struct_shift.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
