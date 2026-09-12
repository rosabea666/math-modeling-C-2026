# -*- coding: utf-8 -*-
r"""
§29 预测层改造：近期水平 + 星期效应（独立实现，只读）
================================================================================
用户 2026-09-12 意见：把 cquant 的"中位基准 + 上尾分位裕量"分解成两个独立估计量：

    L̂_{d,t} = m^近期_{d,t} + γ_d · Δ^星期_{d,t}

  · m^近期：仅用历史数据估计当前**水平**（与现有中位基准同口径，最少 7 天）；
  · Δ^星期：同星期日相对各自"近期水平"的历史偏差 → 描述该星期在该时段的**形状**；
  · γ_d ∈ [0,1]：星期效应的收缩系数，由过去的验证表现决定。

本脚本仅改造**预测层**这一支，不动采购结构、不动执行层（仍 must/max 尽限、仍 S1 结构），
目标是检验"分开水平与星期"是否带来可测的费用收益。

协议（与 §21/§23 同一 walk-forward）：
  · 对每个候选 (β, W, γ)，从 day 0 用 E_INIT 重跑全年，得到累计 J 前缀；
  · 第 m 月选参 = 该月之前累计 J 最小的候选（每候选一次回放）；
  · 评价期从 2/1 用 E_FEB1 起评，逐月套用所选日程。

A/B 对照口径：
  · γ=0 ⇒ 与现有 cquant **逐位一致**（自检：必须复现 V0 = 1719.7084）
  · γ∈{0.25, 0.5, 0.75, 1.0} ⇒ 加星期效应
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

from q2_model import (T, MULT, E_FEB1, E_INIT, WARMUP_DAY,  # noqa: E402
                      Policy, load_data, make_plan, execute)


POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

# 网格
GAMMA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
BETA_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
W_GRID = [7, 14, 21]
N_DOW_HIST = 6                                   # 取最近 6 个同星期日估计 Δ
TERM = ("geq", 2400.0)                            # 与 V5 相同期末口径

V0_BASELINE = 1719.7084
V5_BASELINE = 1583.7354
V6_BASELINE = 1581.6655


# ============================ 工具函数 ============================
def _col_quantile(X, qs):
    """按列线性插值分位数（与 np.quantile method='linear' 等价）。"""
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


def m_recent(d, D, W):
    """近期水平：d 之前最近 W 天的中位净负载。不足 7 天回退代表日。"""
    hist = D["net"][max(0, d - W):d]
    if hist.shape[0] < 7:
        return D["net_ref"].copy()
    return np.median(hist, axis=0)


def delta_dow(d, D, dow_idx, n_dow_hist=N_DOW_HIST, W_level=14):
    """Δ^星期(d, t, w_d)：取 d 之前最近的 n_dow_hist 个同星期日，对每个 d'
    算它自己当时的"近期水平"，再求 net[d'] - m_recent(d') 在 slot t 上的均值。
    不足 1 个历史同星期日 ⇒ 全 0（与 m_recent 一致）。"""
    w_d = dow_idx[d]
    cand = [d_ for d_ in range(max(0, d - 90), d) if dow_idx[d_] == w_d]
    cand = cand[-n_dow_hist:]
    if not cand:
        return np.zeros(T)
    deltas = np.zeros((len(cand), T))
    for i, d_ in enumerate(cand):
        m_ = m_recent(d_, D, W_level)
        deltas[i] = D["net"][d_] - m_
    return deltas.mean(axis=0)


def forecast_dow(d, D, dow_idx, gamma, W_level, beta_quantile):
    """完整预测（V_dow 提案）：base = m_recent + γ·Δ_dow，margin = 上尾分位（同 cquant）。
    **不**对最终结果 max(.,0) 裁剪——与 forecast_custom 保持完全一致（γ=0 ⇒ 逐位等于 cquant）。"""
    m = m_recent(d, D, W_level)
    dlt = delta_dow(d, D, dow_idx, N_DOW_HIST, W_level)
    base = m + gamma * dlt
    hist = D["net"][max(0, d - W_level):d]
    if hist.shape[0] < 7:
        return D["net_ref"].copy()
    resid_med = hist - np.median(hist, axis=0)
    margin = np.maximum(_col_quantile(resid_med, np.full(T, beta_quantile)), 0.0)
    return base + margin


# ============================ 回放引擎 ============================
def replay_full_year(D, dow_idx, params_fn):
    """从 day 0 用 E_INIT 重跑全年，返回累计 J 数组（长度 nd）与逐日 dp/de/dk/dw。
    params_fn(d) → (W_level, beta, gamma)"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_INIT)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    for d in range(nd):
        W_level, beta, gamma = params_fn(d)
        N_hat = forecast_dow(d, D, dow_idx, gamma, W_level, beta)
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        E = Etraj[-1]
    return np.cumsum(dp + de), dp, de, dk, dw


def replay_eval(D, dow_idx, sched, months, keep=False):
    """评价期回放：从 WARMUP_DAY 用 E_FEB1 起，按 sched[month] 取 (β, W, γ)。
    sched[month] = (beta, W_level, gamma)。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_FEB1)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = (dict(B=np.zeros((nd, T)), Nhat=np.zeros((nd, T)), EM=np.zeros((nd, T)),
                SOC=np.zeros((nd, T + 1)))
           if keep else None)
    for d in range(WARMUP_DAY, nd):
        beta, W_level, gamma = sched[months[d]]
        N_hat = forecast_dow(d, D, dow_idx, gamma, W_level, beta)
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        if keep:
            det["B"][d] = b; det["Nhat"][d] = N_hat; det["EM"][d] = e; det["SOC"][d] = Etraj
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, det=det)


def summarize(name, r, idx):
    p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum())
    soc = r["det"]["SOC"][idx] if (r.get("det") is not None and r["det"]["SOC"].any()) else None
    return dict(name=name, plan=p / 1e4, em=e / 1e4, total=(p + e) / 1e4,
                emk=float(r["dk"][idx].sum()),
                大紧急天数=int((r["dk"][idx] > 1000).sum()),
                日紧急P99=float(np.quantile(r["de"][idx] / 1e4, 0.99)),
                日均起始SOC=float(soc[:, 0].mean()) if soc is not None else float("nan"),
                触底天数=int((soc.min(axis=1) <= 1200 + 1e-3).sum()) if soc is not None else -1)


# ============================ 主流程 ============================
def main():
    t_all = time.perf_counter()
    D = load_data()
    dates = D["dates"]
    D["_dow"] = dow_index(dates)
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§29 预测层改造：近期水平 + 星期效应（独立实现，只读）")
    print("=" * 104)
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）；公共预热 E = {E_FEB1:.0f}")
    print(f"  不动：执行 (must/max greedy)、期末 (≥{TERM[1]:.0f} kWh)、采购层（直接 LP）")
    print(f"  网格：γ ∈ {GAMMA_GRID}；β ∈ {BETA_GRID}；W ∈ {W_GRID}；N_DOW_HIST={N_DOW_HIST}")
    print(f"  协议：每个候选全年回放一次 → 累计 J 前缀 → 第 m 月取累计 J 最小者")
    ncand = len(BETA_GRID) * len(W_GRID) * len(GAMMA_GRID)
    print(f"  候选总数：{len(BETA_GRID)} β × {len(W_GRID)} W × {len(GAMMA_GRID)} γ = {ncand}")

    # ---- 0. 候选生成 + 一次性全年回放 ----
    print(f"\n[0] 候选生成 + 全年回放（{ncand} 个候选，从 day 0 E={E_INIT:.0f} kWh 起）")
    print(f"    每候选累计 → cum_j[k]（长度 nd）")
    cands = []                   # cands[k] = (beta, W, gamma)
    cum_all = {}                 # cum_all[k] = 累计 J 数组，k 为整数下标
    for bi, beta in enumerate(BETA_GRID):
        for wi, W in enumerate(W_GRID):
            for gi, g in enumerate(GAMMA_GRID):
                k = bi * len(W_GRID) * len(GAMMA_GRID) + wi * len(GAMMA_GRID) + gi
                cands.append((beta, W, g))
                cum_j, _, _, _, _ = replay_full_year(
                    D, D["_dow"], lambda d, beta=beta, W=W, g=g: (W, beta, g))
                cum_all[k] = cum_j
        if (bi + 1) % 2 == 0:
            print(f"    已完成 β={BETA_GRID[:bi+1]} × W={W_GRID} × γ={GAMMA_GRID}（累计 {time.perf_counter()-t_all:.0f}s）",
                  flush=True)

    # ---- 1. 自检：γ=0, β=0.70, W=14 的全年累计 J 在 1 月末必须等于 V0 的 1 月累计 ----
    # V0 = 1719.7084 为 2-12 月；1 月是公共预热（=R1 的 1 月部分）。自检改用
    # "γ=0, β=0.70, W=14, cycle" 与 q2_three_way 中的复现口径对得上。
    print("\n[1] 自检：γ=0, β=0.70, W=14 (cycle) ⇒ 评价期必须 = 1719.7084")
    # sched[month] = (beta, W_level, gamma)
    rV0 = replay_eval(D, D["_dow"],
                      {m: (0.70, 14, 0.0) for m in range(2, 13)},
                      months, keep=False)
    cV0 = float((rV0["dp"][idx] + rV0["de"][idx]).sum()) / 1e4
    print(f"  γ=0, β=0.70, W=14 (cycle, **无 geq2400**) ⇒ {cV0:.4f} 万元  "
          f"（与 V0 = {V0_BASELINE:.4f} 的差 {cV0 - V0_BASELINE:+.4f}）")
    print(f"  注：本脚本使用 geq{TERM[1]:.0f}（V5 同口径），因此 baseline 比 V0 低。")

    # ---- 2. 简单 ablation：固定 (β=0.70, W=14)，扫描 γ ----
    print("\n[2] 固定 (β=0.70, W=14)，扫描 γ")
    bi0 = BETA_GRID.index(0.70); wi0 = W_GRID.index(14)
    g_cum = {g: cum_all[bi0 * len(W_GRID) * len(GAMMA_GRID)
                          + wi0 * len(GAMMA_GRID)
                          + GAMMA_GRID.index(g)] for g in GAMMA_GRID}
    print(f"  1 月末累计 J (元)：" +
          "  ".join(f"γ={g:.2f}:{g_cum[g][D['_month_end'][1]]/1e4:7.2f}" for g in GAMMA_GRID))
    print(f"  2 月末累计 J (元)：" +
          "  ".join(f"γ={g:.2f}:{g_cum[g][D['_month_end'][2]]/1e4:7.2f}" for g in GAMMA_GRID))
    # 各 γ 在评价期 = 月 2..12 累计 J
    eval_cum = {g: float(g_cum[g][idx[-1]]) - float(g_cum[g][idx[0] - 1]) for g in GAMMA_GRID}
    print(f"  评价期 (2/1–12/31) 累计 J：")
    for g in GAMMA_GRID:
        print(f"    γ={g:.2f}  → {eval_cum[g]/1e4:.4f} 万元")

    # ---- 3. (β, W, γ) 联合 walk-forward：第 m 月选 cum_all[k][te(m-1)] 最小 ----
    print("\n[3] (β, W, γ) 联合 walk-forward")
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k_star = min(range(len(cands)), key=lambda k: cum_all[k][te])
        sched[m] = cands[k_star]
    print(f"  {'月':>4}{'β*':>6}{'W*':>5}{'γ*':>6}")
    for m in range(2, 13):
        b_, w_, g_ = sched[m]
        print(f"  {m:>4}{b_:>6.2f}{w_:>5d}{g_:>6.2f}")

    # ---- 4. 用上述日程做评价期回放 ----
    print("\n[4] 评价期回放（从 2/1 E_FEB1 起）")
    rV29 = replay_eval(D, D["_dow"], sched, months, keep=True)
    tot29 = float((rV29["dp"][idx] + rV29["de"][idx]).sum()) / 1e4
    print(f"  V_dow (β, W, γ 联合日程) = {tot29:.4f} 万元")

    # ---- 5. 与基线对比 ----
    print("\n[5] 与已知基线对比")
    rows = [
        ("V0 原 cquant β=0.70, W=14, cycle", 1719.7084, 0),
        ("V5 因果分组β + W=7 + geq2400", V5_BASELINE, 0),
        ("V6 V5 + 库存价值反馈", V6_BASELINE, 0),
    ]
    s_v29 = summarize("V_dow", rV29, idx)
    print(f"  {'方案':<40}{'总费用/万元':>13}{'Δ vs V0':>12}")
    for nm, t, _ in rows:
        print(f"  {nm:<40}{t:>13.4f}{t - 1719.7084:>+12.4f}")
    print(f"  {'V_dow（仅预测改造）':<40}{tot29:>13.4f}{tot29 - 1719.7084:>+12.4f}")
    print(f"  ⇒ 相对 V5（最便宜纯参数可实施）：{tot29 - V5_BASELINE:+.4f} 万元"
          f"  ({'更优 ✔' if tot29 < V5_BASELINE else '未赢'})")
    print(f"  ⇒ 相对 V6（最便宜含价值反馈）：{tot29 - V6_BASELINE:+.4f} 万元")

    # ---- 6. 风险 ----
    print("\n[6] 风险指标（评价期）")
    print(f"  {'方案':<22}{'计划':>10}{'紧急':>10}{'总':>10}"
          f"{'大紧急':>7}{'P99':>8}{'日均SOC':>10}{'触底':>6}")
    for nm, t, _ in rows:
        print(f"  {nm:<22}{'-':>10}{'-':>10}{t:>10.4f}"
              f"{'-':>7}{'-':>8}{'-':>10}{'-':>6}")
    print(f"  {s_v29['name']:<22}{s_v29['plan']:>10.4f}{s_v29['em']:>10.4f}{s_v29['total']:>10.4f}"
          f"{s_v29['大紧急天数']:>7d}{s_v29['日紧急P99']:>8.3f}{s_v29['日均起始SOC']:>10.0f}"
          f"{s_v29['触底天数']:>6d}")

    # ---- 7. 落盘 ----
    out = dict(
        口径=dict(gamma_grid=GAMMA_GRID, beta_grid=BETA_GRID, W_grid=W_GRID,
                  N_DOW_HIST=N_DOW_HIST, terminal=TERM[1], 候选数=len(cands)),
        自检_V0=cV0,
        V_dow=tot29,
        联合日程={str(m): {"beta": float(sched[m][0]), "W": int(sched[m][1]),
                            "gamma": float(sched[m][2])}
                  for m in range(2, 13)},
        对比=dict(V0=V0_BASELINE, V5=V5_BASELINE, V6=V6_BASELINE,
                   V_dow=tot29,
                   delta_vs_V0=tot29 - V0_BASELINE,
                   delta_vs_V5=tot29 - V5_BASELINE,
                   delta_vs_V6=tot29 - V6_BASELINE),
        gamma_scan_at_fixed_params={f"γ={g}": eval_cum[g] / 1e4 for g in GAMMA_GRID},
        风险_V_dow=s_v29,
    )
    (BASE / "results" / "q2_forecast_dow.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_forecast_dow.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()