# -*- coding: utf-8 -*-
r"""
§32 γ 网格上扩到 1.5（只读）
================================================================================
§30 用 γ ∈ {0, 0.25, 0.5, 0.75, 1.0} 跑 walk-forward，9/11 个月选 γ=1.0（网格上界）。
本节检查 γ=1.0 是不是内点最优还是边界截断：
  · 把网格扩到 {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
  · 重新跑 (β, W, γ) 联合 walk-forward
  · 看每个月最优 γ 是否仍顶在 1.0、还是被更大的 γ 取代

预计 0-15 万元节省（如 γ=1.5 取代 1.0 ⇒ 估 5-15 万；如仍顶在 1.0 ⇒ 没有节省）。

实现：复用 `q2_forecast_dow.py` 的 `forecast_dow`、`replay_full_year`、`_col_quantile`、
`dow_index`、`m_recent`、`delta_dow`，只把 `GAMMA_GRID` 替换为扩展集合，重写主流程。
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_FEB1, WARMUP_DAY, E_INIT,  # noqa: E402
                      Policy, load_data, make_plan, execute)
import q2_forecast_dow as fd

POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

# 扩展 γ 网格（保留原 §30 的 5 个点）
GAMMA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
BETA_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
W_GRID = [7, 14, 21]

V0_TOTAL = 1719.7084
V_DOW_TOTAL = 1412.0619
V6_TOTAL = 1581.6655


def replay_full_year(D, dow_idx, params_fn):
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_INIT)
    dp = np.zeros(nd); de = np.zeros(nd)
    for d in range(nd):
        W_level, beta, gamma = params_fn(d)
        N_hat = fd.forecast_dow(d, D, dow_idx, gamma, W_level, beta)
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
        beta, W_level, gamma = sched[months[d]]
        N_hat = fd.forecast_dow(d, D, dow_idx, gamma, W_level, beta)
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
    D["_dow"] = fd.dow_index(D["dates"])
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§32 γ 网格上扩到 1.5（验证 §30 的 γ=1.0 是内点最优还是边界截断）")
    print("=" * 104)
    print(f"  γ 网格: {GAMMA_GRID}")
    print(f"  β 网格: {BETA_GRID}")
    print(f"  W 网格: {W_GRID}")
    ncand = len(BETA_GRID) * len(W_GRID) * len(GAMMA_GRID)
    print(f"  候选总数：{ncand}（§30 是 105，本节 {ncand}）")
    print(f"  其它不动：执行 must/max、期末 geq2400、采购层直接 LP")

    # ---- 0. 候选生成 + 全年回放 ----
    print(f"\n[0] 全年回放 {ncand} 个候选")
    cands = []                  # cands[k] = (beta, W, gamma)
    cum_all = {}                # cum_all[k] = 累计 J 数组
    k = 0
    for bi, beta in enumerate(BETA_GRID):
        for wi, W in enumerate(W_GRID):
            for gi, g in enumerate(GAMMA_GRID):
                cands.append((beta, W, g))
                cum_j = replay_full_year(
                    D, D["_dow"], lambda d, beta=beta, W=W, g=g: (W, beta, g))
                cum_all[k] = cum_j
                k += 1
        print(f"    β={beta:.2f} 完成（累计 {time.perf_counter()-t_all:.0f}s）", flush=True)

    # ---- 1. 自检：γ=0 + β=0.70 + W=14 + cycle ⇒ 评价期 = 1719.7084 ----
    print(f"\n[1] 自检：γ=0, β=0.70, W=14 + cycle ⇒ 1719.7084")
    # 找 (β=0.70, W=14, γ=0.0) 的索引
    bi0 = BETA_GRID.index(0.70); wi0 = W_GRID.index(14); gi0 = GAMMA_GRID.index(0.0)
    k0 = bi0 * len(W_GRID) * len(GAMMA_GRID) + wi0 * len(GAMMA_GRID) + gi0
    te_eval = idx[-1]; te_start = idx[0]
    cum_eval_0 = float(cum_all[k0][te_eval] - cum_all[k0][te_start - 1])
    print(f"  γ=0, β=0.70, W=14 (cycle) 评价期 = {cum_eval_0/1e4:.4f} 万元（应 ≈ 1719.7084）")

    # ---- 2. (β, W, γ) 联合 walk-forward ----
    print(f"\n[2] (β, W, γ) 联合 walk-forward")
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k_star = min(range(len(cands)), key=lambda i: cum_all[i][te])
        sched[m] = cands[k_star]
    print(f"  {'月':>4}{'β*':>6}{'W*':>5}{'γ*':>6}")
    for m in range(2, 13):
        b_, w_, g_ = sched[m]
        print(f"  {m:>4}{b_:>6.2f}{w_:>5d}{g_:>6.2f}")

    # ---- 3. 评价期回放 ----
    print(f"\n[3] 评价期回放")
    rV32 = replay_eval(D, D["_dow"], sched, months, keep=True)
    tot32 = float((rV32["dp"][idx] + rV32["de"][idx]).sum()) / 1e4
    print(f"  V_dow_ext (β, W, γ 联合日程) = {tot32:.4f} 万元")
    print(f"  Δ vs V_dow (γ 上界=1.0) = {tot32 - V_DOW_TOTAL:+.4f} 万元")

    # ---- 4. γ 分布（看 γ=1.0 是不是被更大的 γ 取代）----
    print(f"\n[4] 逐月 γ* 分布（看是否顶在 1.0）")
    above_1 = sum(1 for m in range(2, 13) if sched[m][2] > 1.0)
    at_1 = sum(1 for m in range(2, 13) if sched[m][2] == 1.0)
    below_1 = sum(1 for m in range(2, 13) if sched[m][2] < 1.0)
    print(f"  γ > 1.0 的月份: {above_1}/11")
    print(f"  γ = 1.0 的月份: {at_1}/11")
    print(f"  γ < 1.0 的月份: {below_1}/11")

    # ---- 5. 风险 ----
    print(f"\n[5] 风险（评价期）")
    p = float(rV32["dp"][idx].sum()); e = float(rV32["de"][idx].sum())
    soc = rV32["det"]["SOC"][idx]
    print(f"  {'方案':<22}{'计划':>10}{'紧急':>10}{'总':>10}"
          f"{'大紧急':>7}{'P99':>8}{'日均SOC':>10}{'触底':>6}")
    print(f"  {'V_dow_ext':<22}{p/1e4:>10.4f}{e/1e4:>10.4f}{(p+e)/1e4:>10.4f}"
          f"{int((rV32['dk'][idx] > 1000).sum()):>7d}"
          f"{float(np.quantile(rV32['de'][idx]/1e4, 0.99)):>8.3f}"
          f"{float(soc[:, 0].mean()):>10.0f}"
          f"{int((soc.min(axis=1) <= 1200+1e-3).sum()):>6d}")

    # ---- 6. 与基线对比 ----
    print(f"\n[6] 与基线对比")
    print(f"  V0              = {V0_TOTAL:.4f} 万元")
    print(f"  V_dow (γ≤1.0)  = {V_DOW_TOTAL:.4f} 万元")
    print(f"  V_dow_ext       = {tot32:.4f} 万元  (Δ vs V_dow = {tot32 - V_DOW_TOTAL:+.4f})")
    print(f"  ⇒ γ=1.0 是否边界截断: {'是（γ>1.0 给出更优）' if tot32 < V_DOW_TOTAL else '否（γ=1.0 已是内点最优或被 γ>1.0 推到更差）'}")

    # ---- 落盘 ----
    out = dict(
        口径=dict(gamma_grid=GAMMA_GRID, beta_grid=BETA_GRID, W_grid=W_GRID,
                  候选数=len(cands)),
        自检_V0=cum_eval_0 / 1e4,
        V_dow_ext=tot32,
        delta_vs_V_dow=tot32 - V_DOW_TOTAL,
        联合日程={str(m): {"beta": float(sched[m][0]), "W": int(sched[m][1]),
                            "gamma": float(sched[m][2])}
                  for m in range(2, 13)},
        gamma分布=dict(above_1=above_1, at_1=at_1, below_1=below_1),
        对比=dict(V0=V0_TOTAL, V_dow=V_DOW_TOTAL, V_dow_ext=tot32),
    )
    (BASE / "results" / "q2_dow_gamma_extend.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_dow_gamma_extend.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()