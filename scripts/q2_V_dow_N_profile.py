# -*- coding: utf-8 -*-
r"""
§39 V_dow_N 风险与月度成本档案（只读，不改模型）
================================================================================
主方案 = V_dow_N = 1395.6995 万元（§37 walk-forward 锁定）。
本节重放评价期，输出**完整的风险档案**，供论文 §7 使用：

  · 月度成本（计划 + 紧急 + 总）
  · 紧急费 P50/P90/P99
  · 大紧急天数 (>1000 kWh)
  · 触底天数 (日 SOC min < 1200)
  · 触顶天数 (日 SOC max ≥ E_MAX-1)
  · 起始/终止/平均 SOC
  · 计划量、紧急量、弃置量
  · 与 V_dow（1412.06）的逐月对比

输出：results/q2_V_dow_N_profile.json
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

from q2_model import (T, MULT, E_FEB1, E_MAX, E_MIN, WARMUP_DAY,  # noqa: E402
                      Policy, load_data, make_plan, execute)

POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

V_DOW_N_TOTAL = 1395.6995  # §37
V_DOW_TOTAL = 1412.0619    # §30


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


def main():
    t_all = time.perf_counter()
    D = load_data()
    D["_dow"] = dow_index(D["dates"])
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]

    print("=" * 100)
    print("§39 V_dow_N 风险与月度成本档案")
    print("=" * 100)
    print(f"  主方案：β=0.70 (月2-3) / β=0.75 (月4-12), W=7, γ=1.0, N_DOW_HIST=4")
    print(f"  must/max, 期末 E_H ≥ 2400, 评价期 2025.2.1–12.31")
    print(f"  期望：V_dow_N = 1395.6995 万元")

    # ---- 加载 §37 的逐月日程 ----
    json_37 = json.loads((BASE / "results" / "q2_dow_N_joint.json").read_text(encoding="utf-8"))
    sched = {int(k): (v["beta"], v["W"], v["gamma"], v["N_DOW_HIST"])
             for k, v in json_37["联合日程"].items()}

    # ---- 评价期详细 replay ----
    print(f"\n[1] 评价期 replay (with keep=True)")
    nd = len(D["dates"])
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    B = np.zeros((nd, T)); Nhat = np.zeros((nd, T)); EM = np.zeros((nd, T))
    SOC = np.zeros((nd, T + 1))
    E = float(E_FEB1)
    for d in range(WARMUP_DAY, nd):
        beta, W_level, gamma, n_dow_hist = sched[months[d]]
        N_hat = forecast_dow_N(d, D, D["_dow"], gamma, W_level, beta, n_dow_hist)
        b, c_plan = make_plan(POL, D["price"], N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, D["net"][d], D["price"], E)
        dp[d] = float((b * D["price"]).sum())
        de[d] = float(MULT * (e * D["price"]).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        B[d] = b; Nhat[d] = N_hat; EM[d] = e; SOC[d] = Etraj
        E = Etraj[-1]
    print(f"  replay 完成（{time.perf_counter()-t_all:.0f}s）")

    # ---- 总览 ----
    print(f"\n[2] 总览")
    tot = float((dp[idx] + de[idx]).sum()) / 1e4
    plan = float(dp[idx].sum()) / 1e4
    emerg = float(de[idx].sum()) / 1e4
    print(f"  总费用  = {tot:.4f} 万元")
    print(f"  计划费  = {plan:.4f} 万元")
    print(f"  紧急费  = {emerg:.4f} 万元")
    print(f"  Δ vs V_dow (1412.06) = {tot - V_DOW_TOTAL:+.4f} 万元")
    print(f"  vs 期望 V_dow_N (1395.70) = {tot - V_DOW_N_TOTAL:+.4f} 万元")

    # ---- 月度成本 ----
    print(f"\n[3] 月度成本（万元）")
    print(f"  {'月':>4}{'计划':>12}{'紧急':>12}{'总':>12}{'Δ_上':>10}{'大紧急':>7}{'P99':>8}{'触底':>6}{'触顶':>6}")
    monthly = {}
    for m in range(2, 13):
        m_idx = idx[(months[idx] == m)]
        p_m = float(dp[m_idx].sum()) / 1e4
        e_m = float(de[m_idx].sum()) / 1e4
        tot_m = p_m + e_m
        big_m = int((dk[m_idx] > 1000).sum())
        p99_m = float(np.quantile(de[m_idx] / 1e4, 0.99)) if len(m_idx) else 0.0
        # 触底：日内 SOC min < 1201
        soc_min = SOC[m_idx].min(axis=1)
        touch_bottom = int((soc_min <= E_MIN + 1).sum())
        soc_max = SOC[m_idx].max(axis=1)
        touch_top = int((soc_max >= E_MAX - 1).sum())
        delta = e_m if m == 2 else monthly[m - 1]["cumul"]
        cumul = (tot_m if m == 2 else monthly[m - 1]["cumul"] + tot_m)
        monthly[m] = dict(plan=p_m, emerg=e_m, total=tot_m, cumul=cumul,
                          big_emerg=big_m, p99_emerg=p99_m,
                          touch_bottom=touch_bottom, touch_top=touch_top)
        print(f"  {m:>4}{p_m:>12.4f}{e_m:>12.4f}{tot_m:>12.4f}"
              f"{cumul:>10.2f}{big_m:>7d}{p99_m:>8.3f}{touch_bottom:>6d}{touch_top:>6d}")

    # ---- 风险指标总览 ----
    print(f"\n[4] 风险指标总览（评价期 334 天）")
    soc = SOC[idx]
    de_arr = de[idx]
    print(f"  P50 紧急费  = {float(np.quantile(de_arr/1e4, 0.50)):.3f} 万元")
    print(f"  P90 紧急费  = {float(np.quantile(de_arr/1e4, 0.90)):.3f} 万元")
    print(f"  P95 紧急费  = {float(np.quantile(de_arr/1e4, 0.95)):.3f} 万元")
    print(f"  P99 紧急费  = {float(np.quantile(de_arr/1e4, 0.99)):.3f} 万元")
    print(f"  最大单日紧急 = {float(de_arr.max()/1e4):.3f} 万元")
    print(f"  大紧急天数 (>1000 kWh) = {int((dk[idx] > 1000).sum())}/{len(idx)}")
    print(f"  触底天数 (SOC_min < E_MIN+1) = {int((soc.min(axis=1) <= E_MIN+1).sum())}/{len(idx)}")
    print(f"  触顶天数 (SOC_max = E_MAX)  = {int((soc.max(axis=1) >= E_MAX-1).sum())}/{len(idx)}")
    print(f"  日均起始 SOC = {float(soc[:, 0].mean()):.0f} kWh")
    print(f"  日均最低 SOC = {float(soc.min(axis=1).mean()):.0f} kWh")
    print(f"  日均最高 SOC = {float(soc.max(axis=1).mean()):.0f} kWh")
    print(f"  年末 SOC (12.31 末尾) = {float(soc[-1, -1]):.0f} kWh")
    print(f"  总弃置 (w) = {float(dw[idx].sum()):.0f} kWh = {float(dw[idx].sum())/1e4:.2f} 万元（计费）")

    # ---- 指定日期（与论文 §7.5 对齐）----
    print(f"\n[5] 指定日期计划与紧急（对应论文 §7.5）")
    spec_dates = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
    spec_table = {}
    for ds in spec_dates:
        d_int = int(np.where(D["dates"] == np.datetime64(ds))[0][0])
        if d_int not in idx:
            continue
        # 按论文 §7.5 的"字面时钟区间"格式
        slots_idx = {
            '10:00-10:10': 60, '12:00-12:10': 72, '14:00-14:10': 84,
            '16:00-16:10': 96, '18:00-18:10': 108, '20:00-20:10': 120,
        }
        b_slots = {lbl: float(B[d_int, sl]) for lbl, sl in slots_idx.items()}
        b_total = float(B[d_int].sum())
        emerg_kwh = float(EM[d_int].sum())
        plan_cost = float((B[d_int] * D["price"]).sum())
        emerg_cost = float(MULT * (EM[d_int] * D["price"]).sum())
        spec_table[ds] = dict(
            b_slots=b_slots, b_total=b_total,
            emerg_kwh=emerg_kwh, plan_cost_yuan=plan_cost, emerg_cost_yuan=emerg_cost,
            total_yuan=plan_cost + emerg_cost,
        )
        print(f"  {ds}: b(10/12/14/16/18/20) = "
              + "/".join(f"{b_slots[k]:.2f}" for k in slots_idx)
              + f" | total_b={b_total:.0f} emerg={emerg_kwh:.0f}kWh "
              + f"plan¥{plan_cost:.0f} emerg¥{emerg_cost:.0f} total¥{plan_cost+emerg_cost:.0f}")

    # ---- 落盘 ----
    out = dict(
        主方案="V_dow_N (β=0.70/0.75, W=7, γ=1.0, N=4)",
        总费用=tot,
        计划费=plan, 紧急费=emerg,
        Δ_vs_V_dow=tot - V_DOW_TOTAL,
        Δ_vs_期望=tot - V_DOW_N_TOTAL,
        月度=dict({str(m): {k: v for k, v in val.items() if isinstance(v, (int, float, str))}
                   for m, val in monthly.items()}),
        风险=dict(
            P50_emerg=float(np.quantile(de_arr/1e4, 0.50)),
            P90_emerg=float(np.quantile(de_arr/1e4, 0.90)),
            P95_emerg=float(np.quantile(de_arr/1e4, 0.95)),
            P99_emerg=float(np.quantile(de_arr/1e4, 0.99)),
            max_daily_emerg=float(de_arr.max()/1e4),
            big_emerg_days=int((dk[idx] > 1000).sum()),
            n_eval_days=len(idx),
            touch_bottom_days=int((soc.min(axis=1) <= E_MIN+1).sum()),
            touch_top_days=int((soc.max(axis=1) >= E_MAX-1).sum()),
            mean_start_soc=float(soc[:, 0].mean()),
            mean_min_soc=float(soc.min(axis=1).mean()),
            mean_max_soc=float(soc.max(axis=1).mean()),
            end_soc=float(soc[-1, -1]),
            total_curtail_kwh=float(dw[idx].sum()),
        ),
        指定日期=spec_table,
    )
    (BASE / "results" / "q2_V_dow_N_profile.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_V_dow_N_profile.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
