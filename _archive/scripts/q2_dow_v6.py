# -*- coding: utf-8 -*-
r"""
§31 V_dow + V6 库存价值反馈执行联合验证（只读，不改模型）
================================================================================
用户 2026-09-12 意见：把 §30 的 V_dow 预测层（γ=1.0, W=7, β∈{0.60, 0.70, 0.75}）
与 §24 V6 的库存价值反馈执行（$E^*$目标水位）联合起来，检验是否还有 1-3 万元。

本脚本：
  · 取 V_dow 的逐月日程（来自 `results/q2_forecast_dow.json`）；
  · 对每个月复用 V6 的 FVI 价值函数（在 SOC 网格上）⇒ 逐时目标水位 $E^*(t)$；
  · 用 V_dow 的预测产生购电序列 $b$，再用 V6 的价值反馈执行（`execute_value`）放/充；
  · 连续回放 334 天，统计总费与分解；
  · 与 V_dow（must/max）、V6（V5 起点 + 价值反馈）对比。

如有效 ⇒ V_dow_v6 为新最佳可实施方案；如无效 ⇒ V_dow 仍为最佳。
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
                      E_MIN, E_MAX, P_MAX, ETA_D,  # noqa: E402
                      Policy, load_data, make_plan, execute)
import q2_forecast_dow as fd          # 预测层
import q2_value_adp as v6             # 价值函数 + 价值反馈执行

POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

V_DOW_TOTAL = 1412.0619
V6_TOTAL = 1581.6655
V5_TOTAL = 1583.7354
V0_TOTAL = 1719.7084


def _month_fvi_pool(month_start, D, lookback=v6.FVI_LOOKBACK):
    return v6.fvi_pool(month_start, D, lookback=lookback)


def replay_v6_exec(D, sched, months, dow_idx, keep=False):
    """V_dow 预测 + V6 价值反馈执行。从 WARMUP_DAY E_FEB1 起评。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    # 逐月预算 (β, W, γ) ⇒ 用 V6 的 FVI 算出 Estar
    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        m_start = np.where(months == m)[0].min()
        scen = _month_fvi_pool(m_start, D)
        V, Estar, Phi = v6.estimate_value(price, scen)
        Estar_of[m] = Estar
        Phi_of[m] = Phi

    E = float(E_FEB1)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = (dict(B=np.zeros((nd, T)), Nhat=np.zeros((nd, T)), EM=np.zeros((nd, T)),
                SOC=np.zeros((nd, T + 1)))
           if keep else None)
    for d in range(WARMUP_DAY, nd):
        beta, W_level, gamma = sched[months[d]]
        N_hat = fd.forecast_dow(d, D, dow_idx, gamma, W_level, beta)
        b, _ = make_plan(POL, price, N_hat, E)
        Estar = Estar_of[months[d]]
        _, _, e, _, Etraj = v6.execute_value(b, net[d], price, E, Estar)
        dp[d] = float((b * price).sum())
        de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum())
        dw[d] = float((np.maximum(b - net[d], 0.0) - np.minimum(e, 0.0)).sum())  # 弃置（近似）
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
                触底天数=int((soc.min(axis=1) <= 1200 + 1e-3).sum()) if soc is not None else -1,
                年末SOC=float(soc[-1, -1]) if soc is not None else float("nan"))


def main():
    t_all = time.perf_counter()
    D = load_data()
    D["_dow"] = fd.dow_index(D["dates"])
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]

    # 加载 V_dow 日程
    js = json.loads((BASE / "results" / "q2_forecast_dow.json").read_text(encoding="utf-8"))
    sched = {int(m): (js["联合日程"][m]["beta"], js["联合日程"][m]["W"],
                       js["联合日程"][m]["gamma"])
             for m in js["联合日程"]}
    print("=" * 104)
    print("§31 V_dow + V6 价值反馈执行联合验证")
    print("=" * 104)
    print(f"  V_dow 日程（来自 §30）：")
    for m in range(2, 13):
        print(f"    m={m:>2}  β={sched[m][0]:.2f}  W={sched[m][1]}  γ={sched[m][2]:.2f}")

    # ---- 0. 自检：V_dow 评价期 = 1412.0619 ----
    print(f"\n[0] 自检 V_dow = 1412.0619 万元")
    rVdow = fd.replay_eval(D, D["_dow"], sched, months, keep=True)
    cVdow = float((rVdow["dp"][idx] + rVdow["de"][idx]).sum()) / 1e4
    print(f"  V_dow 评价期 = {cVdow:.4f} 万元（应 ≈ 1412.0619）")

    # ---- 1. V_dow + V6 价值反馈执行 ----
    print(f"\n[1] V_dow 预测 + V6 价值反馈执行（联合）")
    rVdow_v6 = replay_v6_exec(D, sched, months, D["_dow"], keep=True)
    cVdow_v6 = float((rVdow_v6["dp"][idx] + rVdow_v6["de"][idx]).sum()) / 1e4
    print(f"  V_dow_v6 评价期 = {cVdow_v6:.4f} 万元")
    print(f"  Δ vs V_dow = {cVdow_v6 - cVdow:+.4f} 万元")
    print(f"  Δ vs V6 = {cVdow_v6 - V6_TOTAL:+.4f} 万元")
    print(f"  Δ vs V0 = {cVdow_v6 - V0_TOTAL:+.4f} 万元")

    # ---- 2. 风险 ----
    print(f"\n[2] 风险指标（评价期）")
    s_vdow = summarize("V_dow (must/max)", rVdow, idx)
    s_vdow_v6 = summarize("V_dow + V6 value", rVdow_v6, idx)
    print(f"  {'方案':<24}{'计划':>10}{'紧急':>10}{'总':>10}"
          f"{'大紧急':>7}{'P99':>8}{'日均SOC':>10}{'触底':>6}{'年末SOC':>10}")
    for s in (s_vdow, s_vdow_v6):
        print(f"  {s['name']:<24}{s['plan']:>10.4f}{s['em']:>10.4f}{s['total']:>10.4f}"
              f"{s['大紧急天数']:>7d}{s['日紧急P99']:>8.3f}{s['日均起始SOC']:>10.0f}"
              f"{s['触底天数']:>6d}{s['年末SOC']:>10.0f}")

    # ---- 3. 与已有基线对比 ----
    print(f"\n[3] 与基线对比")
    rows = [
        ("V0",  V0_TOTAL, 0),
        ("V6",  V6_TOTAL, 0),
        ("V_dow (must/max)", V_DOW_TOTAL, 0),
    ]
    print(f"  {'方案':<24}{'总/万元':>11}{'Δ vs V0':>10}")
    for nm, t, _ in rows:
        print(f"  {nm:<24}{t:>11.4f}{t - V0_TOTAL:>+10.4f}")
    print(f"  {'V_dow + V6 value':<24}{cVdow_v6:>11.4f}{cVdow_v6 - V0_TOTAL:>+10.4f}")
    print(f"  ⇒ 联合相对 V_dow（must/max）: {cVdow_v6 - V_DOW_TOTAL:+.4f} 万元  "
          f"({'更优 ✔' if cVdow_v6 < V_DOW_TOTAL else '未赢'})")
    print(f"  ⇒ 联合相对 V6（V5 起点 + 价值反馈）: {cVdow_v6 - V6_TOTAL:+.4f} 万元")
    print(f"  ⇒ 与外部 1356.64 差距: {cVdow_v6 - 1356.64:+.4f} 万元")

    # ---- 落盘 ----
    out = dict(
        V_dow_must_max=cVdow,
        V_dow_v6_value=cVdow_v6,
        delta_vs_V_dow=cVdow_v6 - V_DOW_TOTAL,
        delta_vs_V6=cVdow_v6 - V6_TOTAL,
        delta_vs_V0=cVdow_v6 - V0_TOTAL,
        delta_vs_external=cVdow_v6 - 1356.64,
        风险_V_dow_v6=s_vdow_v6,
        对比=dict(V0=V0_TOTAL, V5=V5_TOTAL, V6=V6_TOTAL, V_dow=V_DOW_TOTAL,
                   V_dow_v6=cVdow_v6),
    )
    (BASE / "results" / "q2_dow_v6.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_dow_v6.json")
    print(f"总耗时 {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()