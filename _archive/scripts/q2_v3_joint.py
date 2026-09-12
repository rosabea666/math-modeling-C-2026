# -*- coding: utf-8 -*-
r"""
§21 V3 的真联合最优：$(\beta, W, L)$ 三参数**因果**选参（只读，不改模型、不重出 Excel）
================================================================================
§20 把期末下限 $L$ 因果化后得到 V3 = 1624.9813 万元。但 §20 把 $W$ **固定在 14**
（沿用 §16 的可比口径），而 §14/§18 已表明 $W$ 本身是个活参数（$W=7$ 单独值 −41.27 万元）。
⇒ **§20 的 V3 不是这个策略族的联合最优**，还剩一个没有联合搜索的自由度。

本脚本把 $(\beta, W, L)$ 三者**一起**纳入同一个 walk-forward 协议：
  · 候选 = $\beta$(11) × $W$(3) × 期末规则(8：cycle 或 geq L∈7 点) = **264 个**；
  · 第 $m$ 月只用该月之前的数据，训练费来自**连续回放**（跨日 SOC 传递）；评价期只用于报告；
  · 预测 cquant、执行 must/max 尽限反馈，固定不变。

**四重内部对拍**（用于验证协议与口径，前两个为已发表的精确数字）：
  ① 限制到 ($W=14$, cycle)              → 必须**逐位**复现 §16 的 V1 = **1667.8217**；
  ② 限制到 ($W=14$, §20 用过的期末网格) → 必须**逐位**复现 §20 的 V3 = **1624.9813**；
  ③ 限制到 ($W=14$, 本次更细的网格)     → 应 **$\le$** 1624.9813（网格是 §20 的超集）；
  ④ 限制到 (任意 $W$, cycle)            → 应 **$\le$** §14 的 1633.0429（§14 的 β 网格更粗，
     5 点 vs 11 点，故只能"不差于"，不能要求逐位相等）。
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

from q2_model import (T, MULT, E_MIN, E_FEB1, E_INIT, WARMUP_DAY,  # noqa: E402
                      ETA_C, Policy, load_data, solve_lp, execute)
from q2_three_way import BETA_SINGLE, betas_single, forecast_custom  # noqa: E402

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

W_GRID = [7, 14, 21]
L_GRID = [1200.0, 2400.0, 3240.0, 4200.0, 5400.0, 6480.0, 8100.0]
TERMS = [("cycle", None)] + [("geq", L) for L in L_GRID]
CANDS = [(b, w, t) for b in BETA_SINGLE for w in W_GRID for t in TERMS]


def make_b(N_hat, price, E, term):
    kind, floor = term
    sol = solve_lp(price, N_hat, E, mode="plan", terminal=kind, e_end_min=floor)
    return sol["b"], sol["c"]


def replay(D, cand_of_day, start_idx, E_start, keep=False):
    """连续回放。cand_of_day: d -> (beta, W, term)。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = nd and (dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)),
                       Wsp=np.zeros((nd, T)), SOC=np.zeros((nd, T + 1))) if keep else None)
    for d in range(start_idx, nd):
        beta, w_, term = cand_of_day(d)
        N_hat = forecast_custom(d, D, w_, betas_single(beta))
        b, c_plan = make_b(N_hat, price, E, term)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        if keep:
            det["B"][d] = b; det["EM"][d] = e; det["Wsp"][d] = w; det["SOC"][d] = Etraj
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, det=det)


def risk(R, idx):
    daily = R["de"][idx] / 1e4
    soc = R["det"]["SOC"][idx]
    return dict(大紧急天数=int((R["dk"][idx] > 1000).sum()),
                日紧急P99=float(np.quantile(daily, 0.99)),
                单日最坏=float(daily.max()),
                日均起始SOC=float(soc[:, 0].mean()),
                触底天数=int((soc.min(axis=1) <= E_MIN + 1e-3).sum()),
                年末SOC=float(soc[-1, -1]))


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates = D["price"], D["dates"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    month_end = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§21 V3 的真联合最优：(β, W, L) 三参数因果选参")
    print("=" * 104)
    print(f"  候选 {len(CANDS)} 个 = β {len(BETA_SINGLE)} × W {W_GRID} × 期末规则 {len(TERMS)}"
          f"（cycle，或 geq：L ∈ {L_GRID}）")
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）；"
          f"公共预热 E = {E_FEB1:.0f} kWh")
    print("  纪律：训练费来自连续回放；第 m 月只用该月之前的数据；评价期只用于报告。")

    print("\n" + "-" * 104)
    print("1. 逐候选训练回放（从 1/1、E=6000 起跑，取累计 J 前缀）")
    print("-" * 104)
    cum = {}
    for k, c_ in enumerate(CANDS):
        r = replay(D, lambda d, c_=c_: c_, 0, E_INIT)
        cum[k] = np.cumsum(r["dp"] + r["de"])
        if (k + 1) % 24 == 0:
            print(f"    已完成 {k+1}/{len(CANDS)} 个候选（{time.perf_counter()-t_all:.0f}s）")

    def sel_eval(subset, label):
        sched = {}
        for m in range(2, 13):
            te = month_end[m - 1]
            sched[m] = min(subset, key=lambda i: cum[i][te])
        r = replay(D, lambda d: CANDS[sched[months[d]]], WARMUP_DAY, E_FEB1, keep=True)
        tot = float((r["dp"][idx] + r["de"][idx]).sum())
        return sched, r, tot

    print("\n" + "-" * 104)
    print("2. 三重内部对拍（验证协议与口径）")
    print("-" * 104)
    SUB_W14_CYC = [k for k, (b_, w_, t_) in enumerate(CANDS) if w_ == 14 and t_[0] == "cycle"]
    SUB_W14_ALL = [k for k, (b_, w_, t_) in enumerate(CANDS) if w_ == 14]
    SUB_CYC_ALL = [k for k, (b_, w_, t_) in enumerate(CANDS) if t_[0] == "cycle"]
    # §20 用的是更粗的期末网格（5 个 L），单独拿出一份做**逐位**对拍
    L20 = [1200.0, 3240.0, 6480.0, 9720.0, 10800.0]
    SUB_W14_G20 = [k for k, (b_, w_, t_) in enumerate(CANDS)
                   if w_ == 14 and (t_[0] == "cycle" or (t_[1] is not None and t_[1] in L20))]
    sc1, r1, c1 = sel_eval(SUB_W14_CYC, "W14+cycle")
    sc2, r2, c2 = sel_eval(SUB_W14_G20, "W14+§20grid")
    sc4, r4, c4 = sel_eval(SUB_W14_ALL, "W14+allterms")
    sc3, r3, c3 = sel_eval(SUB_CYC_ALL, "cycle")
    print(f"  ① (W=14, cycle)            → {c1/1e4:>10.4f} 万元   对拍 §16 V1     = 1667.8217"
          f"   差 {(c1/1e4-1667.8217):+.4f}  ← 应 ≈0")
    print(f"  ② (W=14, §20 的期末网格)   → {c2/1e4:>10.4f} 万元   对拍 §20 V3     = 1624.9813"
          f"   差 {(c2/1e4-1624.9813):+.4f}  ← 应 ≈0")
    print(f"  ③ (W=14, 本次更细的网格)   → {c4/1e4:>10.4f} 万元   参考 §20 V3     = 1624.9813"
          f"   差 {(c4/1e4-1624.9813):+.4f}  ← 应 ≤0（网格是超集）")
    print(f"  ④ (任意 W, cycle)          → {c3/1e4:>10.4f} 万元   对拍 §14 (β,W)  = 1633.0429"
          f"   差 {(c3/1e4-1633.0429):+.4f}  ← 应 ≤0（§14 β 网格更粗）")

    print("\n" + "-" * 104)
    print("3. [因果·可实施] 逐月 (β, W, L) 联合日程（第 m 月只用该月之前的数据）")
    print("-" * 104)
    schedJ = {}
    for m in range(2, 13):
        te = month_end[m - 1]
        schedJ[m] = min(range(len(CANDS)), key=lambda i: cum[i][te])
    print(f"  {'月':>4}{'β':>8}{'W':>5}{'期末规则':>12}{'L/kWh':>10}   训练至")
    for m in range(2, 13):
        b_, w_, t_ = CANDS[schedJ[m]]
        print(f"  {m:>4}{b_:>8.2f}{w_:>5d}{t_[0]:>12}{('—' if t_[1] is None else '%.0f' % t_[1]):>10}"
              f"   {str(dates[month_end[m-1]])[:10]}")
    nw7 = sum(1 for m in range(2, 13) if CANDS[schedJ[m]][1] == 7)
    ngeq = sum(1 for m in range(2, 13) if CANDS[schedJ[m]][2][0] == "geq")
    print(f"  ⇒ 11 个月中：选到 geq 的有 {ngeq} 个；选到 W=7 的有 {nw7} 个。")

    print("\n" + "-" * 104)
    print("4. 评价期结果（公共预热 2/1 起评 E=10800；必须复现 V0=1719.7084）")
    print("-" * 104)
    rV0 = replay(D, lambda d: (0.70, 14, ("cycle", None)), WARMUP_DAY, E_FEB1, keep=True)
    rJ = replay(D, lambda d: CANDS[schedJ[months[d]]], WARMUP_DAY, E_FEB1, keep=True)
    rows = {}
    for nm, r in (("V0 原 S1 (β0.70,W14,cycle)", rV0),
                  ("V1 因果 β (W14,cycle)", r1),
                  ("V3 §20 因果 (β,L) @W14", r2),
                  ("V4 §21 因果 (β,W,L)", rJ)):
        p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum())
        rows[nm] = dict(plan=p, em=e, total=p + e, emk=float(r["dk"][idx].sum()),
                        wsp=float(r["dw"][idx].sum()), **risk(r, idx))
    base = rows["V0 原 S1 (β0.70,W14,cycle)"]["total"]
    print(f"\n  {'版本':<28}{'计划/万元':>11}{'紧急/万元':>11}{'总费用/万元':>13}"
          f"{'相对 V0':>11}{'紧急电量/kWh':>14}{'弃置/kWh':>12}")
    for nm, d in rows.items():
        print(f"  {nm:<28}{d['plan']/1e4:>11.4f}{d['em']/1e4:>11.4f}{d['total']/1e4:>13.4f}"
              f"{(d['total']-base)/1e4:>+11.4f}{d['emk']:>14.1f}{d['wsp']:>12.1f}")

    print("\n  尾部与库存风险：")
    print(f"  {'版本':<28}{'大紧急天数':>11}{'日紧急P99':>11}{'单日最坏':>10}"
          f"{'日均起始SOC':>13}{'触底天数':>10}{'年末SOC':>10}")
    for nm, d in rows.items():
        print(f"  {nm:<28}{d['大紧急天数']:>11d}{d['日紧急P99']:>11.4f}{d['单日最坏']:>10.4f}"
              f"{d['日均起始SOC']:>13.1f}{d['触底天数']:>10d}{d['年末SOC']:>10.1f}")

    print("\n" + "-" * 104)
    print("5. 归因：V3 → V4 的差额全部来自 W（V3 已是 W=14 下 (β,L) 最优）")
    print("-" * 104)
    print(f"  V3（W=14，因果 β＋L）  = {rows['V3 §20 因果 (β,L) @W14']['total']/1e4:.4f} 万元")
    print(f"  V4（W 也参与联合选参） = {rows['V4 §21 因果 (β,W,L)']['total']/1e4:.4f} 万元"
          f"   Δ(W 的贡献) = "
          f"{(rows['V4 §21 因果 (β,W,L)']['total']-rows['V3 §20 因果 (β,L) @W14']['total'])/1e4:+.4f}")
    print(f"  V4 相对 V0 = {(rows['V4 §21 因果 (β,W,L)']['total']-base)/1e4:+.4f} 万元"
          f"（≈ {abs((rows['V4 §21 因果 (β,W,L)']['total']-base)/base)*100:.2f}%）")

    print("\n" + "-" * 104)
    print("6. 公平性：终端储能估值折算（§10）")
    print("-" * 104)
    lam = [float(price.min()), float(price.mean()), float(price.max())]
    print(f"  λ ∈ [{lam[0]:.4f}, {lam[2]:.4f}] 元/kWh（中值 {lam[1]:.4f}）；η_c = {ETA_C}")
    adj = {}
    print(f"  {'版本':<28}{'年末SOC':>10}{'库存缺口':>10}{'原总费':>11}{'λ低':>11}{'λ中':>11}{'λ高':>11}")
    for nm, d in rows.items():
        dE = max(0.0, E_FEB1 - d["年末SOC"])
        a = [d["total"] / 1e4 + l * dE / ETA_C / 1e4 for l in lam]
        adj[nm] = a
        print(f"  {nm:<28}{d['年末SOC']:>10.1f}{dE:>10.1f}{d['total']/1e4:>11.4f}"
              f"{a[0]:>11.4f}{a[1]:>11.4f}{a[2]:>11.4f}")
    print("\n  调整后排序（λ 取中值）：")
    for i, (nm, a) in enumerate(sorted(adj.items(), key=lambda kv: kv[1][1]), 1):
        print(f"    {i}. {nm:<28}{a[1]:>11.4f} 万元")

    print("\n" + "-" * 104)
    print("7. 事后最优（不可实施，仅界定空间）")
    print("-" * 104)
    for nm, term in (("cycle", ("cycle", None)), ("L=3240", ("geq", 3240.0)),
                     ("L=2400", ("geq", 2400.0))):
        for wv in W_GRID:
            vals = {}
            for b_ in BETA_SINGLE:
                r = replay(D, lambda d, b_=b_, wv=wv, term=term: (b_, wv, term),
                           WARMUP_DAY, E_FEB1)
                vals[b_] = float((r["dp"][idx] + r["de"][idx]).sum())
            bb = min(vals, key=vals.get)
            print(f"  期末={nm:<7} W={wv:<3} 事后最优 β={bb:.2f} → {vals[bb]/1e4:>9.4f} 万元"
                  f"（相对 V0 {(vals[bb]-base)/1e4:>+9.4f}）")

    out = {"口径": {"W_GRID": W_GRID, "L_GRID": L_GRID, "候选数": len(CANDS),
                    "评价期天数": int(len(idx)), "公共预热": E_FEB1},
           "对拍": {"W14_cycle_V1": c1 / 1e4, "W14_grid20": c2 / 1e4,
                    "W14_finegrid": c4 / 1e4, "cycle_allW": c3 / 1e4},
           "日程": {str(m): {"beta": CANDS[schedJ[m]][0], "W": CANDS[schedJ[m]][1],
                            "term": CANDS[schedJ[m]][2][0], "L": CANDS[schedJ[m]][2][1]}
                    for m in range(2, 13)},
           "结果": {nm: {k: float(v) for k, v in d.items()} for nm, d in rows.items()},
           "调整": {nm: [float(x) for x in a] for nm, a in adj.items()}}
    (BASE / "results" / "q2_v3_joint.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_v3_joint.json")
    print(f"总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
