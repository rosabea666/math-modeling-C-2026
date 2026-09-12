# -*- coding: utf-8 -*-
r"""
§20 期末下限 L 的**因果可实施化**检验（只读，不改模型、不重出 Excel）
==============================================================================
§17 发现一个便宜杠杆：把日循环的回充水平由 E_MAX 降到约 3240 kWh，可省 **33.9957 万元**
（机制：腾出空间吸收光伏 ⇒ 弃置 −14.2%，紧急费与尾部均不恶化）。但 **L 是在评价期上
事后选出的（post-hoc），不可实施**，§12-8 因此要求"并入时间顺序选参流程后再评估"。

本脚本正是做这件事：把 $(\beta, L)$ **联合**纳入 §14/§16 的同一 walk-forward 协议
（第 m 月只用该月之前的数据、训练费来自连续回放），再看那 34 万元还剩多少。

同时必须补做 §17 漏掉的一项**公平性检查**：降低 L 会让电池库存**逐日下漂**（§17 记录
日首储备 10628 → 5294 kWh），故须按 §10 的终端估值 $\lambda$ 把它折算补回，并与
尾部风险一并报告。

口径（与 §16/§17/§19 逐项一致）：
  · 预测 cquant、W=14、执行 must/max 尽限反馈（固定不变）；
  · 训练回放连续（跨日 SOC 传递），起点 E_INIT=6000；
  · 评价期 2/1–12/31、公共预热 E = 10800；评价期**只用于报告**；
  · 目标即完整执行后的 J = Σp·b + 5Σp·e，**不按预测误差、也不按紧急电量单独选参**。

候选：$\beta \in$ BETA_SINGLE（11 个）× 期末规则 ∈ {cycle(E_H=E_0), geq(L), L 取 5 个}
      = 66 个候选。其中"期末恒取 cycle"那一列即 §16 的 V1，用于内部对拍（应复现 −51.8867）。
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

from q2_model import (T, MULT, E_MIN, E_MAX, E_INIT, E_FEB1, WARMUP_DAY,  # noqa: E402
                      ETA_C, Policy, load_data, solve_lp, execute)
from q2_three_way import BETA_SINGLE, betas_single, forecast_custom  # noqa: E402

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")
W = 14
L_GRID = [1200.0, 3240.0, 6480.0, 9720.0, 10800.0]
TERMS = [("cycle", None)] + [("geq", L) for L in L_GRID]
CANDS = [(b, t) for b in BETA_SINGLE for t in TERMS]


def make_b(N_hat, price, E, term):
    """按 (期末规则) 生成当日计划 b 与充电意图 c。"""
    kind, floor = term
    sol = solve_lp(price, N_hat, E, mode="plan", terminal=kind, e_end_min=floor)
    return sol["b"], sol["c"]


def replay(D, cand_of_day, start_idx, E_start, keep=False):
    """连续回放。cand_of_day: d -> (beta, term)。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    s0 = np.zeros(nd); s1 = np.zeros(nd)
    det = nd and (dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)),
                       Wsp=np.zeros((nd, T)), SOC=np.zeros((nd, T + 1))) if keep else None)
    for d in range(start_idx, nd):
        beta, term = cand_of_day(d)
        N_hat = forecast_custom(d, D, W, betas_single(beta))
        b, c_plan = make_b(N_hat, price, E, term)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        s0[d] = E; s1[d] = Etraj[-1]
        if keep:
            det["B"][d] = b; det["EM"][d] = e; det["Wsp"][d] = w; det["SOC"][d] = Etraj
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, s0=s0, s1=s1, det=det)


def risk(R, idx):
    """尾部与库存风险指标。"""
    daily = R["de"][idx] / 1e4                      # 日紧急费（万元）
    soc = R["det"]["SOC"][idx]
    # 注意：dk 是"每日紧急电量合计"（长度 = 天数），不能再按 T 重排。
    return dict(大紧急天数=int((R["dk"][idx] > 1000).sum()),
                日紧急P99=float(np.quantile(daily, 0.99)),
                单日最坏=float(daily.max()),
                日均起始SOC=float(soc[:, 0].mean()),
                触底天数=int((soc.min(axis=1) <= E_MIN + 1e-3).sum()),
                年末SOC=float(soc[-1, -1]))


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    month_end = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]
    loc = {m: np.where(months[idx] == m)[0] for m in range(2, 13)}

    print("=" * 104)
    print("§20 期末下限 L 的因果可实施化检验（(β, L) 联合 walk-forward 选参）")
    print("=" * 104)
    print(f"  候选 {len(CANDS)} 个 = β {len(BETA_SINGLE)} 个 × 期末规则 {len(TERMS)} 个"
          f"（cycle，或 geq：L ∈ {L_GRID}）")
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）；"
          f"公共预热 E = {E_FEB1:.0f} kWh；W = {W}")
    print("  纪律：训练费来自连续回放；第 m 月只用该月之前的数据；评价期只用于报告。")

    # ---------- 1. 每个候选一条全年连续回放（供 walk-forward 选参） ----------
    print("\n" + "-" * 104)
    print("1. 逐候选训练回放（从 1/1、E=6000 起跑，取累计 J 前缀）")
    print("-" * 104)
    cum = {}
    for k, (b_, t_) in enumerate(CANDS):
        r = replay(D, lambda d, b_=b_, t_=t_: (b_, t_), 0, E_INIT)
        cum[k] = np.cumsum(r["dp"] + r["de"])
        if (k + 1) % 12 == 0:
            print(f"    已完成 {k+1}/{len(CANDS)} 个候选（{time.perf_counter()-t_all:.0f}s）")

    # 内部对拍：只用 cycle 的候选应复现 V1
    cyc = [k for k, (b_, t_) in enumerate(CANDS) if t_[0] == "cycle"]
    sched_cyc = {}
    for m in range(2, 13):
        te = month_end[m - 1]
        sched_cyc[m] = CANDS[cyc[min(range(len(cyc)), key=lambda i: cum[cyc[i]][te])]][0]
    r_cyc = replay(D, lambda d: (sched_cyc[months[d]], ("cycle", None)),
                   WARMUP_DAY, E_FEB1, keep=True)
    c_cyc = float((r_cyc["dp"][idx] + r_cyc["de"][idx]).sum())
    print(f"\n  [内部对拍] 仅用 cycle 的因果日程 → {c_cyc/1e4:.4f} 万元"
          f"（§16 的 V1 = 1667.8217，差 {(c_cyc/1e4-1667.8217):+.4f}）")

    # ---------- 2. walk-forward 选出 (β, L) 日程 ----------
    print("\n" + "-" * 104)
    print("2. [因果·可实施] 逐月 (β, L) 日程（第 m 月只用该月之前的数据）")
    print("-" * 104)
    sched = {}
    for m in range(2, 13):
        te = month_end[m - 1]
        k = min(range(len(CANDS)), key=lambda i: cum[i][te])
        sched[m] = k
    print(f"  {'月':>4}{'β':>8}{'期末规则':>14}{'L/kWh':>10}   训练至")
    for m in range(2, 13):
        b_, t_ = CANDS[sched[m]]
        print(f"  {m:>4}{b_:>8.2f}{t_[0]:>14}{('—' if t_[1] is None else '%.0f' % t_[1]):>10}"
              f"   {str(dates[month_end[m-1]])[:10]}")
    nlow = sum(1 for m in range(2, 13) if CANDS[sched[m]][1][0] == "geq"
               and CANDS[sched[m]][1][1] is not None and CANDS[sched[m]][1][1] <= 3240)
    ngeq = sum(1 for m in range(2, 13) if CANDS[sched[m]][1][0] == "geq")
    print(f"  ⇒ 11 个月中，选到 geq（不要求回满）的有 {ngeq} 个；选到低下限 (L ≤ 3240) 的有 {nlow} 个。")

    # ---------- 3. 评价期结果 ----------
    print("\n" + "-" * 104)
    print("3. 评价期结果（公共预热 2/1 起评 E=10800；必须复现 V0=1719.7084）")
    print("-" * 104)
    rV0 = replay(D, lambda d: (0.70, ("cycle", None)), WARMUP_DAY, E_FEB1, keep=True)
    rSel = replay(D, lambda d: CANDS[sched[months[d]]], WARMUP_DAY, E_FEB1, keep=True)

    rows = {}
    for nm, r in (("V0 原 S1 (β=0.70,cycle)", rV0),
                  ("V1 因果 β (cycle)", r_cyc),
                  ("§20 因果 (β,L)", rSel)):
        p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum())
        rows[nm] = dict(plan=p, em=e, total=p + e, emk=float(r["dk"][idx].sum()),
                        wsp=float(r["dw"][idx].sum()), **risk(r, idx))
    print(f"\n  {'版本':<26}{'计划/万元':>11}{'紧急/万元':>11}{'总费用/万元':>13}"
          f"{'紧急电量/kWh':>14}{'弃置/kWh':>12}")
    for nm, d in rows.items():
        print(f"  {nm:<26}{d['plan']/1e4:>11.4f}{d['em']/1e4:>11.4f}{d['total']/1e4:>13.4f}"
              f"{d['emk']:>14.1f}{d['wsp']:>12.1f}")
    p0 = rows["V0 原 S1 (β=0.70,cycle)"]["total"]
    print("\n  相对 V0 的差额：")
    for nm, d in rows.items():
        print(f"    {nm:<26} {d['plan']/1e4-p0/1e4+d['em']/1e4:>+10.4f} 万元"
              f"（计划 {(d['plan']-rows['V0 原 S1 (β=0.70,cycle)']['plan'])/1e4:>+8.4f}，"
              f"紧急 {(d['em']-rows['V0 原 S1 (β=0.70,cycle)']['em'])/1e4:>+8.4f}）")

    print("\n  尾部与库存风险：")
    print(f"  {'版本':<26}{'大紧急天数':>11}{'日紧急P99':>11}{'单日最坏':>10}"
          f"{'日均起始SOC':>13}{'触底天数':>10}{'年末SOC':>10}")
    for nm, d in rows.items():
        print(f"  {nm:<26}{d['大紧急天数']:>11d}{d['日紧急P99']:>11.4f}{d['单日最坏']:>10.4f}"
              f"{d['日均起始SOC']:>13.1f}{d['触底天数']:>10d}{d['年末SOC']:>10.1f}")

    # ---------- 3b. 归因：把 Δ 拆成"L 本身"与"L 之下 β 重选" ----------
    print("\n" + "-" * 104)
    print("3b. 归因：V1 → §20 的 −42.8404 万元里，多少是 L 本身、多少是 β 重选")
    print("-" * 104)
    cSel = rows["§20 因果 (β,L)"]["total"]
    att = {}
    for tag, term in (("B 同一 β 日程 + geq 3240（只换期末规则）", ("geq", 3240.0)),
                      ("  参考 同一 β 日程 + geq 1200（完全放开）", ("geq", 1200.0))):
        r_a = replay(D, lambda d, term=term: (sched_cyc[months[d]], term),
                     WARMUP_DAY, E_FEB1, keep=True)
        att[tag] = float((r_a["dp"][idx] + r_a["de"][idx]).sum())
    v1 = rows["V1 因果 β (cycle)"]["total"]
    print(f"  A = V1（cycle β 日程 + cycle）              = {v1/1e4:.4f} 万元")
    print(f"  B = 同一 β 日程 + geq 3240（只换期末规则）   = "
          f"{att['B 同一 β 日程 + geq 3240（只换期末规则）']/1e4:.4f} 万元"
          f"   Δ(L 本身)      = {(att['B 同一 β 日程 + geq 3240（只换期末规则）']-v1)/1e4:+.4f}")
    print(f"  C = §20 因果 (β,L) 联合日程                = {cSel/1e4:.4f} 万元"
          f"   Δ(L 下 β 重选) = {(cSel-att['B 同一 β 日程 + geq 3240（只换期末规则）'])/1e4:+.4f}")
    print(f"  合计 vs V1 = {(cSel-v1)/1e4:+.4f} 万元；"
          f"对照 L=1200 相对 V1 = {(att['  参考 同一 β 日程 + geq 1200（完全放开）']-v1)/1e4:+.4f} 万元"
          f"（3240 优于完全放开 {(att['B 同一 β 日程 + geq 3240（只换期末规则）']-att['  参考 同一 β 日程 + geq 1200（完全放开）'])/1e4:+.4f}）")

    # ---------- 4. 终端估值折算（§10） ----------
    print("\n" + "-" * 104)
    print("4. 公平性：终端储能估值折算（低 L 会让库存下漂，必须补回）")
    print("-" * 104)
    lam = [float(price.min()), float(price.mean()), float(price.max())]
    print(f"  λ ∈ [{lam[0]:.4f}, {lam[2]:.4f}] 元/kWh（中值 {lam[1]:.4f}）；η_c = {ETA_C}")
    print(f"  {'版本':<26}{'年末SOC':>10}{'库存缺口':>10}{'原总费':>11}{'λ低':>11}{'λ中':>11}{'λ高':>11}")
    adj = {}
    for nm, d in rows.items():
        dE = max(0.0, E_FEB1 - d["年末SOC"])
        a = [d["total"] / 1e4 + l * dE / ETA_C / 1e4 for l in lam]
        adj[nm] = a
        print(f"  {nm:<26}{d['年末SOC']:>10.1f}{dE:>10.1f}{d['total']/1e4:>11.4f}"
              f"{a[0]:>11.4f}{a[1]:>11.4f}{a[2]:>11.4f}")
    print("\n  调整后排序（λ 取中值）：")
    for i, (nm, a) in enumerate(sorted(adj.items(), key=lambda kv: kv[1][1]), 1):
        print(f"    {i}. {nm:<26}{a[1]:>11.4f} 万元")
    print("  ⇒ 若 §20 因果日程在补回库存后仍低于 V1，则该杠杆**可实施且非'吃老本'**。")

    # ---------- 5. 事后最优（不可实施，仅界定空间） ----------
    print("\n" + "-" * 104)
    print("5. 事后最优（在评价期上直接挑，**不可实施**，仅供界定空间）")
    print("-" * 104)
    best = None
    # 只挑"结构上最相关"的几个：cycle 家族与 L=3240 家族
    for nm, term in (("cycle", ("cycle", None)), ("L=3240", ("geq", 3240.0)),
                     ("L=1200", ("geq", 1200.0))):
        vals = {}
        for b_ in BETA_SINGLE:
            r = replay(D, lambda d, b_=b_, term=term: (b_, term), WARMUP_DAY, E_FEB1)
            vals[b_] = float((r["dp"][idx] + r["de"][idx]).sum())
        bb = min(vals, key=vals.get)
        print(f"  期末={nm:<8} 事后最优 β={bb:.2f} → {vals[bb]/1e4:.4f} 万元"
              f"（相对 V0 {(vals[bb]-p0)/1e4:+.4f}）")

    out = {"口径": {"W": W, "候选数": len(CANDS), "L_GRID": L_GRID,
                    "评价期天数": int(len(idx)), "公共预热": E_FEB1},
           "日程": {str(m): {"beta": CANDS[sched[m]][0],
                            "term": CANDS[sched[m]][1][0],
                            "L": CANDS[sched[m]][1][1]} for m in range(2, 13)},
           "结果": {nm: {k: float(v) for k, v in d.items()} for nm, d in rows.items()},
           "调整": {nm: [float(x) for x in a] for nm, a in adj.items()},
           "归因": {k.strip(): float(v / 1e4) for k, v in att.items()},
           "对拍_V1": float(c_cyc / 1e4)}
    (BASE / "results" / "q2_terminal_select.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_terminal_select.json")
    print(f"总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
