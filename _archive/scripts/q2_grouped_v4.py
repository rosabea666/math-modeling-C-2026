# -*- coding: utf-8 -*-
r"""
§23 分时段 $\beta$ 在 V4 下是否仍然无效？（只读，不改模型）
================================================================================
§15/§16 在 **`cycle` ＋ $W=14$** 下测过分时段 $\beta$（4 组：早峰/日间/晚峰/其他），结论是
"**小幅有效、但增量仅 0.23%**"（V2 = 1663.9087 相对 V1 的 1667.8217 只省 3.9130 万元），
并因此**放弃了分组**。

但 §21 把 $W$ 放开到 7、期末放到 $L=2400$ 之后，**预测的保守程度彻底变了**：
储备窗缩短 ⇒ 储备变小 ⇒ 计划形状变了。⇒ "**分组 $\beta$ 在 V4 下是否仍然无效**"
**从未被检验**（§21 的候选表只有单一 $\beta$）。

本脚本只回答这一个问题，其余全部钉在 V4 的选择上：
  · 期末固定 `geq 2400`，$W\in\{7,14\}$（V4 在这两个值上都选过；$W=21$ 从未胜出，略去）；
  · **arm A（单一 $\beta$）**：11 个 $\beta$ × 2 个 $W$ = 22 候选
      ⇒ 逐位对拍：必须复现 §21 的 **V4 = 1588.9721**
        （V4 每月都选了 `geq2400` 且 $W\in\{7,14\}$ ⇒ 限定到该子集不改变最小值）；
  · **arm B（分时段 $\beta$）**：$\beta$ 4 组各取 $[0.70,0.80,0.90]$ ⇒ $3^4=81$ 组合 × 2 个 $W$ = 162 候选
        （组的划分由**给定电价结构**决定，事前可定义，与评价期无关，见 `q2_three_way.GIDX`）。
两个 arm 跑**同一套** walk-forward 协议，故可比。
"""
from __future__ import annotations

import io
import itertools
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
from q2_three_way import (BETA_SINGLE, LEVELS, GNAMES,  # noqa: E402
                          betas_single, betas_grouped, forecast_custom)

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

TERM = ("geq", 2400.0)          # 固定为 §21 的 V4 选择
W_GRID = [7, 14]
CAND_A = [("S", b, w) for b in BETA_SINGLE for w in W_GRID]
CAND_B = [("G", v4, w) for v4 in itertools.product(LEVELS, repeat=4) for w in W_GRID]


def mk_betas(kind, vec):
    return betas_single(vec) if kind == "S" else betas_grouped(vec)


def label(c):
    kind, vec, w = c
    if kind == "S":
        return f"单一 β={vec:.2f}, W={w}"
    return "分组 [" + ",".join(f"{g}={v:.2f}" for g, v in zip(GNAMES, vec)) + f"], W={w}"


def make_b(N_hat, price, E, term):
    kind, floor = term
    sol = solve_lp(price, N_hat, E, mode="plan", terminal=kind, e_end_min=floor)
    return sol["b"], sol["c"]


def replay(D, cand_of_day, start_idx, E_start, keep=False, term=TERM):
    """连续回放。cand_of_day: d -> (kind, vec, W)。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = nd and (dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)),
                       Wsp=np.zeros((nd, T)), SOC=np.zeros((nd, T + 1))) if keep else None)
    for d in range(start_idx, nd):
        kind, vec, w_ = cand_of_day(d)
        N_hat = forecast_custom(d, D, w_, mk_betas(kind, vec))
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
    print("§23 分时段 β 在 V4 下是否仍然无效？（单一 β vs 4 组分组 β，同一 walk-forward 协议）")
    print("=" * 104)
    print(f"  期末固定 {TERM[0]} {TERM[1]:.0f} kWh；W ∈ {W_GRID}；组划分 = {GNAMES}（由给定电价结构事前确定）")
    print(f"  arm A（单一 β）  {len(CAND_A):>3d} 候选 = β {len(BETA_SINGLE)} × W {len(W_GRID)}")
    print(f"  arm B（分组 β）  {len(CAND_B):>3d} 候选 = {len(LEVELS)}^4={len(LEVELS)**4} 组合 × W {len(W_GRID)}")
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）；"
          f"公共预热 E = {E_FEB1:.0f} kWh")

    print("\n" + "-" * 104)
    print("0. 口径校验（固定参数的全年回放，对拍已发表数字）")
    print("-" * 104)
    rV0 = replay(D, lambda d: ("S", 0.70, 14), WARMUP_DAY, E_FEB1, term=("cycle", None))
    cV0 = float((rV0["dp"][idx] + rV0["de"][idx]).sum())
    print(f"  β=0.70, W=14, cycle  → {cV0/1e4:>9.4f} 万元  对拍 V0 = 1719.7084  差 {(cV0/1e4-1719.7084):+.4f}")
    print("  注：§16 的 V2 是**逐月因果日程**、不是某个固定分组向量，故无法用单次固定回放复现；")
    print("      arm A 对 V4 的逐位复现（见下）承担本轮的协议校验。")

    print("\n" + "-" * 104)
    print("1. [因果·可实施] 两个 arm 各自的逐月选参")
    print("-" * 104)
    res = {}
    for arm, cands in (("A", CAND_A), ("B", CAND_B)):
        cum = {}
        for k, c_ in enumerate(cands):
            r = replay(D, lambda d, c_=c_: c_, 0, E_INIT)
            cum[k] = np.cumsum(r["dp"] + r["de"])
        sched = {}
        for m in range(2, 13):
            te = month_end[m - 1]
            sched[m] = min(range(len(cands)), key=lambda i: cum[i][te])
        r = replay(D, lambda d: cands[sched[months[d]]], WARMUP_DAY, E_FEB1, keep=True)
        tot = float((r["dp"][idx] + r["de"][idx]).sum())
        res[arm] = dict(cands=cands, sched=sched, r=r, tot=tot)
        print(f"\n  --- arm {arm}（{len(cands)} 候选，累计 {time.perf_counter()-t_all:.0f}s）---")
        for m in range(2, 13):
            print(f"  {m:>4}   {label(cands[sched[m]])}")
        print(f"  ⇒ arm {arm} 评价期 = {tot/1e4:.4f} 万元")

    print("\n" + "-" * 104)
    print("2. 关键判定")
    print("-" * 104)
    ca, cb = res["A"]["tot"], res["B"]["tot"]
    print(f"  arm A（单一 β，W∈{W_GRID}，geq2400） = {ca/1e4:>9.4f} 万元   对拍 §21 V4 = 1588.9721"
          f"  差 {(ca/1e4-1588.9721):+.4f}  ← 应 ≈0")
    print(f"  arm B（分组 β，4 组，geq2400）        = {cb/1e4:>9.4f} 万元")
    print(f"  ⇒ 分组 β 相对单一 β = {(cb-ca)/1e4:+.4f} 万元（{'更优' if cb < ca else '更差'}）")
    ng = sum(1 for m in range(2, 13) if res["B"]["cands"][res["B"]["sched"][m]][0] == "G"
             and len(set(res["B"]["cands"][res["B"]["sched"][m]][1])) > 1)
    print(f"  arm B 逐月选到的**真正分组**（组间取值不全相同）月份数 = {ng}/11")

    print("\n" + "-" * 104)
    print("3. 费用分解与风险")
    print("-" * 104)
    rows = {"arm A（单一 β）": res["A"]["r"], "arm B（分组 β）": res["B"]["r"]}
    print(f"  {'版本':<20}{'计划/万元':>11}{'紧急/万元':>11}{'总费用/万元':>13}{'相对 V0':>11}"
          f"{'紧急电量/kWh':>14}{'弃置/kWh':>12}")
    for nm, r in rows.items():
        p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum())
        print(f"  {nm:<20}{p/1e4:>11.4f}{e/1e4:>11.4f}{(p+e)/1e4:>13.4f}"
              f"{((p+e)-cV0)/1e4:>+11.4f}{float(r['dk'][idx].sum()):>14.1f}"
              f"{float(r['dw'][idx].sum()):>12.1f}")
    print(f"\n  {'版本':<20}{'大紧急天数':>11}{'日紧急P99':>11}{'单日最坏':>10}"
          f"{'日均起始SOC':>13}{'触底天数':>10}{'年末SOC':>10}")
    for nm, r in rows.items():
        k = risk(r, idx)
        print(f"  {nm:<20}{k['大紧急天数']:>11d}{k['日紧急P99']:>11.4f}{k['单日最坏']:>10.4f}"
              f"{k['日均起始SOC']:>13.1f}{k['触底天数']:>10d}{k['年末SOC']:>10.1f}")
    lam = [float(price.min()), float(price.mean()), float(price.max())]
    print(f"\n  λ ∈ [{lam[0]:.4f}, {lam[2]:.4f}] 元/kWh（中值 {lam[1]:.4f}）；η_c = {ETA_C}")
    for nm, r in rows.items():
        p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum()); tot = p + e
        dE = max(0.0, E_FEB1 - float(r["det"]["SOC"][idx][-1, -1]))
        a = [tot / 1e4 + l * dE / ETA_C / 1e4 for l in lam]
        print(f"  {nm:<20} 库存缺口 {dE:>8.1f} kWh ⇒ 原 {tot/1e4:.4f}｜"
              f"λ低 {a[0]:.4f}｜λ中 {a[1]:.4f}｜λ高 {a[2]:.4f}")

    print("\n" + "-" * 104)
    print("4. 事后界（不可实施，仅界定空间）：两组结构各自在评价期上的最好固定参数")
    print("-" * 104)
    bestS = None
    for b_ in BETA_SINGLE:
        for w_ in W_GRID:
            r = replay(D, lambda d, b_=b_, w_=w_: ("S", b_, w_), WARMUP_DAY, E_FEB1)
            v = float((r["dp"][idx] + r["de"][idx]).sum())
            if bestS is None or v < bestS[0]:
                bestS = (v, b_, w_)
    print(f"  单一 β 事后最优：β={bestS[1]:.2f}, W={bestS[2]} → {bestS[0]/1e4:.4f} 万元"
          f"（相对 V0 {(bestS[0]/1e4-cV0/1e4):+.4f}）")
    bestG = None
    for v4 in itertools.product(LEVELS, repeat=4):
        for w_ in W_GRID:
            r = replay(D, lambda d, v4=v4, w_=w_: ("G", v4, w_), WARMUP_DAY, E_FEB1)
            v = float((r["dp"][idx] + r["de"][idx]).sum())
            if bestG is None or v < bestG[0]:
                bestG = (v, v4, w_)
    print(f"  分组 β 事后最优：{bestG[1]} , W={bestG[2]} → {bestG[0]/1e4:.4f} 万元"
          f"（相对 V0 {(bestG[0]/1e4-cV0/1e4):+.4f}）")

    out = {"口径": {"TERM": list(TERM), "W_GRID": W_GRID, "LEVELS": LEVELS, "GNAMES": GNAMES,
                    "评价期天数": int(len(idx)), "公共预热": E_FEB1},
           "校验": {"V0": cV0 / 1e4},
           "armA": {"total": ca / 1e4,
                    "sched": {str(m): label(CAND_A[res["A"]["sched"][m]]) for m in range(2, 13)}},
           "armB": {"total": cb / 1e4,
                    "sched": {str(m): label(CAND_B[res["B"]["sched"][m]]) for m in range(2, 13)}},
           "事后界": {"single": bestS[0] / 1e4, "grouped": bestG[0] / 1e4}}
    (BASE / "results" / "q2_grouped_v4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_grouped_v4.json")
    print(f"总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
