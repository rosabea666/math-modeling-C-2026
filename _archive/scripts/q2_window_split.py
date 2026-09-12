# -*- coding: utf-8 -*-
r"""
§22 检验 §21.8 的第一条限制：$W=7$ 是"可行下界"还是真的最优？（只读，不改模型）
================================================================================
§21 的联合选参在 **10/11 个月**把历史窗口顶在 $W=7$ 上。而 `forecast_custom` 有一道守卫：
历史不足 7 天时退回参考曲线（`net_ref`），**所以 $W<7$ 根本没法直接测**。
⇒ "越短越好"的趋势是在**可行域边界**被截断的，真正的最优可能落在 $W<7$。

本轮把"窗口"**拆成两个**（这正是 §18 诊断所指向的方向："基准偏高就是储备本身"）：
  · **水平窗 $W_b$**：基准（中位）只看最近 $W_b$ 天 ⇒ 决定基线跟当期水平的贴合速度；
  · **储备窗 $W_r$**：上尾分位数（＝储备）由最近 $W_r$ 天的**滚动残差**构成。
残差按"当天之前 $W_b$ 天"的基线逐日滚动计算（**严格因果，无未来信息**）：
      base(d) = median(net[d-Wb : d])
      resid_k  = net[d-k] - median(net[d-k-Wb : d-k]),  k = 1..Wr
      N̂(d)    = base(d) + max(quantile(resid, β), 0)

⇒ $(W_b, W_r)$ 把 §21 的单一 $W$ 泛化了：$W_b$ 与 $W_r$ 都等于 $W$ 时，它与原估计量
**接近但不相同**（残差的参照基线不同），故**不能**拿 §21 的 V4 直接与本轮 arm B 比。

**两个 arm，同口径的因果选参**（期末一律固定为 V4 选出的 `geq 2400`，$W$ 只影响预测）：
  · **arm A**（原估计量）：β×$W\in\{7,14,21\}$ = 33 候选
      ⇒ 逐位对拍：必须复现 §21 的 **V4 = 1588.9721**（因为 V4 每月都选了 geq2400、W∈{7,14}，
        所以"限定到本 arm"的最小值与全局最小值同点，**应当逐位相等**）。
  · **arm B**（拆分估计量）：β×$W_b\in\{3,5,7,14,28\}$×$W_r\in\{7,14,21\}$ = 165 候选
      ⇒ 这才是本轮的新问题：**把水平窗缩到 7 以下，还能不能继续变好？**
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
from q2_three_way import BETA_SINGLE, betas_single, forecast_custom, _col_quantile  # noqa: E402

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

TERM = ("geq", 2400.0)          # 固定为 §21 的 V4 选择
WB_GRID = [3, 5, 7, 14, 28]
WR_GRID = [7, 14, 21]
W_A = [7, 14, 21]

CAND_A = [(b, "A", w, 0) for b in BETA_SINGLE for w in W_A]
CAND_B = [(b, "B", wb, wr) for b in BETA_SINGLE for wb in WB_GRID for wr in WR_GRID]


def forecast_split(d, D, Wb, Wr, betas):
    """水平窗 Wb ＋ 储备窗 Wr（滚动残差）。严格因果：残差的参照基线也只用当天之前的数据。"""
    net = D["net"]
    if d < 1:
        return D["net_ref"].copy()
    lo = max(0, d - Wb)
    if d - lo < 1:
        return D["net_ref"].copy()
    base = np.median(net[lo:d], axis=0)
    resid = []
    for k in range(1, Wr + 1):
        j = d - k
        if j < 1:
            break
        lj = max(0, j - Wb)
        if j - lj < 1:
            continue
        resid.append(net[j] - np.median(net[lj:j], axis=0))
    if len(resid) < 3:
        return base.copy()
    q = _col_quantile(np.vstack(resid), betas)
    return base + np.maximum(q, 0.0)


def make_b(N_hat, price, E, term):
    kind, floor = term
    sol = solve_lp(price, N_hat, E, mode="plan", terminal=kind, e_end_min=floor)
    return sol["b"], sol["c"]


def replay(D, cand_of_day, start_idx, E_start, keep=False, term=TERM):
    """连续回放。cand_of_day: d -> (beta, est, Wb, Wr)。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = nd and (dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)),
                       Wsp=np.zeros((nd, T)), SOC=np.zeros((nd, T + 1))) if keep else None)
    for d in range(start_idx, nd):
        beta, est, wb, wr = cand_of_day(d)
        if est == "A":
            N_hat = forecast_custom(d, D, wb, betas_single(beta))
        else:
            N_hat = forecast_split(d, D, wb, wr, betas_single(beta))
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


def label(c):
    b_, est, p1, p2 = c
    return f"β={b_:.2f} {'W=%d' % p1 if est == 'A' else 'Wb=%d,Wr=%d' % (p1, p2)}"


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates = D["price"], D["dates"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    month_end = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§22 历史窗口能否再短：把 W 拆成 水平窗 Wb ＋ 储备窗 Wr（只读）")
    print("=" * 104)
    print(f"  期末规则固定为 V4 选出的 {TERM[0]} {TERM[1]:.0f} kWh（§21 显示 L 对结论不敏感）")
    print(f"  arm A（原估计量）  {len(CAND_A):>3d} 个候选 = β {len(BETA_SINGLE)} × W {W_A}")
    print(f"  arm B（拆分估计量）{len(CAND_B):>3d} 个候选 = β {len(BETA_SINGLE)} × Wb {WB_GRID} × Wr {WR_GRID}")
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）；"
          f"公共预热 E = {E_FEB1:.0f} kWh")

    print("\n" + "-" * 104)
    print("1. 口径校验（固定 β 的全年回放，对拍已发表数字）")
    print("-" * 104)
    rV0 = replay(D, lambda d: (0.70, "A", 14, 0), WARMUP_DAY, E_FEB1,
                 term=("cycle", None))
    cV0 = float((rV0["dp"][idx] + rV0["de"][idx]).sum())
    rPH = replay(D, lambda d: (0.90, "A", 7, 0), WARMUP_DAY, E_FEB1)
    cPH = float((rPH["dp"][idx] + rPH["de"][idx]).sum())
    print(f"  β=0.70, W=14, cycle      → {cV0/1e4:>9.4f} 万元   对拍 V0        = 1719.7084  差 {(cV0/1e4-1719.7084):+.4f}")
    print(f"  β=0.90, W=7, geq2400     → {cPH/1e4:>9.4f} 万元   对拍 §21 事后界 = 1580.2007  差 {(cPH/1e4-1580.2007):+.4f}")

    print("\n" + "-" * 104)
    print("2. [因果·可实施] 两个 arm 各自的逐月选参（同一协议）")
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
        print(f"  {'月':>4}   选出的预测参数")
        for m in range(2, 13):
            print(f"  {m:>4}   {label(cands[sched[m]])}")
        print(f"  ⇒ arm {arm} 评价期 = {tot/1e4:.4f} 万元")

    print("\n" + "-" * 104)
    print("3. 关键判定")
    print("-" * 104)
    ca, cb = res["A"]["tot"], res["B"]["tot"]
    print(f"  arm A（原估计量，W≥7）        = {ca/1e4:>9.4f} 万元   对拍 §21 V4 = 1588.9721"
          f"  差 {(ca/1e4-1588.9721):+.4f}  ← 应 ≈0")
    print(f"  arm B（拆分估计量，Wb 可 <7） = {cb/1e4:>9.4f} 万元")
    print(f"  ⇒ 拆分估计量相对原始估计量 = {(cb-ca)/1e4:+.4f} 万元"
          f"（{'更优' if cb < ca else '更差'}）")
    wb_used = [res["B"]["cands"][res["B"]["sched"][m]][2] for m in range(2, 13)]
    wr_used = [res["B"]["cands"][res["B"]["sched"][m]][3] for m in range(2, 13)]
    print(f"  arm B 逐月选到的 Wb：{wb_used}")
    print(f"  arm B 逐月选到的 Wr：{wr_used}")
    print(f"  ⇒ Wb<7 的月份数 = {sum(1 for x in wb_used if x < 7)}/11"
          f"；Wb=7 的月份数 = {sum(1 for x in wb_used if x == 7)}/11")

    print("\n" + "-" * 104)
    print("4. 风险与终端估值（win 的一方 vs arm A）")
    print("-" * 104)
    rows = {"arm A（原估计量）": res["A"]["r"], "arm B（拆分估计量）": res["B"]["r"]}
    print(f"  {'版本':<22}{'计划/万元':>11}{'紧急/万元':>11}{'总费用/万元':>13}"
          f"{'大紧急天数':>11}{'日紧急P99':>11}{'单日最坏':>10}{'日均起始SOC':>13}{'年末SOC':>10}")
    for nm, r in rows.items():
        p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum())
        k = risk(r, idx)
        print(f"  {nm:<22}{p/1e4:>11.4f}{e/1e4:>11.4f}{(p+e)/1e4:>13.4f}"
              f"{k['大紧急天数']:>11d}{k['日紧急P99']:>11.4f}{k['单日最坏']:>10.4f}"
              f"{k['日均起始SOC']:>13.1f}{k['年末SOC']:>10.1f}")
    lam = [float(price.min()), float(price.mean()), float(price.max())]
    print(f"\n  λ ∈ [{lam[0]:.4f}, {lam[2]:.4f}] 元/kWh（中值 {lam[1]:.4f}）；η_c = {ETA_C}")
    for nm, r in rows.items():
        p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum()); tot = p + e
        dE = max(0.0, E_FEB1 - float(r["det"]["SOC"][idx][-1, -1]))
        a = [tot / 1e4 + l * dE / ETA_C / 1e4 for l in lam]
        print(f"  {nm:<22} 库存缺口 {dE:>8.1f} kWh ⇒ 原 {tot/1e4:.4f}｜"
              f"λ低 {a[0]:.4f}｜λ中 {a[1]:.4f}｜λ高 {a[2]:.4f}")

    print("\n" + "-" * 104)
    print("5. 事后最优（不可实施，仅界定空间）：拆分估计量在各 (Wb,Wr) 上的最好 β")
    print("-" * 104)
    for wb in [3, 5, 7]:
        for wr in [7, 14, 21]:
            vals = {}
            for b_ in BETA_SINGLE:
                r = replay(D, lambda d, b_=b_, wb=wb, wr=wr: (b_, "B", wb, wr),
                           WARMUP_DAY, E_FEB1)
                vals[b_] = float((r["dp"][idx] + r["de"][idx]).sum())
            bb = min(vals, key=vals.get)
            print(f"  Wb={wb:<3} Wr={wr:<3} 事后最优 β={bb:.2f} → {vals[bb]/1e4:>9.4f} 万元"
                  f"（相对 V0 {(vals[bb]-cV0)/1e4:>+9.4f}）")

    out = {"口径": {"TERM": list(TERM), "WB_GRID": WB_GRID, "WR_GRID": WR_GRID,
                    "WN": W_A, "评价期天数": int(len(idx)), "公共预热": E_FEB1},
           "校验": {"V0": cV0 / 1e4, "posthoc_W7": cPH / 1e4},
           "armA": {"total": ca / 1e4,
                    "sched": {str(m): label(CAND_A[res["A"]["sched"][m]]) for m in range(2, 13)}},
           "armB": {"total": cb / 1e4,
                    "sched": {str(m): label(CAND_B[res["B"]["sched"][m]]) for m in range(2, 13)},
                    "Wb": wb_used, "Wr": wr_used}}
    (BASE / "results" / "q2_window_split.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_window_split.json")
    print(f"总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
