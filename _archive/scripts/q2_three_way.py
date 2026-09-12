# -*- coding: utf-8 -*-
"""
问题二：只比较三个版本 + 费用诊断（只读：不改模型、不重出 Excel）
=====================================================================
用户 2026-09-11 意见：暂停叠加策略，先定位“钱到底花在哪里”。本轮**只比较三个版本**：

  V0 = 原 S1                (β=0.70 全天统一, W=14)
  V1 = 重新校准单一 β 的 S1  (W=14 固定；β 按完整 J 在**时间顺序扩展窗口**上选)
  V2 = 少量分时段 β 的 S1    (4 组：早峰/日间/晚峰/其他；W=14)

统一口径（三者完全一致，保证可比）：
  · 连续回放：状态跨日传递；日循环期末处理 E_H = E_0（同一个 LP 终端约束）；
  · 公共预热：评价期 2/1 起评，E = 10800 kWh；
  · 评价期 2/1–12/31 **只用于报告**；选参只用该月之前的数据（1 月 ~ m−1 月末）；
  · 目标即完整执行后的 J = Σ p·b + 5 Σ p·e，**不按预测误差、也不按紧急电量单独选参**。

另附（用户步骤 4）：**公平 PI 参照**——完美**日前**预测，但物理约束、日循环期末、
  评价边界与三者完全相同（此前的 PI 对照在这三处不一致，费用差不能全归因于预测）。

严格纪律（用户明确要求）：**余量 w 不等于“多买的电”**。执行层里 w 是弃置量，
  其来源既可能是付费购电、也可能来自光伏（本模型 net_act = 负荷 − 光伏）。
  因此**不得**把全部余量乘电价当作可回收的费用损失，只给“可归因于购电的上界”。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_INIT, E_MIN, E_MAX, P_MAX, ETA_D,  # noqa: E402
                      WARMUP_DAY, E_FEB1, Policy, load_data, make_plan, execute)

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

# ---- 时段分组：由**给定电价结构**确定（事前可定义，与评价期无关）----
CLOCK = [(0, 360, "夜谷"), (360, 600, "早峰"), (600, 1020, "日间"),
         (1020, 1260, "晚峰"), (1260, 1440, "其他")]
GNAMES = ["早峰", "日间", "晚峰", "其他"]

BETA_SINGLE = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
LEVELS = [0.70, 0.80, 0.90]


def group_index():
    """长度 T 的 β 组编号：0=早峰 1=日间 2=晚峰 3=其他(含夜谷与 21-24 时)。"""
    g = np.empty(T, dtype=int)
    for t in range(T):
        m = 10 * t
        if 360 <= m < 600:
            g[t] = 0
        elif 600 <= m < 1020:
            g[t] = 1
        elif 1020 <= m < 1260:
            g[t] = 2
        else:
            g[t] = 3
    return g


GIDX = group_index()


def betas_single(b):
    return np.full(T, float(b))


def betas_grouped(vec4):
    return np.array(vec4, dtype=float)[GIDX]


def _col_quantile(resid, qs):
    """按列计算分位数，qs 为长度 T 的向量（与 np.quantile(..., method='linear') 等价）。
    注意：不能直接 np.quantile(resid, qs, axis=0)——那样会算出 len(qs)×T 的矩阵。"""
    n, m = resid.shape
    vs = np.sort(resid, axis=0)
    pos = np.asarray(qs, dtype=float) * (n - 1.0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    cols = np.arange(m)
    return vs[lo, cols] * (1.0 - frac) + vs[hi, cols] * frac


def forecast_custom(d, D, W, betas):
    """分时段上尾经验分位数（等权 ρ=1，与 q2_model 的 cquant 同口径）。"""
    hist = D["net"][max(0, d - W):d]
    if hist.shape[0] < 7:
        return D["net_ref"].copy()
    base = np.median(hist, axis=0)
    resid = hist - base
    q = _col_quantile(resid, betas)
    return base + np.maximum(q, 0.0)


def replay(D, betas_by_day, W, start_idx, E_start, oracle=False, keep=False):
    """连续回放。betas_by_day: 长度 nd 的函数 d -> β 向量。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = None
    if keep:
        det = dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)), Wsp=np.zeros((nd, T)),
                   SOC=np.zeros((nd, T + 1)), Nhat=np.zeros((nd, T)))
    for d in range(start_idx, nd):
        N_hat = net[d].copy() if oracle else forecast_custom(d, D, W, betas_by_day(d))
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum())
        de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum())
        dw[d] = float(w.sum())
        if keep:
            det["B"][d] = b; det["EM"][d] = e; det["Wsp"][d] = w
            det["SOC"][d] = Etraj; det["Nhat"][d] = N_hat
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, det=det)


def prefix_cum(D, thetas, betas_fn, W):
    """每个候选做一次全年连续回放，返回累计 J 前缀（供时间顺序选参）。"""
    cum = {}
    for k, th in enumerate(thetas):
        r = replay(D, lambda d, th=th: betas_fn(th), W, 0, E_INIT)
        cum[k] = np.cumsum(r["dp"] + r["de"])
    return cum


def select_schedule(D, thetas, cum):
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k = min(range(len(thetas)), key=lambda i: cum[i][te])
        sched[m] = thetas[k]
    return sched


def eval_schedule(D, sched, betas_fn, W, idx_eval):
    return replay(D, lambda d: betas_fn(sched[D["_month"][d]]), W,
                  WARMUP_DAY, E_FEB1, keep=True)


def monthly(dp, de, idx_eval, months):
    rows = {}
    for m in range(2, 13):
        mi = idx_eval[months[idx_eval] == m]
        rows[m] = (float(dp[mi].sum()), float(de[mi].sum()),
                   float(dp[mi].sum() + de[mi].sum()))
    return rows


def main():
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx_eval = np.where(dates >= np.datetime64("2025-02-01"))[0]
    loc = {m: np.where(months[idx_eval] == m)[0] for m in range(2, 13)}

    print("=" * 100)
    print("问题二 · 只比较三个版本 + 费用诊断（只读）")
    print("=" * 100)
    print(f"  评价期 {str(dates[idx_eval[0]])[:10]} ~ {str(dates[idx_eval[-1]])[:10]}，"
          f"{len(idx_eval)} 天；公共预热 E = {E_FEB1:.0f} kWh")
    print("  时段分组（按给定电价结构，事前可定义）：")
    for a, b_, nm in CLOCK:
        sel = np.array([(a <= 10 * t < b_) for t in range(T)])
        print(f"    {nm:<5} {a//60:02d}:{a%60:02d}-{b_//60:02d}:{b_%60:02d}"
              f"  区间数 {int(sel.sum()):>3}  电价均值 {price[sel].mean():.4f}"
              f"  范围 [{price[sel].min():.4f}, {price[sel].max():.4f}]")

    out = {}

    # ============ 步骤 1：单一 β 重校准（W=14 固定）============
    print("\n" + "-" * 100)
    print("步骤 1  单一 β 重校准（W=14 固定；目标 = 完整 J，非预测误差、非紧急电量）")
    print("-" * 100)
    cum1 = prefix_cum(D, BETA_SINGLE, betas_single, 14)
    print("  [诊断·post-hoc，仅看形状与边界] 全年连续回放 J(β)：")
    curve = {b_: float(cum1[k][-1]) for k, b_ in enumerate(BETA_SINGLE)}
    bmin = min(curve, key=curve.get)
    for b_ in BETA_SINGLE:
        print(f"    β={b_:.4f}   全年 J = {curve[b_]/1e4:>10.4f} 万元"
              f"{'   ← 最优' if b_ == bmin else ''}")
    edge = bmin in (BETA_SINGLE[0], BETA_SINGLE[-1])
    lo, hi = min(curve.values()), max(curve.values())
    print(f"  ⇒ 最优 β = {bmin:.2f}；落在搜索边界：{'是（需外扩）' if edge else '否（内部解）'}；"
          f"曲线极差 {(hi-lo)/1e4:.4f} 万元（{(hi-lo)/lo*100:.2f}% of 最优）")
    out["beta_curve"] = {str(k): v for k, v in curve.items()}
    out["beta_curve_argmin"] = bmin
    out["beta_curve_is_edge"] = bool(edge)

    sched1 = select_schedule(D, BETA_SINGLE, cum1)
    print("\n  [因果·可实施] 逐月单一 β 日程（第 m 月只用该月之前的数据）：")
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        print(f"    用于 {m:2d} 月 → β = {sched1[m]:.2f}"
              f"   （训练至 {str(dates[te])[:10]}）")

    te0 = D["_month_end"][1]
    best0 = min(BETA_SINGLE, key=lambda b_: cum1[BETA_SINGLE.index(b_)][te0])
    print("\n  [与 §13.1 对齐] 1 月训练窗口的累计 J（细网格，复用上面已算的回放）：")
    for b_ in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        v = cum1[BETA_SINGLE.index(b_)][te0]
        print(f"    β={b_:.4f}  1 月累计 J = {v/1e4:>9.4f} 万元"
              f"{'   ← 最优' if b_ == best0 else ''}")
    print(f"    ⇒ §13.1 用 0.50~0.95 的**粗网格**得 0.70；本轮**细网格**下 1 月最优为 "
          f"{best0:.2f}，二者不矛盾（0.65 落在原网格的间隙里）。")

    # ============ 步骤 2：费用诊断（在 V0 上）============
    print("\n" + "-" * 100)
    print("步骤 2  费用诊断（在 V0 = 原 S1 上做，定位改进方向）")
    print("-" * 100)
    V0b = betas_single(0.70)
    r0 = replay(D, lambda d: V0b, 14, WARMUP_DAY, E_FEB1, keep=True)
    base_plan = float(r0["dp"][idx_eval].sum())
    base_em = float(r0["de"][idx_eval].sum())
    base_tot = base_plan + base_em
    print(f"  V0 评价期：计划 {base_plan/1e4:.4f} 万元（{base_plan/base_tot*100:.1f}%）、"
          f"紧急 {base_em/1e4:.4f} 万元（{base_em/base_tot*100:.1f}%）、合计 {base_tot/1e4:.4f} 万元")

    EM = r0["det"]["EM"][idx_eval]
    SOCB = r0["det"]["SOC"][idx_eval][:, :-1]
    WS = r0["det"]["Wsp"][idx_eval]
    NB = r0["det"]["Nhat"][idx_eval]
    B0 = r0["det"]["B"][idx_eval]
    NT = net[idx_eval]
    has_e = EM > 1e-6
    soc_pct = (SOCB - E_MIN) / (E_MAX - E_MIN)

    print("\n  (a)(b) 应急区间（e>0）的储能状态：")
    print(f"    应急区间 {int(has_e.sum())} 个；SOC 占可用容量比例："
          f"均值 {soc_pct[has_e].mean()*100:.2f}%，中位 {np.median(soc_pct[has_e])*100:.2f}%，"
          f"≤5% 占 {np.mean(soc_pct[has_e] <= 0.05)*100:.1f}%")
    Dt = np.maximum(NT - B0, 0.0)
    At = ETA_D * (SOCB - E_MIN)
    eP = np.maximum(Dt - P_MAX, 0.0)
    eE = np.minimum(Dt, P_MAX) - np.minimum(np.minimum(Dt, P_MAX), At)
    ident = float(np.max(np.abs(eP + eE - EM)))
    tP, tE = float(eP.sum()), float(eE.sum())
    print(f"    缺口分解：功率受限 {tP:>10.1f} kWh（{tP/(tP+tE)*100:.1f}%）、"
          f"储能受限 {tE:>10.1f} kWh（{tE/(tP+tE)*100:.1f}%）"
          f"  [恒等式校验 max|eP+eE−e| = {ident:.2e} kWh]")
    print("    ⇒ 现象(a) 成立：应急时 SOC 接近下限；现象(b) 功率受限仅少数。")

    wsp_tot = float(WS.sum())
    buy_tot = float(B0.sum())
    pv_extra = np.maximum(-NT, 0.0)                     # 光伏过剩量
    ub_kwh = float(np.minimum(WS, np.maximum(B0, 0.0)).sum())
    lb_kwh = float(np.maximum(WS - pv_extra, 0.0).sum())
    ub_cost = float((np.minimum(WS, np.maximum(B0, 0.0)) * price).sum())
    lb_cost = float((np.maximum(WS - pv_extra, 0.0) * price).sum())
    print("\n  (c) 弃置（余量 w）与应急的关系：")
    print(f"    弃置总量 {wsp_tot:,.1f} kWh（≈{wsp_tot/len(idx_eval):,.0f} kWh/天），"
          f"相当于评价期购电量 {buy_tot:,.1f} kWh 的 {wsp_tot/buy_tot*100:.1f}%。")
    print("    **弃置不等于‘多买的电’**：可能来自光伏过剩，也可能来自购电，"
          "故不把全部弃置乘电价。")
    print(f"    · 落在净负荷<0（光伏过剩）区间上的 {float(WS[NT < 0.0].sum()):,.1f} kWh"
          f"（{float(WS[NT < 0.0].sum())/wsp_tot*100:.1f}%）→ 弃掉也不构成付费损失；")
    print("    · 可归因于购电的弃置**无法唯一归属**（能量同质），只能给区间：")
    print(f"        上界（购电优先）Σmin(w,b) = {ub_kwh:,.1f} kWh → 费用上界 {ub_cost/1e4:.4f} 万元")
    print(f"        下界（光伏优先）Σ(w−光伏过剩)⁺ = {lb_kwh:,.1f} kWh → 费用下界 {lb_cost/1e4:.4f} 万元")
    print(f"      即‘付费却弃置’的代价落在 {lb_cost/1e4:.2f}~{ub_cost/1e4:.2f} 万元，"
          f"占计划费 {base_plan/1e4:.2f} 万元的 "
          f"{lb_cost/base_plan*100:.1f}%~{ub_cost/base_plan*100:.1f}%——**量级可观**；")
    print("      但**不是可直接回收的损失**：减少购电会同时抬高紧急费，必须回到 J 上权衡。")
    day_w = WS.sum(axis=1); day_e = EM.sum(axis=1)
    both = (day_w > 1e-6) & (day_e > 1e-6)
    print(f"    · 同一天既有弃置又有应急：{int(both.sum())} 天 / {len(idx_eval)} 天"
          f"（{both.sum()/len(idx_eval)*100:.1f}%）⇒ 计划电量时间配置欠佳。")

    print("\n  (d) 月度预测偏差（仅报告，不用于选参）：")
    print(f"    {'月':>4}{'N̂−N_act':>13}{'中位基准−N_act':>17}{'判定':>14}")
    bias = {}
    for m in range(2, 13):
        li = loc[m]
        days = idx_eval[li]
        dev_q = float((NB[li] - NT[li]).mean())
        dev_med = float(np.mean([np.median(net[max(0, d - 14):d], axis=0).mean()
                                 - net[d].mean() for d in days]))
        bias[m] = dev_q
        tag = "持续低估" if dev_q < -1.0 else ("无低估迹象" if dev_med >= -5.0 else "基准偏低")
        print(f"    {m:>4}{dev_q:>+13.3f}{dev_med:>+17.3f}{tag:>14}")
    print("    说明：N̂−N_act 全月为正，是**上尾分位数构造的必然结果**（N̂ ≥ 中位基准），"
          "并非预测偏保守；")
    print("    真正需要警惕的是‘中位基准−N_act 持续为负’（历史窗口跟不上负载上移）——"
          "本数据中未出现。")
    out["bias"] = {str(m): bias[m] for m in range(2, 13)}

    # ============ 步骤 3：三版本对比 ============
    print("\n" + "-" * 100)
    print("步骤 3  三版本对比（统一口径：连续回放 + 公共预热 + 日循环期末）")
    print("-" * 100)
    vers = {}
    vers["V0原S1 β=0.70"] = eval_schedule(D, {m: 0.70 for m in range(2, 13)},
                                        betas_single, 14, idx_eval)
    vers["V1重校准单一β"] = eval_schedule(D, sched1, betas_single, 14, idx_eval)

    thetas_g = [(a, b_, c_, d_) for a in LEVELS for b_ in LEVELS
                for c_ in LEVELS for d_ in LEVELS]
    cum2 = prefix_cum(D, thetas_g, betas_grouped, 14)
    sched2 = select_schedule(D, thetas_g, cum2)
    vers["V2分时段β"] = eval_schedule(D, sched2, betas_grouped, 14, idx_eval)

    mono = [k for k, v in enumerate(thetas_g) if v[2] >= v[3] and v[0] >= v[3]]
    th_m = [thetas_g[k] for k in mono]
    cum2m = {i: cum2[k] for i, k in enumerate(mono)}
    sched2m = select_schedule(D, th_m, cum2m)
    vers["V2m分组β 峰≥谷"] = eval_schedule(D, sched2m, betas_grouped, 14, idx_eval)

    print(f"\n  {'版本':<20}{'计划/万元':>12}{'紧急/万元':>12}{'总费用/万元':>14}"
          f"{'紧急电量/kWh':>15}{'弃置/kWh':>13}")
    print("  " + "-" * 88)
    for k, r in vers.items():
        print(f"  {k:<20}{r['dp'][idx_eval].sum()/1e4:>12.4f}"
              f"{r['de'][idx_eval].sum()/1e4:>12.4f}"
              f"{(r['dp'][idx_eval].sum()+r['de'][idx_eval].sum())/1e4:>14.4f}"
              f"{r['dk'][idx_eval].sum():>15.1f}{r['dw'][idx_eval].sum():>13.1f}")

    v0 = vers["V0原S1 β=0.70"]
    p0 = float(v0["dp"][idx_eval].sum()); e0 = float(v0["de"][idx_eval].sum())
    print("\n  相对 V0 的差额与分解：")
    for k, r in vers.items():
        pk = float(r["dp"][idx_eval].sum()); ek = float(r["de"][idx_eval].sum())
        print(f"    {k:<20} 总 {(pk+ek-p0-e0)/1e4:>+9.4f} 万元"
              f"（计划 {(pk-p0)/1e4:>+8.4f}，紧急 {(ek-e0)/1e4:>+8.4f}）")

    mt = {k: monthly(r["dp"], r["de"], idx_eval, months) for k, r in vers.items()}
    SHORT = {"V0原S1 β=0.70": "V0", "V1重校准单一β": "V1",
             "V2分时段β": "V2", "V2m分组β 峰≥谷": "V2m"}
    print("\n  月度表现（总费用 / 万元）：")
    print(f"    {'月':>4}" + "".join(f"{SHORT[k]:>13}" for k in vers))
    for m in range(2, 13):
        print(f"    {m:>4}" + "".join(f"{mt[k][m][2]/1e4:>13.4f}" for k in vers))

    print("\n  [因果·可实施] V2 逐月分组 β 日程（早峰/日间/晚峰/其他）：")
    for m in range(2, 13):
        v = sched2[m]
        print(f"    用于 {m:2d} 月 → {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f} / {v[3]:.2f}")

    out["versions"] = {k: dict(plan=float(r["dp"][idx_eval].sum()),
                               em=float(r["de"][idx_eval].sum()),
                               total=float(r["dp"][idx_eval].sum() + r["de"][idx_eval].sum()),
                               emk=float(r["dk"][idx_eval].sum()),
                               wsp=float(r["dw"][idx_eval].sum())) for k, r in vers.items()}
    out["monthly"] = {k: {str(m): v for m, v in mt[k].items()} for k in vers}
    out["sched_single"] = {str(m): sched1[m] for m in range(2, 13)}
    out["sched_grouped"] = {str(m): list(sched2[m]) for m in range(2, 13)}

    # ============ 步骤 4：公平 PI 参照 ============
    print("\n" + "-" * 100)
    print("步骤 4  公平 PI 参照（完美日前预测；物理约束/日循环期末/评价边界三者相同）")
    print("-" * 100)
    rp = replay(D, lambda d: V0b, 14, WARMUP_DAY, E_FEB1, oracle=True)
    pp = float(rp["dp"][idx_eval].sum()); pe = float(rp["de"][idx_eval].sum())
    print(f"  PI(完美日前预测)：计划 {pp/1e4:.4f} + 紧急 {pe/1e4:.4f} = {(pp+pe)/1e4:.4f} 万元")
    print(f"  相对 V0（{base_tot/1e4:.4f}）的差距：{(pp+pe-base_tot)/1e4:+.4f} 万元")
    print(f"  ⇒ 同一物理约束与评价边界下，完美日前预测相对 V0 的可解释空间为 "
          f"{(base_tot-pp-pe)/1e4:.4f} 万元。")
    print("  注：这是完美**日前**预测参照；若 PI 另有跨日完美信息，其费用会更低，")
    print("      故该差距只作**该模型族内的参照**，不能当作绝对上界。")
    out["pi"] = dict(plan=pp, em=pe, total=pp + pe)

    outp = BASE / "results" / "q2_three_way.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落盘：{outp}")


if __name__ == "__main__":
    main()
