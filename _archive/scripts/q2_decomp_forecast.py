# -*- coding: utf-8 -*-
"""
问题二 · 对照一：日电量—日内形状分解预测（只读：不改模型、不重出 Excel）
=====================================================================
用户 2026-09-11 意见：把改进重心前移到**预测与日前采购**，并做一个"分开识别收益"的对照。
本脚本实现**对照一**，只替换 S1 的预测层，其余一律不变：

  预测         ：日电量 → 日内归一化形状 分解，**负荷与光伏分别建模**
                   L_{d,t} = A^L_d · s^L_{d,t}          Σ_t s^L = 1
                   V_{d,t} = A^V_d · s^V_{d,t}          Σ_t s^V = 1
                   N̂_{d,t} = Â^L_d·ŝ^L_{d,t} − Â^V_d·ŝ^V_{d,t}
                 储备**只加在日电量 A 上**（负荷取上尾、光伏取下尾），日内形状不加储备。
                 **不对净负载归一化**——净负载可能为负、日总量可能接近零（用户明确提示）。
  日前采购     ：**保持 S1 原框架**（同一个确定性 LP，同一个日循环期末，同一个目标 Σp·b）
  执行         ：**固定 must/max 尽限反馈**（与 S1 逐元素相同）
  评价         ：连续回放 + 公共预热（2/1 起评 E=10800），评价期 2/1–12/31 只用于报告

口径纪律（与 §16/§17/§18 完全一致）：
  · 选参只用该月之前的数据（walk-forward 前缀累计 J），评价期不参与选参；
  · 目标即完整执行后的 J = Σp·b + 5Σp·e，不按预测误差、不按紧急电量单独选参；
  · 训练费来自**连续回放**（储能跨日耦合），不按天独立初始化。

关键诊断（回答"预测到底贡献了多少"）：
  (1) 因果可选 β_L 的月度日程与费用；
  (2) 事后最优 β_L（不可实施，仅作形状参考）；
  (3) **储备量对照**：对照一的日均"多买量"与 S1 的对比 —— 若对照一以*更少*的储备取胜，
      说明分解在"把储备放对位置"上更有效；若必须*加大*储备才赢，则只是同一权衡曲线上的另一点。
  (4) 形状对照：分解形状 vs S1 的分位数曲线形状（去掉水平后的相关性与峰值位置）。
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

from q2_model import (T, MULT, E_INIT, E_MIN, E_MAX, P_MAX,  # noqa: E402
                      WARMUP_DAY, E_FEB1, Policy, load_data, make_plan, execute)
from q2_three_way import (BETA_SINGLE, betas_single, forecast_custom,  # noqa: E402
                          prefix_cum, select_schedule, eval_schedule,
                          replay as replay_ps)

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

W = 14
BETA_L = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
PV_SHAPE_FRAC = 0.5      # 只用日电量 > 0.5×窗口中位 的天数估计光伏形状（避免阴天噪声）


def decomp_forecast(d, D, W, beta):
    """对照一的预测层：负荷/光伏分别做 日电量 × 日内归一化形状 分解。

    储备加在**日电量**上：Â^L 取历史日电量的上尾分位、Â^V 取下尾分位（越保守净负载越大）。
    日内形状取归一化形状的**中位数**（按分量取中位后重新归一化），不额外加储备。
    """
    ld = D["load"][max(0, d - W):d]
    pv = D["pv"][max(0, d - W):d]
    if ld.shape[0] < 7:
        return D["net_ref"].copy()

    tl = ld.sum(axis=1)                      # 日电量（负荷），恒 > 0
    tv = pv.sum(axis=1)                      # 日电量（光伏）
    sL = ld / tl[:, None]                    # 归一化形状，Σ=1
    mv = float(np.median(tv))
    ok = tv > PV_SHAPE_FRAC * mv
    src = pv[ok] if ok.sum() >= 3 else pv
    den = tv[ok][:, None] if ok.sum() >= 3 else np.maximum(tv, 1e-9)[:, None]
    sV = src / den
    shL = np.median(sL, axis=0)
    shV = np.median(sV, axis=0)
    shL = shL / shL.sum()
    shV = shV / shV.sum()

    mL = float(np.median(tl))
    mV = mv
    A_L = mL + max(0.0, float(np.quantile(tl - mL, beta)))
    A_V = mV + min(0.0, float(np.quantile(tv - mV, 1.0 - beta)))
    return A_L * shL - A_V * shV


def replay_decomp(D, beta_of_day, start_idx, E_start, keep=False):
    """与 q2_three_way.replay 同构，只把预测层换成 decomp_forecast（其余逐字相同口径）。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    dn = np.zeros(nd)                        # 日均"多买量" Σ_t(N̂−N_act)⁺
    det = None
    if keep:
        det = dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)), Wsp=np.zeros((nd, T)),
                   SOC=np.zeros((nd, T + 1)), Nhat=np.zeros((nd, T)))
    for d in range(start_idx, nd):
        N_hat = decomp_forecast(d, D, W, beta_of_day(d))
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum())
        de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum())
        dw[d] = float(w.sum())
        dn[d] = float(np.maximum(N_hat - net[d], 0.0).sum())
        if keep:
            det["B"][d] = b; det["EM"][d] = e; det["Wsp"][d] = w
            det["SOC"][d] = Etraj; det["Nhat"][d] = N_hat
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, dn=dn, det=det)


def main():
    D = load_data()
    price, dates, net, months_all = D["price"], D["dates"], D["net"], D["dates"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in months_all])
    D["_month"] = months
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx_eval = np.where(dates >= np.datetime64("2025-02-01"))[0]
    loc = {m: np.where(months[idx_eval] == m)[0] for m in range(2, 13)}

    print("=" * 104)
    print("问题二 · 对照一：日电量—日内形状分解预测（负荷/光伏分别建模；采购与执行保持不变）")
    print("=" * 104)
    print(f"  评价期 {str(dates[idx_eval[0]])[:10]} ~ {str(dates[idx_eval[-1]])[:10]}（{len(idx_eval)} 天）；"
          f"公共预热 E = {E_FEB1:.0f} kWh；W = {W}；光伏形状只用日电量 > {PV_SHAPE_FRAC:.1f}×中位 的天")
    print("  口径：储备只加在**日电量**上（负荷上尾 / 光伏下尾）；日内形状不加储备；"
          "采购 = S1 同一个确定性 LP；执行 = must/max 尽限反馈。")

    # ---------- 0. 形状合理性（分解 vs 分位数曲线） ----------
    print("\n" + "-" * 104)
    print("0. 预测形状自检（取 7 月 15 日做示例；水平与形状分离后比较）")
    print("-" * 104)
    d0 = int(np.where(dates == np.datetime64("2025-07-15"))[0][0])
    nd0 = decomp_forecast(d0, D, W, 0.80)
    nd1 = forecast_custom(d0, D, W, betas_single(0.70))
    print(f"  {'指标':<22}{'对照一(β_L=0.80)':>20}{'S1(β=0.70)':>18}{'当日实测':>16}")
    print(f"  {'净负载日总量/kWh':<20}{nd0.sum():>20.1f}{nd1.sum():>18.1f}{net[d0].sum():>16.1f}")
    pk = lambda a: "%02d:%02d" % (int(np.argmax(a)) * 10 // 60, int(np.argmax(a)) * 10 % 60)
    print(f"  {'净负载峰值时刻':<20}{pk(nd0):>20}{pk(nd1):>18}{pk(net[d0]):>16}")
    sh0 = np.maximum(nd0, 0.0); sh1 = np.maximum(nd1, 0.0)
    c1 = float(np.corrcoef(sh0, sh1)[0, 1])
    c2 = float(np.corrcoef(sh0, np.maximum(net[d0], 0.0))[0, 1])
    print(f"  曲线形状相关性（尺度无关，越接近 1 日内分布越一致）："
          f"对照一 vs S1 = {c1:.4f}；对照一 vs 当日实测 = {c2:.4f}")
    print("  ⇒ 两族的日内形状几乎同源；差别主要在'储备加在日电量还是加在每个区间'。")

    # ---------- 1. 因果选参：β_L 的月度日程 ----------
    print("\n" + "-" * 104)
    print("1. 因果选参（只用该月之前的数据；目标 = 完整 J）")
    print("-" * 104)
    cum = {}
    for k, b_ in enumerate(BETA_L):
        r = replay_decomp(D, lambda d, b_=b_: b_, 0, E_INIT)
        cum[k] = np.cumsum(r["dp"] + r["de"])
    print("  [诊断·post-hoc，仅看形状与边界] 全年连续回放 J(β_L)：")
    curve = {b_: float(cum[k][-1]) for k, b_ in enumerate(BETA_L)}
    bmin = min(curve, key=curve.get)
    for b_ in BETA_L:
        print(f"    β_L={b_:.2f}   全年 J = {curve[b_]/1e4:>10.4f} 万元"
              f"{'   ← 最优' if b_ == bmin else ''}")
    edge = bmin in (BETA_L[0], BETA_L[-1])
    lo, hi = min(curve.values()), max(curve.values())
    print(f"  ⇒ 事后最优 β_L = {bmin:.2f}；落在边界：{'是（需外扩）' if edge else '否（内部解）'}；"
          f"极差 {(hi-lo)/1e4:.4f} 万元（{(hi-lo)/lo*100:.2f}%）")

    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        k = min(range(len(BETA_L)), key=lambda i: cum[i][te])
        sched[m] = BETA_L[k]
    print("\n  [因果·可实施] 逐月 β_L 日程：")
    for m in range(2, 13):
        print(f"    用于 {m:2d} 月 → β_L = {sched[m]:.2f}"
              f"   （训练至 {str(dates[D['_month_end'][m-1]])[:10]}）")

    # ---------- 2. 与 V0 / V1 的统一直角对比 ----------
    print("\n" + "-" * 104)
    print("2. 统一口径对比（V0 = 原 S1；V1 = 因果重校准单一 β 的 S1；对照一 = 本脚本）")
    print("-" * 104)
    cum1 = prefix_cum(D, BETA_SINGLE, betas_single, W)
    sched1 = select_schedule(D, BETA_SINGLE, cum1)
    res = {}
    res["V0原S1 β=0.70"] = eval_schedule(D, {m: 0.70 for m in range(2, 13)}, betas_single, W, idx_eval)
    res["V1重校准单一β"] = eval_schedule(D, sched1, betas_single, W, idx_eval)
    res["对照一分解预测"] = replay_decomp(D, lambda d: sched[months[d]], WARMUP_DAY, E_FEB1, keep=True)
    res["对照一 事后最优β_L"] = replay_decomp(D, lambda d: bmin, WARMUP_DAY, E_FEB1, keep=True)

    v0 = res["V0原S1 β=0.70"]
    p0 = float(v0["dp"][idx_eval].sum()); e0 = float(v0["de"][idx_eval].sum())
    print(f"\n  {'版本':<22}{'计划/万元':>12}{'紧急/万元':>12}{'总费用/万元':>14}"
          f"{'紧急电量/kWh':>15}{'弃置/kWh':>13}")
    print("  " + "-" * 88)
    met = {}
    for k, r in res.items():
        pk = float(r["dp"][idx_eval].sum()); ek = float(r["de"][idx_eval].sum())
        met[k] = dict(plan=pk, em=ek, total=pk + ek,
                      emk=float(r["dk"][idx_eval].sum()), wsp=float(r["dw"][idx_eval].sum()))
        print(f"  {k:<22}{pk/1e4:>12.4f}{ek/1e4:>12.4f}{(pk+ek)/1e4:>14.4f}"
              f"{met[k]['emk']:>15.1f}{met[k]['wsp']:>13.1f}")
    print("\n  相对 V0 的差额与分解：")
    for k, m in met.items():
        print(f"    {k:<22} 总 {(m['total']-p0-e0)/1e4:>+9.4f} 万元"
              f"（计划 {(m['plan']-p0)/1e4:>+8.4f}，紧急 {(m['em']-e0)/1e4:>+8.4f}）")

    def monthly(r):
        return {m: (float(r["dp"][idx_eval][loc[m]].sum()),
                    float(r["de"][idx_eval][loc[m]].sum())) for m in range(2, 13)}
    mt = {k: monthly(r) for k, r in res.items()}
    SH = {"V0原S1 β=0.70": "V0", "V1重校准单一β": "V1",
          "对照一分解预测": "对照一", "对照一 事后最优β_L": "对照一*"}
    print("\n  月度总费用（万元）：")
    print(f"    {'月':>4}" + "".join(f"{SH[k]:>14}" for k in res))
    for m in range(2, 13):
        print(f"    {m:>4}" + "".join(f"{(mt[k][m][0]+mt[k][m][1])/1e4:>14.4f}" for k in res))

    # ---------- 3. 关键诊断：是"储备放对位置"还是"只是加大了储备" ----------
    print("\n" + "-" * 104)
    print("3. 关键诊断：对照一赢在'储备放对位置'，还是只赢在'储备更大'？")
    print("-" * 104)
    # V0 的日均多买量需要重算（q2_three_way 未记录该量）
    nb0 = np.array([np.maximum(forecast_custom(d, D, W, betas_single(0.70)) - net[d], 0.0).sum()
                    for d in idx_eval])
    nnd = res["对照一分解预测"]["dn"][idx_eval]
    print(f"  {'':<28}{'日均多买量/kWh':>16}{'日均购电/kWh':>16}{'日均弃置/kWh':>16}")
    for nm, r, dn in (("V0 原 S1", v0, nb0), ("对照一（因果）", res["对照一分解预测"], nnd),
                      ("对照一（事后最优 β_L）", res["对照一 事后最优β_L"],
                       res["对照一 事后最优β_L"]["dn"][idx_eval])):
        dn_ = np.array(dn)
        print(f"  {nm:<28}{dn_.mean():>16.1f}{r['dp'][idx_eval].sum()/len(idx_eval)/price.mean():>16.1f}"
              f"{r['dw'][idx_eval].sum()/len(idx_eval):>16.1f}")
    print("  说明：多买量 = Σ_t max(N̂_t − N_act,t, 0)/天，即预测超出实测的部分（储备的直接度量）。")
    print("        若对照一的日均多买量**不高于** S1 而总费用更低，则属于'储备放对位置'；")
    print("        若必须显著加大储备才更低，则只是同一条权衡曲线上的另一点，不算预测改进。")
    # 方差维度：储备在时间上的集中度
    def conc(r, dn):
        nb = r["det"]["Nhat"][idx_eval]; nt = net[idx_eval]
        pos = np.maximum(nb - nt, 0.0)
        s = pos.sum(axis=1, keepdims=True)
        fr = pos / np.maximum(s, 1e-9)
        return float((fr ** 2).sum(axis=1).mean())      # HHI：越大表示储备越集中在少数区间
    print(f"\n  储备时间集中度（HHI，越大越集中）：V0 {conc(v0, None):.4f}；"
          f"对照一 {conc(res['对照一分解预测'], None):.4f}")

    # ---------- 4. SOC 口径检查（终端处理是否可比） ----------
    print("\n" + "-" * 104)
    print("4. SOC 口径检查（确认'终端处理'没有造成比较偏差）")
    print("-" * 104)
    for k, r in res.items():
        soc = r["det"]["SOC"][idx_eval]
        e0d = soc[:, 0]; e1d = soc[:, -1]
        print(f"  {k:<22} 日均起始 SOC {e0d.mean():>9.1f} | 日均末 SOC {e1d.mean():>9.1f}"
              f" | 年末 SOC {soc[-1, -1]:>9.1f} | 触底天数 {int((r['det']['SOC'][idx_eval].min(axis=1) <= E_MIN + 1e-3).sum()):>3d}")
    print("  说明：两者同为'执行层驱动'的 SOC 轨迹（S1 的计划轨迹在 must/max 下不被使用），")
    print("        故此处只检查二者是否可比，不构成对任一方的额外约束。")

    # ---------- 5. 等储备前沿：分解 vs 分位曲线 是否在同一条权衡曲线上 ----------
    print("\n" + "-" * 104)
    print("5. 等储备前沿对比（决定性诊断：'分位曲线族'与'分解族'在同等储备量下谁更便宜）")
    print("-" * 104)

    def _res_of(r):
        nb = r["det"]["Nhat"][idx_eval]; nt = net[idx_eval]
        return float(np.maximum(nb - nt, 0.0).sum() / len(idx_eval))

    fps, fdc = {}, {}
    for b_ in BETA_SINGLE:
        r = replay_ps(D, lambda d, b_=b_: betas_single(b_), W, WARMUP_DAY, E_FEB1, keep=True)
        fps[b_] = (_res_of(r), float((r["dp"][idx_eval] + r["de"][idx_eval]).sum()))
    for b_ in BETA_L:
        r = replay_decomp(D, lambda d, b_=b_: b_, WARMUP_DAY, E_FEB1, keep=True)
        fdc[b_] = (_res_of(r), float((r["dp"][idx_eval] + r["de"][idx_eval]).sum()))

    print(f"  {'分位曲线族':>10}{'储备/kWh':>12}{'总费/万元':>12}   |"
          f"{'分解族':>10}{'储备/kWh':>12}{'总费/万元':>12}")
    for b_, b2 in zip(BETA_SINGLE, BETA_L):
        print(f"  {'β=%.2f' % b_:>10}{fps[b_][0]:>12.0f}{fps[b_][1]/1e4:>12.4f}   |"
              f"  {'β_L=%.2f' % b2:>8}{fdc[b2][0]:>12.0f}{fdc[b2][1]/1e4:>12.4f}")

    def _interp(rows, target):
        xs = np.array([v[0] for v in rows.values()]); ys = np.array([v[1] for v in rows.values()])
        o = np.argsort(xs)
        return float(np.interp(target, xs[o], ys[o]))

    lo_ = max(min(v[0] for v in fps.values()), min(v[0] for v in fdc.values()))
    hi_ = min(max(v[0] for v in fps.values()), max(v[0] for v in fdc.values()))
    print(f"\n  等储备内插比较（两族储备的重叠区间 [{lo_:.0f}, {hi_:.0f}] kWh/天）：")
    print(f"    {'储备/kWh':>12}{'分位族/万元':>14}{'分解族/万元':>14}{'分解优势/万元':>16}")
    gap = []
    for target in np.linspace(lo_, hi_, 7):
        a = _interp(fps, target); c = _interp(fdc, target)
        gap.append(a - c)
        print(f"    {target:>12.0f}{a/1e4:>14.4f}{c/1e4:>14.4f}{(a-c)/1e4:>+16.4f}")
    print(f"  ⇒ 等储备下分解族的平均优势 {np.mean(gap)/1e4:+.4f} 万元"
          f"（区间 [{min(gap)/1e4:+.4f}, {max(gap)/1e4:+.4f}]）")
    print("  判读：若分解族在等储备下**整体低于**分位族，则分解带来的是新机制；")
    print("        若两曲线几乎重合或交叉，则分解只是'同一条权衡曲线上的另一点'，不算预测改进。")
    print("  注：本对比是**同一执行规则（must/max）**下的费用-储备前沿，族间差异只来自预测构造方式。")

    out = {
        "口径": {"W": W, "评价期天数": int(len(idx_eval)), "公共预热": E_FEB1,
                 "光伏形状阈值": PV_SHAPE_FRAC, "储备位置": "日电量层"},
        "beta_curve": {str(k): v for k, v in curve.items()},
        "beta_argmin": bmin, "beta_is_edge": bool(edge),
        "sched": {str(m): sched[m] for m in range(2, 13)},
        "versions": {k: met[k] for k in met},
        "monthly": {k: {str(m): v for m, v in mt[k].items()} for k in mt},
        "多买量日均": {"V0": float(nb0.mean()), "对照一": float(nnd.mean())},
        "等储备前沿": {"分位族": {str(k): list(v) for k, v in fps.items()},
                   "分解族": {str(k): list(v) for k, v in fdc.items()},
                   "等储备优势均值": float(np.mean(gap))},
    }
    outp = BASE / "results" / "q2_decomp.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：{outp}")


if __name__ == "__main__":
    main()
