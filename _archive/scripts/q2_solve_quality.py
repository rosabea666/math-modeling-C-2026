# -*- coding: utf-8 -*-
r"""
§26 原因判别：采购不行，是"没解好"还是"场景不准"？（只读，不改模型）
================================================================================
用户 2026-09-12 意见：**先别调参、别换更复杂模型**。要回答一个判别问题——
"日前采购效果不好，究竟是 ①优化问题没解好（求解质量），还是 ②用来优化的未来场景不够准（场景失配）？"

**任务1（冻结基准）**：固定 V6 = 1581.6655 万元为唯一比较基准；统一 334 天评价期、初始 SOC
与跨日连续、信息时间、购电/应急费规则、时间映射、终端处理。输出计划/紧急/月度/供能余量。
外部 1300–1400 万元方案：**未核验，需费用分解与信息假设**；不围绕它调模型。

**任务2（三值诊断，本轮重点）**：在**相同场景、相同初始状态、相同终端 Φ** 下比较
    J_松弛下界(A)  ≤  J_场景问题最优  ≤  J_当前可行(B)  ≤  J_V6合同(C)
  · A = anticipative 两阶段 LP（共享 b、每场景自由储能）——**放松下界**，只用于量求解差距，
        **不是**全年实际费用下界，放松动作**不可**直接执行；
  · C = V6 合同在场景池上的**因果价值反馈**费用（参考）；
  · B = 坐标下降(CD)从 V6 合同出发在场景池上优化后的**因果**费用（当前可行）；
  · A_V6 = 固定 V6 合同、自由储能的 anticipative 费用 ⇒ **政策差距 = C − A_V6**（因果政策 vs 自由储能）。
  判别：**场景改进 = C − B**（CD 在场景上确实优化了多少）；**迁移 = 实际V6 − 实际CD**（同样的改进
  在实测日是否兑现）。若 场景改进>0 而 迁移<0 ⇒ **场景失配**（优化对了错的未来）；若 B 距 A 很远 ⇒ 求解质量。
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

from q2_model import (T, MULT, E_MIN, E_MAX, ETA_C, E_FEB1, WARMUP_DAY, load_data)  # noqa: E402
import q2_value_adp as va  # noqa: E402
import q2_saa_lp as sa    # noqa: E402

V5_TOTAL = 1583.7354
V6_TOTAL = 1581.6655


def day_cost(b, e, price):
    return float((b * price).sum() + MULT * (e * price).sum())


def replay_v6_keep(D, V5B, Estar_of):
    """任务1：冻结 V6（V5 合同 + 价值反馈）。返回逐日 b/e/w/SOC 与汇总。"""
    price, net, months = D["price"], D["net"], D["_month"]
    nd = len(D["dates"])
    B = np.zeros((nd, T)); EM = np.zeros((nd, T)); W = np.zeros((nd, T))
    soc0 = np.zeros(nd); soc1 = np.zeros(nd)
    E = float(E_FEB1)
    for d in range(WARMUP_DAY, nd):
        Estar = Estar_of[months[d]]
        c, dd, e, w, Etraj = va.execute_value(V5B[d], net[d], price, E, Estar)
        B[d] = V5B[d]; EM[d] = e; W[d] = w; soc0[d] = E; soc1[d] = Etraj[-1]
        E = Etraj[-1]
    return dict(B=B, EM=EM, W=W, soc0=soc0, soc1=soc1)


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    month_start = {m: int(np.min(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§26 原因判别：采购不行 = 没解好 还是 场景不准？")
    print("=" * 104)
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）；"
          f"冻结基准 V6 = {V6_TOTAL} 万元")

    # ---- 价值函数 / 场景池 ----
    print("\n[0] 逐月价值函数（Φ 与 E*）")
    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        scen = va.fvi_pool(month_start[m], D)
        _, Estar, Phi = va.estimate_value(price, scen)
        Estar_of[m], Phi_of[m] = Estar, Phi
    print("    完成")

    # ---- 任务1：冻结 V6 基准 ----
    print("\n[任务1] 冻结 V6 基准（V5 合同 + 价值反馈），输出统一口径")
    sched, rV5 = va.reproduce_v5(D, months)
    V5B = rV5["det"]["B"]
    R6 = replay_v6_keep(D, V5B, Estar_of)
    plan6 = float((R6["B"][idx] * price).sum()) / 1e4
    em6 = float(MULT * (R6["EM"][idx] * price).sum()) / 1e4
    tot6 = plan6 + em6
    print(f"  V6：计划 {plan6:.4f} + 紧急 {em6:.4f} = {tot6:.4f} 万元（对拍 {V6_TOTAL}，差 {tot6 - V6_TOTAL:+.4f}）")
    print(f"  供能余量（弃置）= {R6['W'][idx].sum():,.0f} kWh；年末 SOC = {R6['soc1'][idx][-1]:.0f} kWh")
    print("  V6 月度费用（计划/紧急/合计，万元）：")
    print(f"    {'月':>3}{'计划':>10}{'紧急':>10}{'合计':>10}")
    for m in range(2, 13):
        mi = idx[months[idx] == m]
        p = float((R6["B"][mi] * price).sum()) / 1e4
        e = float(MULT * (R6["EM"][mi] * price).sum()) / 1e4
        print(f"    {m:>3}{p:>10.4f}{e:>10.4f}{p+e:>10.4f}")
    print("  外部 1300–1400 万元方案：**未核验**——需其费用分解（计划/紧急）与信息假设；"
          "在拿到之前不围绕该总数调整模型。")

    # ---- 任务2：三值诊断 ----
    print("\n[任务2] 三值诊断：A(松弛下界) ≤ 最优 ≤ B(CD) ≤ C(V6合同)；同场景/同初始/同 Φ")
    A = np.zeros(len(D["dates"])); AV6 = np.zeros(len(D["dates"]))
    Bc = np.zeros(len(D["dates"])); Cc = np.zeros(len(D["dates"]))
    actCD = np.zeros(len(D["dates"])); actLB = np.zeros(len(D["dates"])); actV6 = np.zeros(len(D["dates"]))
    cache = {}
    t0 = time.perf_counter()
    for d in range(WARMUP_DAY, len(D["dates"])):
        m = months[d]
        E0 = float(R6["soc0"][d])                    # 同一初始状态（V6 实际起点）
        scen, wsc = va.scenario_pool(d, D)
        Phi, Estar = Phi_of[m], Estar_of[m]
        # A：松弛下界（anticipative 最优 b）
        bLB, A[d] = sa.solve_day(price, scen, wsc, E0, Phi, cache)
        # A_V6：固定 V6 合同、自由储能（政策差距用）
        _, AV6[d] = sa.solve_day(price, scen, wsc, E0, Phi, cache, b_fixed=V5B[d])
        # C：V6 合同 + 因果价值反馈
        Cc[d] = va.sim_cost(V5B[d][None, :], E0, scen, wsc, price, Estar, Phi)[0]
        # B：CD 从 V6 合同出发优化（因果价值反馈）
        bCD = va.procure(V5B[d], E0, scen, wsc, price, Estar, Phi)
        Bc[d] = va.sim_cost(bCD[None, :], E0, scen, wsc, price, Estar, Phi)[0]
        # 实测日部署（同一初始状态，因果价值反馈）
        _, _, e, _, _ = va.execute_value(V5B[d], net[d], price, E0, Estar)
        actV6[d] = day_cost(V5B[d], e, price)
        _, _, e, _, _ = va.execute_value(bCD, net[d], price, E0, Estar)
        actCD[d] = day_cost(bCD, e, price)
        _, _, e, _, _ = va.execute_value(bLB, net[d], price, E0, Estar)
        actLB[d] = day_cost(bLB, e, price)
        if (d - WARMUP_DAY) % 40 == 0:
            print(f"    d={d} 累计 {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- 汇总判别 ----
    print("\n" + "=" * 104)
    print("判别汇总（评价期日均，元/天；以及全年合计万元）")
    print("=" * 104)
    sA, sAV6, sB, sC = A[idx].sum(), AV6[idx].sum(), Bc[idx].sum(), Cc[idx].sum()
    aV6, aCD, aLB = actV6[idx].sum(), actCD[idx].sum(), actLB[idx].sum()
    print("  【场景问题上（同一批历史场景，含相同 Φ 终端）】")
    print(f"    A  松弛下界(anticipative 最优 b)   = {sA/1e4:9.4f} 万元")
    print(f"    A_V6 固定V6合同+自由储能           = {sAV6/1e4:9.4f} 万元")
    print(f"    B  CD优化后的合同(因果)            = {sB/1e4:9.4f} 万元")
    print(f"    C  V6合同(因果价值反馈)            = {sC/1e4:9.4f} 万元")
    print(f"    链校验 A≤A_V6≤C：{sA<=sAV6+1e-3 and sAV6<=sC+1e-3}；A≤B≤C：{sA<=sB+1e-3 and sB<=sC+1e-3}")
    print(f"    → 政策差距(C−A_V6) = {(sC-sAV6)/1e4:+.4f} 万元（因果政策 vs 自由储能，同一 V6 合同）")
    print(f"    → 合同差距(A_V6−A) = {(sAV6-sA)/1e4:+.4f} 万元（自由储能下 V6 合同 vs 最优合同）")
    print(f"    → CD在场景上确有的优化(C−B) = {(sC-sB)/1e4:+.4f} 万元")
    print(f"    → CD距松弛下界(B−A) = {(sB-sA)/1e4:+.4f} 万元（含政策差距+求解残差，是求解残差的上界）")
    print("  【实测日部署（同一初始状态，因果价值反馈）】")
    print(f"    V6   = {aV6/1e4:9.4f} 万元")
    print(f"    CD合同 = {aCD/1e4:9.4f} 万元（相对 V6 {(aCD-aV6)/1e4:+.4f}）")
    print(f"    LB合同 = {aLB/1e4:9.4f} 万元（相对 V6 {(aLB-aV6)/1e4:+.4f}）")
    scen_impr = (sC - sB) / 1e4
    act_impr = (aV6 - aCD) / 1e4
    print(f"\n  【迁移判别】场景改进 C−B = {scen_impr:+.4f} 万元；实测迁移 V6−CD = {act_impr:+.4f} 万元；"
          f"迁移比 = {(act_impr/scen_impr if abs(scen_impr)>1e-6 else float('nan')):+.3f}")

    out = dict(口径=dict(评价期天数=int(len(idx)), V6=V6_TOTAL),
               任务1_V6=dict(计划=plan6, 紧急=em6, 合计=tot6,
                            供能余量=float(R6["W"][idx].sum()),
                            月度={str(m): dict(计划=float((R6["B"][idx[months[idx]==m]]*price).sum())/1e4,
                                              紧急=float(MULT*(R6["EM"][idx[months[idx]==m]]*price).sum())/1e4)
                                   for m in range(2,13)}),
               任务2=dict(A_LB=sA/1e4, A_V6=sAV6/1e4, B_CD=sB/1e4, C_V6=sC/1e4,
                         政策差距=(sC-sAV6)/1e4, 合同差距=(sAV6-sA)/1e4,
                         CD场景优化=(sC-sB)/1e4, CD距LB=(sB-sA)/1e4,
                         实际_V6=aV6/1e4, 实际_CD=aCD/1e4, 实际_LB=aLB/1e4,
                         场景改进=scen_impr, 实测迁移=act_impr))
    (BASE / "results" / "q2_solve_quality.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_solve_quality.json   总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
