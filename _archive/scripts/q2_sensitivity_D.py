# -*- coding: utf-8 -*-
"""D 策略核心解释的敏感性检验（只读，不改动 q2_strategy_*.npz / q2_strategies.json）。

背景：D 的执行层写的是 d_t = min(d_star, deficit, P_MAX, 可放量)，即允许"主动少放电、
改用 5 倍价紧急购电"，以把电量留到更高价时段。论文 §二.3 已声明这是主方案解释，
但未给出该解释的量级影响。本脚本在同一日计划 b、同一预测下重跑一个对照：

  D_forced = 与 D 相同的每 4 小时重优化（只影响充电安排），
             但缺口一律按物理上限放电：d_t = min(deficit, P_MAX, 可放量)。

输出 D_forced 的评价期费用，用以界定"主动留电"这一解释贡献了多少降费。
"""
import sys, io
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23")
sys.path.insert(0, str(BASE / "scripts"))
import q2_improve as Q   # noqa: E402

T = Q.T
P_MAX = Q.P_MAX
E_MIN, E_MAX = Q.E_MIN, Q.E_MAX
ETA_C = Q.ETA_C
ETA_D = Q.ETA_D
BETA = 0.7
EVAL = None


def execute_reopt_forced(b, N_hat, net_act_d, price, E_start):
    """D 的重优化 + 强制放电（不允许为留电而少放电）。"""
    c = np.zeros(T); d = np.zeros(T); e = np.zeros(T); w = np.zeros(T)
    E = np.empty(T + 1); E[0] = E_start
    plan_c = {}
    for t0 in range(0, T, 24):
        sol = Q.lp_dispatch(price[t0:], N_hat[t0:], E[t0], b_fixed=b[t0:],
                            emerg_mult=Q.EMERG_MULT, e_end_min=E_start)
        for k in range(24):
            plan_c[t0 + k] = sol["c"][k]
        for t in range(t0, t0 + 24):
            net = b[t] - net_act_d[t]
            if net >= 0:
                c_t = min(plan_c[t], net, P_MAX, max(0.0, (E_MAX - E[t]) / ETA_C))
                c[t] = c_t; w[t] = net - c_t
            else:
                deficit = -net
                d[t] = min(deficit, P_MAX, max(0.0, (E[t] - E_MIN) * ETA_D))
                e[t] = deficit - d[t]
            E[t + 1] = E[t] + ETA_C * c[t] - d[t] / ETA_D
    return c, d, e, w, E


def main():
    price, net_fc, dates, load_act, pv_act, net_act = Q.load_inputs()
    eval_idx = np.where(dates >= pd.Timestamp("2025-02-01"))[0]

    Q.execute_reopt = execute_reopt_forced          # 打补丁
    print("回放 D_forced（禁止主动少放电）...")
    Rd = Q.replay(price, dates, net_fc, net_act, load_act, pv_act, "D", beta=BETA)
    md = Q.metrics(Rd, price, eval_idx)

    zC = np.load(BASE / r"results\q2_strategy_C.npz")
    zD = np.load(BASE / r"results\q2_strategy_D.npz")
    R = lambda z, k: z[k]
    mC = Q.metrics({k: R(zC, k) for k in ("B", "C", "D", "EM", "W")}, price, eval_idx)
    mD = Q.metrics({k: R(zD, k) for k in ("B", "C", "D", "EM", "W")}, price, eval_idx)

    print("\n=== 评价期(2.1-12.31) 费用对比（万元） ===")
    for tag, m in (("C(贪心)", mC), ("D(重优化+可留电)", mD), ("D_forced(重优化+强制放电)", md)):
        print(f"  {tag:26s} 计划 {m['计划费用(万元)']:9.2f} | 紧急 {m['紧急费用(万元)']:8.2f} | "
              f"总 {m['总费用(万元)']:9.2f} | 紧急电量 {m['紧急电量(kWh)']:11.1f} | 紧急日 {m['紧急日数']:3d}")

    em = Rd["EM"][eval_idx]
    print(f"\n  D_forced 紧急电量加权普通电价均值 = {(em * price).sum() / em.sum():.4f} 元/kWh "
          f"(C=1.2486, D=0.9674)")
    print(f"  D_forced 相对 C 总费用变化 = {mC['总费用(万元)'] - md['总费用(万元)']:+.2f} 万元")
    print(f"  D(可留电) 相对 C 总费用变化 = {mC['总费用(万元)'] - mD['总费用(万元)']:+.2f} 万元")
    print(f"  => '允许主动少放电' 这一解释贡献 = "
          f"{md['总费用(万元)'] - mD['总费用(万元)']:+.2f} 万元")
    print(f"  D_forced 年末电量 = {Rd['SOC1'][-1]:.4f} kWh")

    # 一致性校验
    r = Rd["B"] + Rd["D"] + Rd["EM"] - net_act - Rd["C"] - Rd["W"]
    print(f"\n  D_forced 能量平衡最大残差 = {np.max(np.abs(r)):.3e}")
    print(f"  D_forced 实际充放电重叠 = {float(np.minimum(Rd['C'], Rd['D']).max()):.3e}")
    print(f"  D_forced 跨日断裂 = {np.max(np.abs(Rd['SOC1'][:-1] - Rd['SOC0'][1:])):.3e}")
    print(f"  D_forced SOC范围 = [{min(Rd['SOC0'].min(),Rd['SOC1'].min()):.6f}, "
          f"{max(Rd['SOC0'].max(),Rd['SOC1'].max()):.6f}]")


if __name__ == "__main__":
    main()
