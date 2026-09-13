# =============================================================================
# 问题三核心程序：0:00 预报通道融合 + 带双向违约价的滚动合同调整
#
# 与正文 §7 一一对应：
#   (1) pv_energy   附件3整点预报 -> 区间电量（式 7.1，线性插值+区间积分）
#   (2) kappa_0     0:00 通道分档收缩系数（式 7.3，过去 30 天因果滚动回归）
#   (3) adjust_lp   调整层 LP（式 7.4，违约成本内嵌，单个 LP 精确表示）
#   (4) exec_segment 执行层尽限充放电（must/max，与问题二共用）
#
# 主方案 B1 的全年回放 = 问题二计划层(0:00 用融合预测) + 6/12/18 三次
# adjust_lp + exec_segment 分段执行；评价期总费用 1345.7069 万元。
# 完整可运行源程序见支撑材料 问题三/代码/（q3_model.py 一条命令复现）。
# =============================================================================

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

T, DT = 144, 1 / 6               # 日区间数、区间长度 (h)
ETA_C = ETA_D = 0.9              # 充 / 放电效率
E_MIN, E_MAX = 1200.0, 10800.0   # 电量运行区间 (kWh)
P_MAX = 5000.0 * DT              # 单区间最大充/放电量 (kWh)
MULT = 5.0                       # 紧急购电电价倍数
K_ROLL, MIN_SMP = 30, 7          # 因果回归窗口与最小样本
LEAD_GROUPS = ((0, 6), (6, 12), (12, 24))   # 预报时长分档 g


def pv_energy(pv_pts, issue, pv_now):
    """issue 时刻发布的整点预报 -> 当日 144 区间光伏电量 (kWh/区间)。
    pv_pts: P(τ+1 … τ+24) 整点功率预报；pv_now: τ 时刻实测功率（首端点锚定）。
    相邻整点线性插值后对 10 分钟区间梯形积分（τ=0 时首端点取 0）。"""
    xp = np.arange(issue, 25.0)
    fp = np.concatenate([[pv_now], pv_pts[:24 - issue]])
    tt = np.arange(T) / 6.0
    ok = tt >= issue
    out = np.zeros(T)
    out[ok] = 0.5 * (np.interp(tt[ok], xp, fp)
                     + np.interp(tt[ok] + DT, xp, fp)) * DT
    return out


def kappa_0(net_hist, base_hist, dv0_hist):
    """0:00 通道分档收缩系数 κ_g（式 7.3）。
    net_hist/base_hist/dv0_hist: 过去 K_ROLL 天的实测净负载、问题二基准预测、
    两光伏通道差 Ṽ(0)−V̂_hist（均为 [天数, 144]）；样本 < MIN_SMP 回退 0。"""
    k0 = np.zeros(3)
    if len(net_hist) < MIN_SMP:
        return k0
    lead = np.tile(np.arange(T) // 6 + 1, len(net_hist))     # 预报时长 h
    r = (net_hist - base_hist).ravel()
    x = dv0_hist.ravel()
    for g, (a, b) in enumerate(LEAD_GROUPS):
        m = (lead >= a) & (lead < b)
        den = float(x[m] @ x[m])
        if den > 1e-9:
            k0[g] = float(np.clip(-(r[m] @ x[m]) / den, 0.0, 1.0))
    return k0          # N̂ = 问题二预测 − κ_g·[Ṽ(0)−V̂_hist]（式 7.2）


def adjust_lp(price, N_hat, b_plan, E_tau, E0):
    """对剩余 H 个区间重定合同 b'（式 7.4，双向违约价内嵌，单个 LP）。
    min  Σ p·b' + 0.5·Σ p·(u+v) + 5·Σ p·e
    s.t. u >= b_plan−b', v >= b'−b_plan（偏差始终对 0:00 计划计量）
         b'+d+e = N̂+c+w, SOC 递推, E_0=E_tau(实测), E_H=E0(日循环)。"""
    H = len(N_hat)
    ib, ic, id_, ie, iw, iu, iv, iE = (0, H, 2*H, 3*H, 4*H, 5*H, 6*H, 7*H)
    n = 7 * H + (H + 1)
    cobj = np.zeros(n)
    cobj[ib:ib+H] = price
    cobj[iu:iu+H] = cobj[iv:iv+H] = 0.5 * price
    cobj[ie:ie+H] = MULT * price

    Aeq = lil_matrix((2 * H + 2, n)); beq = np.zeros(2 * H + 2)
    for t in range(H):
        Aeq[t, iE+t+1] = 1.0; Aeq[t, iE+t] = -1.0
        Aeq[t, ic+t] = -ETA_C; Aeq[t, id_+t] = 1.0 / ETA_D     # SOC 递推
        r = H + t
        Aeq[r, ib+t] = 1.0; Aeq[r, id_+t] = 1.0; Aeq[r, ie+t] = 1.0
        Aeq[r, ic+t] = -1.0; Aeq[r, iw+t] = -1.0               # 功率平衡
        beq[r] = N_hat[t]
    Aeq[2*H, iE] = 1.0;   beq[2*H] = E_tau
    Aeq[2*H+1, iE+H] = 1.0; beq[2*H+1] = E0

    Aub = lil_matrix((2 * H, n)); bub = np.zeros(2 * H)
    for t in range(H):                       # u >= b_plan − b'
        Aub[t, iu+t] = -1.0; Aub[t, ib+t] = -1.0; bub[t] = -b_plan[t]
        r = H + t                            # v >= b' − b_plan
        Aub[r, iv+t] = -1.0; Aub[r, ib+t] = 1.0; bub[r] = b_plan[t]

    bounds = ([(0.0, None)]*H + [(0.0, P_MAX)]*H + [(0.0, P_MAX)]*H
              + [(0.0, None)]*H + [(0.0, None)]*H + [(0.0, None)]*2*H
              + [(E_MIN, E_MAX)]*(H + 1))
    res = linprog(cobj, A_ub=csr_matrix(Aub), b_ub=bub,
                  A_eq=csr_matrix(Aeq), b_eq=beq, bounds=bounds,
                  method="highs")
    assert res.status == 0, f"调整LP失败: {res.message}"
    return res.x[ib:ib + H]


def exec_segment(b_cur, net_d, E, t0, t1):
    """实时执行（must/max）：计划盈余尽限充电、缺口尽限放电，
    充电只受当前盈余、功率与容量约束；剩余缺口即紧急购电（5p 结算）。"""
    c = np.zeros(t1 - t0); dd = np.zeros(t1 - t0); e = np.zeros(t1 - t0)
    for k, t in enumerate(range(t0, t1)):
        r = b_cur[t] - net_d[t]
        if r >= 0:
            c[k] = min(r, P_MAX, max(0.0, (E_MAX - E) / ETA_C))
            E += ETA_C * c[k]
        else:
            dd[k] = min(-r, P_MAX, max(0.0, (E - E_MIN) * ETA_D))
            e[k] = -r - dd[k]
            E -= dd[k] / ETA_D
    return c, dd, e, E
