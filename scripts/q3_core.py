# -*- coding: utf-8 -*-
"""问题三模型链：多时刻预报下的滚动合同调整。

按论文 §7 的口径实现，四个可独立开关的部件：
  ① 预报融合 : 用附件 3 的 0:00 整点预报，经分档收缩系数 κ_g 修正问题二净负载预测
  ② 计划层   : 带日循环边界的日前 LP（与问题二同）
  ③ 调整层   : 6:00/12:00/18:00 对剩余时段重解合同，偏差对 0:00 计划双向计价
  ④ 执行层   : 尽限充放电，按"当期生效合同"交付，缺口 5p 紧急购电

结算（题面口径，与论文式 (7.3) 等价）：
  J = Σ p_t b'_t  +  ½ Σ p_t |b'_t − b_plan_t|  +  5 Σ p_t e_t
      └ 合同费 ┘     └──── 调整附加费 ────┘     └ 紧急费 ┘
其中 b' 为最终生效合同。无调整时 b' = b_plan，退化为问题二的 Σ p b_plan + 5Σ p e。

方案矩阵（论文表 tab:q3-matrix）：
  A0  = 不用预报通道、不允许调整  ≡ 问题二主方案 E（1395.6995 万元）
  A0K = 用预报通道、不允许调整
  B0  = 不用预报通道、允许调整
  B1  = 两者都用（论文主方案）
  B3  = 每个调整时刻都用当期附件 3 预报重建预测

只读 data/附件*.xlsx；写文件一律由 q3_finalize.py 负责。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

BASE = Path(__file__).resolve().parent.parent

# ============================ 题面常数（与问题二一致） ============================
T = 144
DT = 1.0 / 6.0
ETA_C = 0.9
ETA_D = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
P_MAX = 5000.0 * DT
E_INIT = 6000.0
MULT = 5.0
TOL = 1e-6
HALF = 0.5          # 下调违约电价系数（题面 50%）
OVER = 1.5          # 上调超出部分系数（题面 1.5 倍）

WARMUP_DAY = 31     # 2025-02-01 的日期索引
E_FEB1 = 10800.0    # 公共预热口径：2/1 起评电量

TAUS = (6, 12, 18)                        # 预报/调整时刻（小时）
TAU_IDX = {0: 0, 6: 1, 12: 2, 18: 3}
BUCKETS = ((0, 36), (36, 72), (72, 144))  # 预报时长分档 h∈[0,6)/[6,12)/[12,24)
KAPPA_WIN = 30
KAPPA_MIN_N = 7

# 主方案 E 的逐月 walk-forward 日程（与 q2_dow_N_joint / q2_finalize_E 一致）
SCHED = {m: (0.70, 7, 1.0, 4) for m in (2, 3)}
for _m in range(4, 13):
    SCHED[_m] = (0.75, 7, 1.0, 4)


# ============================ 数据层 ============================
def load_data() -> dict:
    a1 = pd.read_excel(BASE / "data" / "附件1.xlsx", header=0)
    price = a1.iloc[:, 1].to_numpy(float)
    net_ref = (a1.iloc[:, 2].to_numpy(float) - a1.iloc[:, 3].to_numpy(float)) * DT

    xl = pd.ExcelFile(BASE / "data" / "附件2.xlsx")
    load_df = pd.read_excel(xl, sheet_name=0, header=0)
    pv_df = pd.read_excel(xl, sheet_name=1, header=0)
    dates = pd.to_datetime(load_df.iloc[:, 0]).to_numpy()
    load = load_df.iloc[:, 1:145].to_numpy(float) * DT
    pv = pv_df.iloc[:, 1:145].to_numpy(float) * DT
    net = load - pv

    a3 = pd.read_excel(BASE / "data" / "附件3.xlsx", header=0)
    assert a3.shape[0] == len(dates) * 4, f"附件3 行数异常: {a3.shape}"
    slot = a3.iloc[:, 1].astype(str).str.replace(":00", "", regex=False).astype(int)
    order = {"0": 0, "6": 1, "12": 2, "18": 3}
    fcast = np.zeros((len(dates), 4, 24))
    for i in range(len(dates)):
        row = [int(slot.iloc[i * 4 + j]) for j in range(4)]
        assert row == [0, 6, 12, 18], f"附件3 时刻顺序异常 @row{i}: {row}"
        for j in range(4):
            fcast[i, order[str(row[j])]] = a3.iloc[i * 4 + j, 2:26].to_numpy(float)

    assert price.shape[0] == T and net_ref.shape[0] == T and load.shape[1] == T
    return dict(price=price, net_ref=net_ref, dates=dates, load=load, pv=pv,
                net=net, fcast=fcast, pv_ref=a1.iloc[:, 3].to_numpy(float))


def pv_interval(D: dict, d: int, tau: int) -> np.ndarray:
    """附件 3 在 τ 发布的整点预报 → 当日 144 个区间的光伏电量（kWh）。

    τ=0 端点取 0（尚无实测）；τ>0 以当日已过去的平均光伏功率水平锚定端点
    （而非瞬时值），以削弱单点噪声；当日以外的小时（>24:00）不参与。
    相邻整点间线性插值，区间 t 取两端均值乘 Δt。
    """
    F = D["fcast"][d, TAU_IDX[tau]]
    if tau == 0:
        anchor = 0.0
    else:
        anchor = float(D["pv"][d, :6 * tau].mean()) / DT   # 已过去时段的平均功率 kW
    hours = np.arange(tau, 25.0)                          # 仅到当日 24:00
    vals = np.concatenate([[anchor], F[:24 - tau]])
    out = np.zeros(T)
    t0 = 6 * tau
    mins = np.arange(t0 * 10, 1440 + 10, 10)
    g = np.interp(mins / 60.0, hours, vals)
    out[t0:] = 0.5 * (g[:-1] + g[1:]) * DT
    return np.clip(out, 0.0, None)


# ============================ 预测层（问题二预测器，逐字复刻） ============================
def _col_quantile(X: np.ndarray, qs: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        out[j] = np.quantile(X[:, j], qs[j])
    return out


def dow_index(dates) -> np.ndarray:
    return np.array([pd.Timestamp(d).dayofweek for d in dates])


def m_recent(d: int, X: np.ndarray, W: int) -> np.ndarray:
    hist = X[max(0, d - W):d]
    if hist.shape[0] < 7:
        if d >= 30:
            return np.median(X[max(0, d - 30):d], axis=0)
        return np.zeros(X.shape[1])
    return np.median(hist, axis=0)


def delta_dow(d: int, X: np.ndarray, dow_idx: np.ndarray,
              n_dow_hist: int, W_level: int) -> np.ndarray:
    w_d = dow_idx[d]
    cand = [d_ for d_ in range(max(0, d - 90), d) if dow_idx[d_] == w_d]
    cand = cand[-n_dow_hist:]
    if not cand:
        return np.zeros(X.shape[1])
    deltas = np.array([X[d_] - m_recent(d_, X, W_level) for d_ in cand])
    return deltas.mean(axis=0)


def forecast_q2(d: int, D: dict, dow_idx: np.ndarray, gamma: float,
                W: int, beta: float, n_dow: int) -> np.ndarray:
    """问题二主方案 E 的净负载预测（近期水平 + 星期偏差 + 分位裕量）。"""
    if d < 30:
        return D["net_ref"].copy()
    base = m_recent(d, D["net"], W) + gamma * delta_dow(d, D["net"], dow_idx, n_dow, W)
    hist = D["net"][max(0, d - W):d]
    if hist.shape[0] < 7:
        return D["net_ref"].copy()
    resid = hist - np.median(hist, axis=0)
    margin = _col_quantile(resid, np.full(T, beta))
    return base + margin


def pv_hist(d: int, D: dict, dow_idx: np.ndarray, gamma: float,
            W: int, n_dow: int) -> np.ndarray:
    """问题二预测器隐含的光伏分量（同结构，不含裕量）。"""
    if d < 30:
        return D["pv_ref"] * DT            # 代表日光伏区间电量
    return m_recent(d, D["pv"], W) + gamma * delta_dow(d, D["pv"], dow_idx, n_dow, W)


def estimate_kappa(d: int, D: dict, dow_idx: np.ndarray, gamma: float,
                   W: int, beta: float, n_dow: int) -> np.ndarray:
    """分档收缩系数 κ_g：过去 K=30 天的因果滚动回归，样本不足则退回 0。"""
    lo = max(30, d - KAPPA_WIN)
    if d - lo < KAPPA_MIN_N:
        return np.zeros(len(BUCKETS))
    num = np.zeros(len(BUCKETS))
    den = np.zeros(len(BUCKETS))
    for s in range(lo, d):
        if s < 30:
            center = D["net_ref"].copy()
        else:
            center = (m_recent(s, D["net"], W)
                      + gamma * delta_dow(s, D["net"], dow_idx, n_dow, W))
        r = D["net"][s] - center          # 对"中心预测"（不含分位裕量）取残差
        x = pv_interval(D, s, 0) - pv_hist(s, D, dow_idx, gamma, W, n_dow)
        for gi, (a, z) in enumerate(BUCKETS):
            num[gi] += float(np.sum(r[a:z] * x[a:z]))
            den[gi] += float(np.sum(x[a:z] ** 2))
    return np.clip(-num / np.maximum(den, 1e-12), 0.0, 1.0)


def fuse_forecast(N_q2: np.ndarray, V_fc: np.ndarray, V_hist: np.ndarray,
                  kg: np.ndarray) -> np.ndarray:
    """式 (7.2)：N̂ = N_q2 − κ_g [Ṽ(0) − V̂_hist]。"""
    corr = np.zeros(T)
    for gi, (a, z) in enumerate(BUCKETS):
        corr[a:z] = kg[gi] * (V_fc[a:z] - V_hist[a:z])
    return N_q2 - corr


# ============================ 计划层 LP ============================
def solve_plan(price, N_hat, E_start, terminal="cycle"):
    """min Σ p_t b_t，约束含能量平衡、SOC 界、功率上限，末式 E_H = E_start。"""
    H = len(N_hat)
    n = 5 * H + 1
    ib, ic, id_, iw, iE = 0, H, 2 * H, 3 * H, 4 * H
    cobj = np.zeros(n)
    cobj[ib:ib + H] = price

    Aeq = np.zeros((2 * H + 2, n))
    beq = np.zeros(2 * H + 2)
    for t in range(H):
        Aeq[t, iE + t + 1] = 1.0
        Aeq[t, iE + t] = -1.0
        Aeq[t, ic + t] = -ETA_C
        Aeq[t, id_ + t] = 1.0 / ETA_D
        r = H + t
        Aeq[r, ib + t] = 1.0
        Aeq[r, id_ + t] = 1.0
        Aeq[r, ic + t] = -1.0
        Aeq[r, iw + t] = -1.0
        beq[r] = N_hat[t]
    Aeq[2 * H, iE] = 1.0
    beq[2 * H] = E_start
    nrow = 2 * H + 1
    if terminal == "cycle":
        Aeq[nrow, iE + H] = 1.0
        beq[nrow] = E_start
        nrow += 1

    bounds = ([(0.0, None)] * H + [(0.0, P_MAX)] * H + [(0.0, P_MAX)] * H
              + [(0.0, None)] * H + [(E_MIN, E_MAX)] * H + [(E_MIN, E_MAX)])
    res = linprog(cobj, A_eq=csr_matrix(Aeq[:nrow]), b_eq=beq[:nrow],
                  bounds=bounds, method="highs")
    assert res.status == 0, f"计划 LP 失败: {res.message}"
    x = res.x
    return dict(b=x[ib:ib + H], c=x[ic:ic + H], d=x[id_:id_ + H],
                w=x[iw:iw + H], E=x[iE:iE + H + 1], obj=float(res.fun))


# ============================ 调整层 LP（双向违约价） ============================
def solve_adjust(price, N_hat, E_tau, b_plan, E_cycle):
    """对剩余时段重解合同：min Σp b' + ½Σp(u+v)。

    u ≥ b_plan − b'，v ≥ b' − b_plan（偏差一律对 0:00 计划计量，故连续调整不重复计费）；
    供给约束 b' + d ≥ N̂ + c，即合同与放电必须覆盖预测需求，超量为弃置；
    紧急购电不在调整层内决策，只在执行层按实测缺口发生（按 5p 结算）。
    末式 E_H = E_cycle 为计划层的日循环边界。
    """
    H = len(N_hat)
    ib, iu, iv, ic, id_, iE = 0, H, 2 * H, 3 * H, 4 * H, 5 * H
    n = 5 * H + H + 1
    obj = np.zeros(n)
    obj[ib:ib + H] = price
    obj[iu:iu + H] = HALF * price
    obj[iv:iv + H] = HALF * price

    Aeq = np.zeros((H + 2, n))
    beq = np.zeros(H + 2)
    for t in range(H):
        Aeq[t, iE + t + 1] = 1.0
        Aeq[t, iE + t] = -1.0
        Aeq[t, ic + t] = -ETA_C
        Aeq[t, id_ + t] = 1.0 / ETA_D
    Aeq[H, iE] = 1.0
    beq[H] = E_tau
    Aeq[H + 1, iE + H] = 1.0
    beq[H + 1] = E_cycle

    Aub = np.zeros((3 * H, n))
    bub = np.zeros(3 * H)
    for t in range(H):
        Aub[t, ib + t] = -1.0            # b' + d − c ≥ N̂  （供给覆盖需求）
        Aub[t, ic + t] = 1.0
        Aub[t, id_ + t] = -1.0
        bub[t] = -N_hat[t]
        Aub[H + t, ib + t] = 1.0         # u ≥ b_plan − b'
        Aub[H + t, iu + t] = -1.0
        bub[H + t] = b_plan[t]
        Aub[2 * H + t, ib + t] = -1.0    # v ≥ b' − b_plan
        Aub[2 * H + t, iv + t] = -1.0
        bub[2 * H + t] = -b_plan[t]

    bounds = ([(0.0, None)] * H + [(0.0, None)] * H + [(0.0, None)] * H
              + [(0.0, P_MAX)] * H + [(0.0, P_MAX)] * H
              + [(E_MIN, E_MAX)] + [(E_MIN, E_MAX)] * H)
    res = linprog(obj, A_eq=csr_matrix(Aeq), b_eq=beq, A_ub=csr_matrix(Aub),
                  b_ub=bub, bounds=bounds, method="highs")
    assert res.status == 0, f"调整 LP 失败: {res.message}"
    x = res.x
    return dict(b=x[ib:ib + H], u=x[iu:iu + H], v=x[iv:iv + H],
                c=x[ic:ic + H], d=x[id_:id_ + H], E=x[iE:iE + H + 1],
                obj=float(res.fun))


# ============================ 执行层（尽限充放电，段长任意） ============================
def execute_seg(net_act, b_cur, E_start):
    H = len(net_act)
    c = np.zeros(H); d = np.zeros(H); e = np.zeros(H); w = np.zeros(H)
    E = np.empty(H + 1); E[0] = E_start
    for t in range(H):
        net = b_cur[t] - net_act[t]
        if net >= 0:
            avail = max(0.0, (E_MAX - E[t]) / ETA_C)
            c[t] = min(net, P_MAX, avail)
            w[t] = net - c[t]
        else:
            deficit = -net
            avail = max(0.0, (E[t] - E_MIN) * ETA_D)
            d[t] = min(deficit, P_MAX, avail)
            e[t] = deficit - d[t]
        E[t + 1] = E[t] + ETA_C * c[t] - d[t] / ETA_D
    return c, d, e, w, E


# ============================ 单日回放 ============================
def replay_day(d: int, D: dict, dow_idx, E_start: float, *, use_channel: bool,
               allow_adjust: bool, full_fusion: bool = False, beta=None, W=None,
               gamma=None, n_dow=None, kg=None) -> dict:
    date = pd.Timestamp(D["dates"][d])
    if beta is None:
        beta, W, gamma, n_dow = SCHED[int(date.month)]
    price = D["price"]
    N_q2 = forecast_q2(d, D, dow_idx, gamma, W, beta, n_dow)
    V_h = pv_hist(d, D, dow_idx, gamma, W, n_dow)

    if use_channel:
        if kg is None:
            kg = estimate_kappa(d, D, dow_idx, gamma, W, beta, n_dow)
        N_hat = fuse_forecast(N_q2, pv_interval(D, d, 0), V_h, kg)
    else:
        N_hat = N_q2

    plan = solve_plan(price, N_hat, E_start)
    b_plan = plan["b"]
    b_fin = b_plan.copy()

    c = np.zeros(T); dd = np.zeros(T); em = np.zeros(T); w = np.zeros(T)
    E = np.empty(T + 1); E[0] = E_start
    t_cursor = 0
    E_now = E_start

    for tau in (TAUS if allow_adjust else ()):
        t0 = 6 * tau
        if t0 <= t_cursor:
            continue
        cs, ds, es, ws, Es = execute_seg(D["net"][d, t_cursor:t0],
                                         b_fin[t_cursor:t0], E_now)
        c[t_cursor:t0] = cs; dd[t_cursor:t0] = ds
        em[t_cursor:t0] = es; w[t_cursor:t0] = ws
        E[t_cursor:t0 + 1] = Es
        E_now = float(Es[-1]); t_cursor = t0

        if full_fusion:
            V_fc = pv_interval(D, d, tau)
            lead = np.arange(T) * DT - tau          # 距发布时刻的时长（h）
            corr = np.zeros(T)
            for gi, (a, z) in enumerate(BUCKETS):
                m = (lead >= a) & (lead < z) & (np.arange(T) >= t_cursor)
                corr[m] = kg[gi] * (V_fc[m] - V_h[m])
            N_use = (N_q2 - corr)[t_cursor:]
        else:
            N_use = N_hat[t_cursor:]
        sol = solve_adjust(price[t_cursor:], N_use, E_now,
                           b_plan[t_cursor:], E_start)
        b_fin[t_cursor:] = sol["b"]

    cs, ds, es, ws, Es = execute_seg(D["net"][d, t_cursor:],
                                     b_fin[t_cursor:], E_now)
    c[t_cursor:] = cs; dd[t_cursor:] = ds
    em[t_cursor:] = es; w[t_cursor:] = ws
    E[t_cursor:] = Es

    plan_cost = float(np.sum(price * b_plan))
    contract_cost = float(np.sum(price * b_fin))
    adj_fee = float(HALF * np.sum(price * np.abs(b_fin - b_plan)))
    em_cost = float(MULT * np.sum(price * em))
    return dict(date=date, b_plan=b_plan, b_fin=b_fin, c=c, d=dd, e=em, w=w, E=E,
                N_hat=N_hat, N_q2=N_q2, kg=kg, plan_cost=plan_cost,
                contract_cost=contract_cost, adj_fee=adj_fee, em_cost=em_cost,
                total=contract_cost + adj_fee + em_cost)


# ============================ 全年回放 ============================
def replay(D: dict, *, use_channel: bool, allow_adjust: bool,
           full_fusion: bool = False, label: str = "", verbose: bool = True) -> dict:
    dow_idx = dow_index(D["dates"])
    nd = len(D["dates"])
    B = np.zeros((nd, T)); BF = np.zeros((nd, T)); C = np.zeros((nd, T))
    DD = np.zeros((nd, T)); EM = np.zeros((nd, T)); WW = np.zeros((nd, T))
    EE = np.zeros((nd, T + 1)); SOC0 = np.zeros(nd); SOC1 = np.zeros(nd)
    kgs = np.zeros((nd, len(BUCKETS)))
    pc = np.zeros(nd); cc = np.zeros(nd); ac = np.zeros(nd); ec = np.zeros(nd)

    E = E_FEB1
    for d in range(WARMUP_DAY, nd):
        beta, W, gamma, n_dow = SCHED[int(pd.Timestamp(D["dates"][d]).month)]
        if use_channel:
            kg = estimate_kappa(d, D, dow_idx, gamma, W, beta, n_dow)
        else:
            kg = np.zeros(len(BUCKETS))
        R = replay_day(d, D, dow_idx, E, use_channel=use_channel,
                       allow_adjust=allow_adjust, full_fusion=full_fusion,
                       beta=beta, W=W, gamma=gamma, n_dow=n_dow, kg=kg)
        B[d], BF[d], C[d], DD[d], EM[d], WW[d], EE[d] = (
            R["b_plan"], R["b_fin"], R["c"], R["d"], R["e"], R["w"], R["E"])
        SOC0[d], SOC1[d] = float(R["E"][0]), float(R["E"][-1])
        kgs[d] = R["kg"]
        pc[d] = R["plan_cost"]; cc[d] = R["contract_cost"]
        ac[d] = R["adj_fee"]; ec[d] = R["em_cost"]
        E = R["E"][-1]

    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]
    tot = dict(计划购电费_万元=float(pc[idx].sum()) / 1e4,
               最终合同费_万元=float(cc[idx].sum()) / 1e4,
               调整附加费_万元=float(ac[idx].sum()) / 1e4,
               紧急费_万元=float(ec[idx].sum()) / 1e4)
    tot["总费_万元"] = tot["最终合同费_万元"] + tot["调整附加费_万元"] + tot["紧急费_万元"]
    out = dict(B=B, BF=BF, C=C, D=DD, EM=EM, W=WW, E=EE, SOC0=SOC0, SOC1=SOC1,
               KGS=kgs, idx=idx, costs=tot, label=label)
    if verbose:
        print(f"[{label:16s}] 合同 {tot['最终合同费_万元']:9.4f} | 附加 "
              f"{tot['调整附加费_万元']:8.4f} | 紧急 {tot['紧急费_万元']:9.4f} "
              f"| 总费 {tot['总费_万元']:9.4f} 万元")
    return out


def audit(R, D: dict, name="") -> dict:
    idx = R["idx"]
    # 执行按"最终生效合同"交付，故平衡式用 BF 而非 B
    res = R["BF"][idx] + R["D"][idx] + R["EM"][idx] - D["net"][idx] - R["C"][idx] - R["W"][idx]
    E = R["E"][idx]
    return dict(name=name,
                平衡残差=float(np.max(np.abs(res))),
                充放电重叠=float(np.minimum(R["C"], R["D"])[idx].max()),
                跨日断裂=float(np.max(np.abs(R["SOC1"][idx[:-1]] - R["SOC0"][idx[1:]]))),
                SOC下限=float(E.min()), SOC上限=float(E.max()),
                功率越界=float(max(0.0, (R["C"][idx] - P_MAX).max(),
                                   (R["D"][idx] - P_MAX).max())))


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    D = load_data()
    dd = int(np.where(D["dates"] == np.datetime64("2025-09-23"))[0][0])
    print("=== 附件3 对齐抽检（kW）2025-09-23 ===")
    print(" 附件3 12:00 发布 前 8 小时预报:", np.round(D["fcast"][dd, 2, :8], 1))
    print(" 附件2 同日 13:00-20:00 实测光伏功率:",
          np.round([float(D["pv"][dd, 6 * h - 1]) / DT for h in range(13, 21)], 1))
    print("\n=== 2×2 方案矩阵 ===")
    A0 = replay(D, use_channel=False, allow_adjust=False, label="A0")
    print(f"  A0 与问题二主方案 1395.6995 之差: {A0['costs']['总费_万元'] - 1395.6995:+.4f} 万元")
    A0K = replay(D, use_channel=True, allow_adjust=False, label="A0K")
    B0 = replay(D, use_channel=False, allow_adjust=True, label="B0")
    B1 = replay(D, use_channel=True, allow_adjust=True, label="B1(主方案)")
    B3 = replay(D, use_channel=True, allow_adjust=True, full_fusion=True, label="B3(全融合)")
    print("\n=== 一致性审计 ===")
    for R in (A0, A0K, B0, B1, B3):
        print(" ", audit(R, D, R["label"]))
