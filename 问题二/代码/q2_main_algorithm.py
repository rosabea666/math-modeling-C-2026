# -*- coding: utf-8 -*-
"""问题二 主方案 E（V_dow_N）主要模型与算法 · 自包含参考实现
================================================================
本文件把论文主方案的三层滚动决策框架整合为一份可直接运行的代码，
与 scripts/q2_model.py（模型骨架）+ scripts/q2_finalize_E.py（主方案入口）
逐位等价；独立文件，便于论文附录与评审查阅。

三层框架（每日 0:00 滚动决策，储能电量跨日连续传递）：
  ① 预测层  星期偏差校正的分布预测：
       N̂_{d,t} = m_{d,t} + γ·Δdow_{d,t} + M_{β,t}
       m_{d,t}   近 W=7 日同时刻净负载中位数
       Δdow_{d,t} 最近 N=4 个同星期日的 (净负载 − 当时基准) 均值，γ=1.0
       M_{β,t}   近 W 日窗口内"中心化净负载"的 β 经验分位数（≥0）
       参数 (β, W, γ, N) 逐月 walk-forward 选定（只用上月末之前数据）
       冷启动：d < 30 时整体回退为代表日曲线 N^fc
  ② 计划层  日前线性规划（HiGHS），显式日循环 E_H = E_0：
       min  Σ_t p_t g_t
       s.t. E_{t+1} = E_t + η_c c_t − d_t/η_d           （储能动态）
            g_t + d_t = N̂_t + c_t + w_t                （功率平衡）
            0 ≤ c_t, d_t ≤ P_MAX；E_MIN ≤ E_t ≤ E_MAX；E_H = E_0
       输出计划购电量 b_t = g_t*（提交结算）与充电意图 c_t*（执行参考）
  ③ 执行层  实测到达后按 must/max 尽限规则逐区间仿真：
       余量（b_t ≥ N_t）：c_t = min(余量, P_MAX, 可充空间)，余下弃置 w_t
       缺口（b_t < N_t）：d_t = min(缺口, P_MAX, 可放电量)，余下紧急购电 e_t
  ④ 结算（题面给定）：J = Σ p_t b_t + 5·Σ p_t e_t

评价口径：2025-02-01 起评（E = 10800 kWh，公共预热），至 12-31 共 334 天。
运行：python q2_main_algorithm.py   （约 3 s，打印并校验总费用 1395.6995 万元）
"""
import sys, io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent

# ============================ 题面给定常数 ============================
T = 144                       # 每天 144 个 10 分钟区间
DT = 1.0 / 6.0                # 区间长度（h）
ETA_C = ETA_D = 0.9           # 充/放电效率（附录）
E_MIN, E_MAX = 1200.0, 10800.0      # 储能电量运行区间 kWh（附录）
P_MAX = 5000.0 * DT           # 单区间最大充/放电量（5000 kW × 1/6 h）
MULT = 5.0                    # 紧急购电电价倍数（题面）
WARMUP_DAY = 31               # 2025-02-01 的日期索引
E_FEB1 = 10800.0              # 公共预热后 2 月 1 日的储能电量


# ============================ 数据层 ============================
def load_data():
    """附件1（代表日电价/负载/光伏）与附件2（全年实测负载/光伏）。
    时间标签为区间末端，区间 t 覆盖 (t·10min, (t+1)·10min]，功率×DT 折算电量。"""
    a1 = pd.read_excel(BASE / "data" / "附件1.xlsx", sheet_name=0, header=0)
    price = a1.iloc[:, 1].to_numpy(float)
    net_ref = (a1.iloc[:, 2].to_numpy(float) - a1.iloc[:, 3].to_numpy(float)) * DT

    xl = pd.ExcelFile(BASE / "data" / "附件2.xlsx")
    load_df = pd.read_excel(xl, sheet_name=0, header=0)
    pv_df = pd.read_excel(xl, sheet_name=1, header=0)
    dates = pd.to_datetime(load_df.iloc[:, 0]).to_numpy()
    net = (load_df.iloc[:, 1:145].to_numpy(float)
           - pv_df.iloc[:, 1:145].to_numpy(float)) * DT     # (365,144) kWh
    return dict(price=price, net_ref=net_ref, dates=dates, net=net)


# ============================ ① 预测层 ============================
def _col_quantile(X, qs):
    """X: (n, T)，qs: (T,)。逐列取对应分位数，返回 (T,)。"""
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        out[j] = np.quantile(X[:, j], qs[j])
    return out


def m_recent(d, net, W):
    """近 W 日同时刻中位数基准；历史不足 7 天时回退 30 日中位数或零向量。"""
    hist = net[max(0, d - W):d]
    if hist.shape[0] < 7:
        if d >= 30:
            return np.median(net[max(0, d - 30):d], axis=0)
        return np.zeros(net.shape[1])
    return np.median(hist, axis=0)


def delta_dow(d, net, dow_idx, n_dow_hist, W_level):
    """最近 n_dow_hist 个同星期日 (净负载 − 当时基准) 的同时刻均值。"""
    w_d = dow_idx[d]
    cand = [d_ for d_ in range(max(0, d - 90), d) if dow_idx[d_] == w_d]
    cand = cand[-n_dow_hist:]
    if not cand:
        return np.zeros(net.shape[1])
    deltas = np.zeros((len(cand), net.shape[1]))
    for i, d_ in enumerate(cand):
        deltas[i] = net[d_] - m_recent(d_, net, W_level)
    return deltas.mean(axis=0)


def forecast_dow_N(d, D, dow_idx, gamma, W_level, beta, n_dow_hist):
    """星期偏差校正预测。因果性：只用第 1..d-1 天实测；d<30 整体回退代表日。"""
    if d < 30:
        return D["net_ref"].copy()
    m = m_recent(d, D["net"], W_level)
    dlt = delta_dow(d, D["net"], dow_idx, n_dow_hist, W_level)
    base = m + gamma * dlt
    net_hist = D["net"][max(0, d - W_level):d]
    if net_hist.shape[0] < 7:
        return D["net_ref"].copy()
    resid = net_hist - np.median(net_hist, axis=0)      # 窗口内中心化净负载
    margin = _col_quantile(resid, np.full(T, beta))     # M_{β,t}
    return base + margin


# ============================ ② 计划层 LP ============================
def solve_plan_lp(price, N_hat, E_start):
    """日计划 LP：min Σ p_t g_t，显式日循环 E_H = E_start。"""
    H = len(N_hat)
    n = 4 * H + (H + 1)
    ig, ic, id_, iw, iE = 0, H, 2 * H, 3 * H, 4 * H
    cobj = np.zeros(n)
    cobj[ig:ig + H] = price

    Aeq = lil_matrix((2 * H + 2, n))
    beq = np.zeros(2 * H + 2)
    for t in range(H):                                   # 储能动态
        Aeq[t, iE + t + 1] = 1.0
        Aeq[t, iE + t] = -1.0
        Aeq[t, ic + t] = -ETA_C
        Aeq[t, id_ + t] = 1.0 / ETA_D
        r = H + t                                        # 功率平衡
        Aeq[r, ig + t] = 1.0
        Aeq[r, id_ + t] = 1.0
        Aeq[r, ic + t] = -1.0
        Aeq[r, iw + t] = -1.0
        beq[r] = N_hat[t]
    Aeq[2 * H, iE] = 1.0                                 # 日初电量
    beq[2 * H] = E_start
    Aeq[2 * H + 1, iE + H] = 1.0                         # 日循环 E_H = E_0
    beq[2 * H + 1] = E_start

    bounds = ([(0.0, None)] * H + [(0.0, P_MAX)] * H + [(0.0, P_MAX)] * H
              + [(0.0, None)] * H + [(E_MIN, E_MAX)] * H + [(E_MIN, E_MAX)])
    res = linprog(cobj, A_eq=csr_matrix(Aeq), b_eq=beq, bounds=bounds,
                  method="highs")
    assert res.status == 0, f"LP 失败: {res.message}"
    x = res.x
    return x[ig:ig + H], x[ic:ic + H]                    # b（计划购电）, c*（充电意图）


# ============================ ③ 执行层（must/max 尽限规则） ============================
def execute(b, net_act, E_start):
    """逐区间物理平衡仿真。
    余量侧 max：余量在物理上限内尽量充（不受计划充电意图约束），余下弃置；
    缺口侧 must：缺口由储能按物理上限补足，余下按 5 倍电价紧急购电。"""
    c = np.zeros(T); d = np.zeros(T); e = np.zeros(T); w = np.zeros(T)
    E = np.empty(T + 1); E[0] = E_start
    for t in range(T):
        net = b[t] - net_act[t]                          # >0 余量；<0 缺口
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


# ============================ ④ 结算（题面给定） ============================
def settle(b, e, price):
    """J = Σ p_t·b_t + 5·Σ p_t·e_t。计划电量全量计费（弃置不退费）。"""
    return float((b * price).sum()), float(MULT * (e * price).sum())


# ============================ 逐月 walk-forward 参数日程 ============================
# (β, W, γ, N_DOW_HIST)，由 q2_dow_N_joint.py 用上月末之前数据联合选定
SCHED = {m: (0.70, 7, 1.0, 4) for m in (2, 3)}
for m in range(4, 13):
    SCHED[m] = (0.75, 7, 1.0, 4)


# ============================ 全年回放 ============================
def replay(D):
    price, net, dates = D["price"], D["net"], D["dates"]
    dow_idx = np.array([pd.Timestamp(x).dayofweek for x in dates])
    months = np.array([x.astype("datetime64[M]").astype(int) % 12 + 1
                       for x in dates])
    nd = len(dates)
    B = np.zeros((nd, T)); EM = np.zeros((nd, T))
    SOC0 = np.zeros(nd); SOC1 = np.zeros(nd)
    E = float(E_FEB1)
    for d in range(WARMUP_DAY, nd):
        beta, W, gamma, n_dow = SCHED[months[d]]
        N_hat = forecast_dow_N(d, D, dow_idx, gamma, W, beta, n_dow)
        b, _ = solve_plan_lp(price, N_hat, E)
        _, _, e, _, Etraj = execute(b, net[d], E)
        B[d], EM[d] = b, e
        SOC0[d], SOC1[d] = Etraj[0], Etraj[-1]
        E = Etraj[-1]                                    # 跨日耦合
    return B, EM, SOC0, SOC1


def main():
    D = load_data()
    B, EM, SOC0, SOC1 = replay(D)
    idx = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]
    plan = float((B[idx] * D["price"]).sum()) / 1e4
    em = float(MULT * (EM[idx] * D["price"]).sum()) / 1e4
    print(f"评价期 {len(idx)} 天（2025-02-01 至 12-31）")
    print(f"计划费 {plan:.4f} 万元 | 紧急费 {em:.4f} 万元 | "
          f"总费 {plan + em:.4f} 万元")
    print(f"年末储能电量 {SOC1[-1]:.2f} kWh")
    assert abs(plan + em - 1395.6995) < 0.01, "与主方案已核验结果不一致"


if __name__ == "__main__":
    main()
