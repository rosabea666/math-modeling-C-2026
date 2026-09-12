# 问题二核心程序：星期偏差校正预测 + 日前计划 LP + 尽限执行 + 结算
# 摘自 问题二/代码/q2_main_algorithm.py（完整可运行源程序见支撑材料）
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

T, DT = 144, 1.0 / 6.0
ETA_C = ETA_D = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
P_MAX = 5000.0 * DT
MULT = 5.0

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
