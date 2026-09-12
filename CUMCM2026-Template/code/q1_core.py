# 问题一核心程序：确定性单日储能经济调度 LP 的装配、求解与独立复核
# 摘自 问题一/算法/q1_solve.py（完整可运行源程序见支撑材料）
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

T, DT = 144, 1.0 / 6.0
ETA_C = ETA_D = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
P_MAX = 5000.0 * DT
E0 = 6000.0

def solve_q1(price, L, V):
    """L, V: kWh/区间。返回最优解字典。"""
    # 变量布局: g[0:T], c[0:T], d[0:T], w[0:T], E[0:T+1]
    n = 4 * T + (T + 1)
    ig, ic, id_, iw, iE = 0, T, 2 * T, 3 * T, 4 * T

    cobj = np.zeros(n)
    cobj[ig:ig + T] = price

    # 等式：SOC递推 T 行 + 能量平衡 T 行 + E_0=6000 + E_144=6000
    Aeq = lil_matrix((2 * T + 2, n))
    beq = np.zeros(2 * T + 2)
    for t in range(T):
        r = t
        Aeq[r, iE + t + 1] = 1.0
        Aeq[r, iE + t] = -1.0
        Aeq[r, ic + t] = -ETA_C
        Aeq[r, id_ + t] = 1.0 / ETA_D
        r2 = T + t
        Aeq[r2, ig + t] = 1.0
        Aeq[r2, id_ + t] = 1.0
        Aeq[r2, ic + t] = -1.0
        Aeq[r2, iw + t] = -1.0
        beq[r2] = L[t] - V[t]
    Aeq[2 * T, iE] = 1.0
    beq[2 * T] = E0
    Aeq[2 * T + 1, iE + T] = 1.0
    beq[2 * T + 1] = E0

    bounds = (
        [(0.0, None)] * T +                    # g
        [(0.0, P_MAX)] * T +                   # c
        [(0.0, P_MAX)] * T +                   # d
        [(0.0, None)] * T +                    # w
        [(E_MIN, E_MAX)] * (T + 1)             # E
    )

    res = linprog(cobj, A_eq=csr_matrix(Aeq), b_eq=beq, bounds=bounds, method="highs")
    assert res.status == 0, f"LP未收敛: status={res.status} {res.message}"
    x = res.x
    return {
        "g": x[ig:ig + T], "c": x[ic:ic + T], "d": x[id_:id_ + T],
        "w": x[iw:iw + T], "E": x[iE:iE + T + 1], "cost": float(res.fun),
        "solver_msg": res.message,
    }

def validate(sol, price, L, V):
    """独立复核：能量守恒、SOC边界、日循环、功率上限、互斥、费用复算"""
    g, c, d, w, E = sol["g"], sol["c"], sol["d"], sol["w"], sol["E"]
    checks = {}
    bal = g + V + d - L - c - w
    checks["能量平衡最大残差(kWh)"] = float(np.max(np.abs(bal)))
    soc_res = E[1:] - E[:-1] - ETA_C * c + d / ETA_D
    checks["SOC递推最大残差(kWh)"] = float(np.max(np.abs(soc_res)))
    checks["E最小值(kWh)"] = float(E.min())
    checks["E最大值(kWh)"] = float(E.max())
    checks["E0(kWh)"] = float(E[0])
    checks["E144(kWh)"] = float(E[-1])
    checks["充电功率上限越界"] = float(max(0.0, c.max() - P_MAX))
    checks["放电功率上限越界"] = float(max(0.0, d.max() - P_MAX))
    both = np.minimum(c, d)
    checks["同时充放电最大重叠(kWh)"] = float(both.max())
    cost_recalc = float(np.dot(price, g))
    checks["费用复算(元)"] = cost_recalc
    checks["费用与求解器差(元)"] = abs(cost_recalc - sol["cost"])
    checks["购电总量(kWh)"] = float(g.sum())
    checks["充电总量(kWh)"] = float(c.sum())
    checks["放电总量(kWh)"] = float(d.sum())
    checks["余量总量(kWh)"] = float(w.sum())
    checks["负载总量(kWh)"] = float(L.sum())
    checks["光伏总量(kWh)"] = float(V.sum())
    ok = (checks["能量平衡最大残差(kWh)"] < 1e-6 and checks["SOC递推最大残差(kWh)"] < 1e-6
          and checks["E最小值(kWh)"] >= E_MIN - TOL and checks["E最大值(kWh)"] <= E_MAX + TOL
          and abs(E[0] - E0) < TOL and abs(E[-1] - E0) < TOL
          and checks["同时充放电最大重叠(kWh)"] < 1e-6
          and checks["费用与求解器差(元)"] < 1e-4)
    return checks, ok
