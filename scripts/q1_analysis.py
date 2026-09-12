# -*- coding: utf-8 -*-
"""
Q1 经济分析增强：
1) 无储能基准 C0 = sum p_t * max(L_t - V_t, 0)，节省率与购电量对比
2) 容量窗口敏感性（E_max = 1200 + 9600*s，E0=E144=6000 固定）
3) 功率上限敏感性（P_max = 5000*s kW）
   边际价值：元/(kWh容量) 与 元/(kW功率)
所有变体保持初末储能 6000 kWh 一致，不把初始电量当免费收益。
"""
import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

BASE = Path(__file__).resolve().parent.parent
DT = 1.0 / 6.0
T = 144
ETA_C = ETA_D = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
WINDOW = E_MAX - E_MIN          # 9600
P_MAX_KW = 5000.0
E0 = 6000.0


def load_day():
    df = pd.read_excel(BASE / r"data\附件1.xlsx", sheet_name=0, header=0)
    assert len(df) == 144
    return (df.iloc[:, 1].to_numpy(float),
            df.iloc[:, 2].to_numpy(float) * DT,
            df.iloc[:, 3].to_numpy(float) * DT)


def solve(price, L, V, e_min, e_max, p_kw):
    p_cap = p_kw * DT
    n = 4 * T + (T + 1)
    ig, ic, id_, iw, iE = 0, T, 2 * T, 3 * T, 4 * T
    cobj = np.zeros(n)
    cobj[ig:ig + T] = price
    Aeq = lil_matrix((2 * T + 2, n))
    beq = np.zeros(2 * T + 2)
    for t in range(T):
        Aeq[t, iE + t + 1] = 1.0
        Aeq[t, iE + t] = -1.0
        Aeq[t, ic + t] = -ETA_C
        Aeq[t, id_ + t] = 1.0 / ETA_D
        r = T + t
        Aeq[r, ig + t] = 1.0
        Aeq[r, id_ + t] = 1.0
        Aeq[r, ic + t] = -1.0
        Aeq[r, iw + t] = -1.0
        beq[r] = L[t] - V[t]
    Aeq[2 * T, iE] = 1.0
    beq[2 * T] = E0
    Aeq[2 * T + 1, iE + T] = 1.0
    beq[2 * T + 1] = E0
    bounds = ([(0.0, None)] * T + [(0.0, p_cap)] * T + [(0.0, p_cap)] * T
              + [(0.0, None)] * T + [(e_min, e_max)] * (T + 1))
    res = linprog(cobj, A_eq=csr_matrix(Aeq), b_eq=beq, bounds=bounds, method="highs")
    assert res.status == 0, f"LP失败 e=[{e_min},{e_max}] p={p_kw}: {res.message}"
    x = res.x
    return {"cost": float(res.fun), "g": x[ig:ig + T].sum(),
            "c": x[ic:ic + T].sum(), "d": x[id_:id_ + T].sum(),
            "status": res.status, "message": res.message,
            "nit": getattr(res, "nit", None)}


def main():
    price, L, V = load_day()

    # ---- 1. 无储能基准 ----
    g0 = np.maximum(L - V, 0.0)
    C0 = float(price @ g0)
    base = solve(price, L, V, E_MIN, E_MAX, P_MAX_KW)
    C1 = base["cost"]
    print("=== 无储能基准 ===")
    print(f"  C0(无储能) = {C0:.4f} 元, 购电量 G0 = {g0.sum():.4f} kWh")
    print(f"  C1(有储能) = {C1:.4f} 元, 购电量 G1 = {base['g']:.4f} kWh")
    print(f"  节省 = {C0 - C1:.4f} 元, 节省率 = {(C0 - C1) / C0 * 100:.2f}%")
    print(f"  购电量变化 = {base['g'] - g0.sum():+.4f} kWh ({(base['g'] - g0.sum()) / g0.sum() * 100:+.2f}%)")
    print(f"  求解器: status={base['status']} nit={base['nit']} msg={base['message']}")

    # ---- 2. 容量敏感性（固定下限1200 kWh，抬升上限的扩容路径；E0=E144=6000不变） ----
    # 缩容段（上限压到6000为止，保证E0可行）+ 基准 + 扩容段
    e_max_list = [6000.0, 7200.0, 8400.0, 9600.0, 10800.0, 12000.0, 13200.0]
    cap_rows = []
    for em in e_max_list:
        r = solve(price, L, V, E_MIN, em, P_MAX_KW)
        cap_rows.append({"e_max": em, "window_kwh": em - E_MIN, "cost": r["cost"],
                         "c": r["c"], "d": r["d"]})
        print(f"  E上限 {em:7.0f} (窗口 {em-E_MIN:6.0f} kWh) -> 费用 {r['cost']:.2f} 元")
    # 右侧差分 MV+(h)，多个小步长检验稳定性（LP价值函数可能有折点）
    mv_steps = {}
    for h in (60.0, 120.0, 480.0, 1200.0):
        c_h = solve(price, L, V, E_MIN, E_MAX + h, P_MAX_KW)["cost"]
        mv_steps[f"h={h:.0f}"] = {"cost": c_h, "mv": (C1 - c_h) / h}
        print(f"  MV+({h:.0f} kWh) = {(C1 - c_h) / h:.4f} 元/(kWh·日)  [费用 {c_h:.2f}]")
    # 窗口 9600->10800 区间平均收益
    c_10800 = solve(price, L, V, E_MIN, 12000.0, P_MAX_KW)["cost"]
    avg_gain = (C1 - c_10800) / 1200.0
    print(f"  窗口9600->10800 kWh: 降费 {C1 - c_10800:.2f} 元, 区间平均 {avg_gain:.4f} 元/(kWh·日)")

    # ---- 3. 功率上限敏感性 ----
    pow_scales = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    pow_rows = []
    for s in pow_scales:
        r = solve(price, L, V, E_MIN, E_MAX, P_MAX_KW * s)
        pow_rows.append({"scale": s, "p_kw": P_MAX_KW * s, "cost": r["cost"],
                         "c": r["c"], "d": r["d"]})
        print(f"  功率上限 {P_MAX_KW*s:8.0f} kW -> 费用 {r['cost']:.2f} 元")
    c_minus = solve(price, L, V, E_MIN, E_MAX, P_MAX_KW * 0.95)["cost"]
    c_plus = solve(price, L, V, E_MIN, E_MAX, P_MAX_KW * 1.05)["cost"]
    mv_pow = (c_minus - c_plus) / (2 * 0.05 * P_MAX_KW)   # 元/kW
    print(f"  功率边际价值 ≈ {mv_pow:.4f} 元/kW（5000 kW 处，±250 kW 差分）")

    out = {
        "baseline_no_storage": {"cost": C0, "purchase_kwh": float(g0.sum())},
        "with_storage": {"cost": C1, "purchase_kwh": base["g"],
                          "charge_kwh": base["c"], "discharge_kwh": base["d"],
                          "solver_status": base["status"], "solver_msg": base["message"],
                          "iterations": base["nit"]},
        "saving_yuan": C0 - C1,
        "saving_rate": (C0 - C1) / C0,
        "purchase_delta_kwh": base["g"] - float(g0.sum()),
        "capacity_sensitivity": cap_rows,
        "power_sensitivity": pow_rows,
        "capacity_mv_right_diff": mv_steps,
        "capacity_avg_gain_9600_10800": avg_gain,
        "capacity_cost_window_10800": c_10800,
        "marginal_value_per_kw": mv_pow,
    }
    with open(BASE / r"results\q1_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写出 results/q1_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
