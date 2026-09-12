# -*- coding: utf-8 -*-
"""
2026 CUMCM C题 问题2：全年滚动日计划 LP + 因果执行仿真（唯一复现入口）

信息口径（K1，分析报告第6/10节）：
  - 每天 0:00 制定计划时仅可见：附件1代表日（负载、光伏预测功率）+ 固定电价 + 当前实际SOC；
    当天实测负载/光伏（附件2）在计划时不可见。
  - 计划 LP：以附件1预测为输入，E_0=E_144=E_start（日循环防短视；电价每日相同，无跨日套利损失）。
  - 执行：区间 net = b_t + V_act - L_act；net>=0 时充电 min(c*, net)（SOC截断），富余为 w；
    net<0 时放电 min(-net, P_MAX, 可放电量)，仍不足部分为紧急购电 e_t（5倍电价）。
    执行器只读当前实际值，无前瞻。
  - SOC 跨日连续，2025-01-01 0:00 = 6000 kWh；1月为预热，2.1-12.31 为正式评价期。
  - 费用：C2 = Σ p_t·b_t + 5·Σ p_t·e_t（正式评价期内）。
"""
import sys
import io
import json
import hashlib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
import openpyxl

BASE = Path(r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23")
DT = 1.0 / 6.0
T = 144
ETA_C = ETA_D = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
P_MAX = 5000.0 * DT
E_INIT = 6000.0
EMERG_MULT = 5.0
TOL = 1e-6
EVAL_START = pd.Timestamp("2025-02-01")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs():
    a1 = pd.read_excel(BASE / r"data\附件1.xlsx", sheet_name=0, header=0)
    assert len(a1) == 144
    price = a1.iloc[:, 1].to_numpy(float)
    load_fc = a1.iloc[:, 2].to_numpy(float) * DT
    pv_fc = a1.iloc[:, 3].to_numpy(float) * DT

    xl = pd.ExcelFile(BASE / r"data\附件2.xlsx")
    print("附件2工作表:", xl.sheet_names)
    load_df = pd.read_excel(xl, sheet_name=0, header=0)
    pv_df = pd.read_excel(xl, sheet_name=1, header=0)
    assert len(load_df) == 365 and len(pv_df) == 365
    dates = pd.to_datetime(load_df.iloc[:, 0])
    assert (dates == pd.to_datetime(pv_df.iloc[:, 0])).all()
    assert dates.iloc[0] == pd.Timestamp("2025-01-01") and dates.iloc[-1] == pd.Timestamp("2025-12-31")
    load_act = load_df.iloc[:, 1:145].to_numpy(float) * DT   # (365,144) kWh
    pv_act = pv_df.iloc[:, 1:145].to_numpy(float) * DT
    assert load_act.shape == (365, 144) and pv_act.shape == (365, 144)
    # 校验: 光伏夜间(列0-5, 0:10-1:00)应基本为0
    assert np.abs(pv_act[:, :5]).max() < 1.0, "光伏表夜间非零，表序可能有误"
    assert np.all(load_act >= 0) and np.all(pv_act >= 0) and np.all(price >= 0)
    return price, load_fc, pv_fc, dates, load_act, pv_act


# ---------- 日计划 LP（与Q1同族，E_0=E_144=E_start） ----------
def build_lp_matrices():
    n = 4 * T + (T + 1)
    ig, ic, id_, iw, iE = 0, T, 2 * T, 3 * T, 4 * T
    Aeq = lil_matrix((2 * T + 2, n))
    for t in range(T):
        Aeq[t, iE + t + 1] = 1.0
        Aeq[t, iE + t] = -1.0
        Aeq[t, ic + t] = -ETA_C
        Aeq[t, id_ + t] = 1.0 / ETA_D
        r2 = T + t
        Aeq[r2, ig + t] = 1.0
        Aeq[r2, id_ + t] = 1.0
        Aeq[r2, ic + t] = -1.0
        Aeq[r2, iw + t] = -1.0
    Aeq[2 * T, iE] = 1.0
    Aeq[2 * T + 1, iE + T] = 1.0
    bounds = ([(0.0, None)] * T + [(0.0, P_MAX)] * T + [(0.0, P_MAX)] * T
              + [(0.0, None)] * T + [(E_MIN, E_MAX)] * (T + 1))
    return csr_matrix(Aeq), bounds, n, (ig, ic, id_, iw, iE)


def plan_day(price, L_fc, V_fc, E_start, Aeq, bounds, n, idx):
    ig, ic, id_, iw, iE = idx
    beq = np.zeros(2 * T + 2)
    beq[T:2 * T] = L_fc - V_fc
    beq[2 * T] = E_start
    beq[2 * T + 1] = E_start
    cobj = np.zeros(n)
    cobj[ig:ig + T] = price
    res = linprog(cobj, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    assert res.status == 0, f"日计划LP失败: {res.message}"
    x = res.x
    return x[ig:ig + T], x[ic:ic + T], x[id_:id_ + T]


# ---------- 因果执行器 ----------
def execute_day(b, c_plan, L_act, V_act, E_start):
    c = np.zeros(T)
    d = np.zeros(T)
    e = np.zeros(T)
    w = np.zeros(T)
    E = np.empty(T + 1)
    E[0] = E_start
    for t in range(T):
        net = b[t] + V_act[t] - L_act[t]
        if net >= 0:
            c_t = min(c_plan[t], net, P_MAX, (E_MAX - E[t]) * 1.0 / ETA_C if ETA_C > 0 else 0.0)
            # 充电受SOC上限约束（输入侧）：E+ETA_C*c <= E_MAX
            c_t = min(c_t, max(0.0, (E_MAX - E[t]) / ETA_C))
            c[t] = c_t
            w[t] = net - c_t
        else:
            deficit = -net
            d_t = min(deficit, P_MAX, max(0.0, (E[t] - E_MIN) * ETA_D))
            d[t] = d_t
            e[t] = deficit - d_t
        E[t + 1] = E[t] + ETA_C * c[t] - d[t] / ETA_D
    return c, d, e, w, E


def merge_events(e):
    """把相邻紧急区间合并为事件: 返回 [(start_idx, end_idx_excl, energy)]"""
    events = []
    t = 0
    while t < T:
        if e[t] > TOL:
            s = t
            tot = 0.0
            while t < T and e[t] > TOL:
                tot += e[t]
                t += 1
            events.append((s, t, tot))
        else:
            t += 1
    return events


def event_label(s, e_):
    def fmt(i):
        mins = i * 10
        if mins >= 1440:
            return f"{(mins - 1440) // 60}:{(mins - 1440) % 60:02d}+1"
        return f"{mins // 60}:{mins % 60:02d}"
    return f"{fmt(s)}-{fmt(e_)}"


def main():
    price, load_fc, pv_fc, dates, load_act, pv_act = load_inputs()
    Aeq, bounds, n, idx = build_lp_matrices()

    ndays = len(dates)
    B = np.zeros((ndays, T))   # 计划购电
    C = np.zeros((ndays, T))   # 实际充电
    D = np.zeros((ndays, T))   # 实际放电
    EM = np.zeros((ndays, T))  # 紧急购电
    W = np.zeros((ndays, T))   # 余量
    SOC0 = np.zeros(ndays)
    SOC1 = np.zeros(ndays)

    E_start = E_INIT
    for day in range(ndays):
        b, c_plan, _ = plan_day(price, load_fc, pv_fc, E_start, Aeq, bounds, n, idx)
        c, d, e, w, E = execute_day(b, c_plan, load_act[day], pv_act[day], E_start)
        B[day], C[day], D[day], EM[day], W[day] = b, c, d, e, w
        SOC0[day] = E[0]
        SOC1[day] = E[-1]
        E_start = E[-1]
        if day % 60 == 0:
            print(f"  仿真 {dates.iloc[day].date()} ...")

    # ---------- 校验 ----------
    checks = {}
    # 执行层能量守恒: b + V + d + e = L + c + w （紧急购电e是供给项）（逐区间）
    res = B + pv_act + D + EM - load_act - C - W
    checks["执行能量平衡最大残差(kWh)"] = float(np.max(np.abs(res)))
    checks["SOC最小值(kWh)"] = float(min(SOC0.min(), SOC1.min()))
    checks["SOC最大值(kWh)"] = float(max(SOC0.max(), SOC1.max()))
    # SOC 递推复算
    soc_calc = SOC0 + (ETA_C * C - D / ETA_D).cumsum(axis=1)[:, -1]
    checks["日末SOC递推最大偏差(kWh)"] = float(np.max(np.abs(soc_calc - SOC1)))
    # 跨日连续
    checks["跨日SOC断裂(kWh)"] = float(np.max(np.abs(SOC1[:-1] - SOC0[1:])))
    checks["充电越界"] = float(max(0.0, C.max() - P_MAX))
    checks["放电越界"] = float(max(0.0, D.max() - P_MAX))
    checks["紧急购电为负"] = float(max(0.0, -EM.min()))

    # ---------- 费用（正式评价期 2.1-12.31） ----------
    eval_mask = (dates >= EVAL_START).to_numpy()
    plan_cost_day = (B * price).sum(axis=1)
    emerg_cost_day = EMERG_MULT * (EM * price).sum(axis=1)
    tot_plan = float(plan_cost_day[eval_mask].sum())
    tot_emerg = float(emerg_cost_day[eval_mask].sum())
    checks["评价期天数"] = int(eval_mask.sum())
    checks["计划购电费用(元)"] = tot_plan
    checks["紧急购电费用(元)"] = tot_emerg
    checks["总费用(元)"] = tot_plan + tot_emerg
    checks["紧急购电总量(kWh)"] = float(EM[eval_mask].sum())
    checks["紧急区间数"] = int((EM[eval_mask] > TOL).sum())
    ev_days = [d for d in range(ndays) if eval_mask[d] and EM[d].max() > TOL]
    checks["紧急事件日数"] = len(ev_days)
    checks["评价期购电总量(kWh)"] = float(B[eval_mask].sum())
    checks["评价期充电(kWh)"] = float(C[eval_mask].sum())
    checks["评价期放电(kWh)"] = float(D[eval_mask].sum())
    checks["评价期余量(kWh)"] = float(W[eval_mask].sum())
    checks["12-31日末SOC(kWh)"] = float(SOC1[-1])

    ok = (checks["执行能量平衡最大残差(kWh)"] < 1e-6
          and checks["SOC最小值(kWh)"] >= E_MIN - 1e-6
          and checks["SOC最大值(kWh)"] <= E_MAX + 1e-6
          and checks["跨日SOC断裂(kWh)"] < 1e-9
          and checks["充电越界"] == 0.0 and checks["放电越界"] == 0.0)

    print("\n=== 全年仿真校验 ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"  校验总评: {'PASS' if ok else 'FAIL'}")

    # ---------- 写 result2.xlsx ----------
    src = BASE / r"data\附件5\result2.xlsx"
    out = BASE / r"results\result2.xlsx"
    wb = openpyxl.load_workbook(src)

    ws = wb["计划购电量"]
    eval_dates = dates[eval_mask]
    assert ws.max_row - 1 == int(eval_mask.sum())
    for i, day in enumerate(np.where(eval_mask)[0]):
        for t in range(T):
            ws.cell(row=i + 2, column=t + 2, value=round(float(B[day, t]), 4))
        ws.cell(row=i + 2, column=146, value=round(float(B[day].sum()), 4))
        ws.cell(row=i + 2, column=147, value=round(float(plan_cost_day[day]), 4))

    ws2 = wb["充放电量"]
    ws2.delete_rows(2, ws2.max_row - 1)
    for day in np.where(eval_mask)[0]:
        for bblk in range(6):
            s, e_ = bblk * 24, (bblk + 1) * 24
            row = [dates.iloc[day] if bblk == 0 else None,
                   f"{bblk*4}:00-{(bblk+1)*4}:00",
                   round(float(C[day, s:e_].sum()), 4),
                   round(float(D[day, s:e_].sum()), 4),
                   None, None]
            if bblk == 0:
                row[4] = "0:00"
                row[5] = round(float(SOC0[day]), 4)
            elif bblk == 1:
                row[4] = "24:00"
                row[5] = round(float(SOC1[day]), 4)
            ws2.append(row)

    ws3 = wb["紧急购电量"]
    ws3.delete_rows(2, ws3.max_row - 1)
    for day in np.where(eval_mask)[0]:
        events = merge_events(EM[day])
        if not events:
            ws3.append([dates.iloc[day], None, None])
        else:
            for j, (s, e_, tot) in enumerate(events):
                ws3.append([dates.iloc[day] if j == 0 else None,
                            event_label(s, e_), round(tot, 4)])
    wb.save(out)
    print(f"已写出 {out}")

    # ---------- 指定日期（表3） ----------
    print("\n=== 指定日期紧急购电 ===")
    for ds in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
        day = int(np.where(dates == pd.Timestamp(ds))[0][0])
        evs = merge_events(EM[day])
        txt = "; ".join(f"{event_label(s,e_)}: {tot:.2f} kWh" for s, e_, tot in evs) or "无"
        print(f"  {ds}: {txt} | 计划购电 {B[day].sum():.2f} kWh, 计划费 {plan_cost_day[day]:.2f} 元")

    np.savez_compressed(BASE / r"results\q2_solution.npz",
                        B=B, C=C, D=D, EM=EM, W=W, SOC0=SOC0, SOC1=SOC1,
                        price=price, load_fc=load_fc, pv_fc=pv_fc,
                        load_act=load_act, pv_act=pv_act,
                        dates=dates.to_numpy(), eval_mask=eval_mask,
                        plan_cost_day=plan_cost_day, emerg_cost_day=emerg_cost_day)

    summary = {
        "question": "q2", "validation_pass": bool(ok), "checks": checks,
        "input_sha256": {"附件1.xlsx": sha256(BASE / r"data\附件1.xlsx"),
                          "附件2.xlsx": sha256(BASE / r"data\附件2.xlsx"),
                          "result2模板": sha256(BASE / r"data\附件5\result2.xlsx")},
        "output": str(out),
    }
    with open(BASE / r"results\q2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("已写出 results/q2_summary.json 与 results/q2_solution.npz")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
