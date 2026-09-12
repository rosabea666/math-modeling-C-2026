# -*- coding: utf-8 -*-
"""
Q2 递进策略对比（唯一复现入口）
  A   现有基线：附件1代表日预测 + 日循环LP + 贪心执行（见 q2_solve.py，此处复算对照）
  B   因果预测校准：净负载预测 = 代表日净负载 + 过去W日分时段残差中位数
  B48 B + 48h跨日计划窗口（次日预测同一历史窗口，末端2日循环边界处理）
  C   B + 风险预留：净负载再加 分时段β分位数（残差上尾），β在1月预热期按总费用选定、2月冻结
  D   C + 执行层每4小时重优化（计划购电b固定，未来区间用同一预测，允许低价段主动小额紧急购电）
  PI  完美信息日循环参考方案：每日以实测净负载做计划（同物理/结算规则，仅放松信息约束；
      保留日循环约束，与A-D约束集不同，非严格下界）

因果性：任何一天 d 的预测只用 d-1 及以前的实测；β 只用预热期（1月）数据，窗口W=14天为固定常数；
PI 仅作参考对照，不是可实施策略，也不是严格下界。
"""
import sys
import io
import json
import hashlib
from pathlib import Path

if __name__ == "__main__":
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
P_MAX = 5000.0 * DT
E_INIT = 6000.0
EMERG_MULT = 5.0
TOL = 1e-6
W_HIST = 14          # 历史窗口（天）
MIN_HIST = 7         # 冷启动：少于此历史不校准
BETA_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
WARMUP_END = 31      # 1月 = 前31天


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs():
    a1 = pd.read_excel(BASE / r"data\附件1.xlsx", sheet_name=0, header=0)
    price = a1.iloc[:, 1].to_numpy(float)
    net_fc = (a1.iloc[:, 2].to_numpy(float) - a1.iloc[:, 3].to_numpy(float)) * DT  # 代表日净负载
    xl = pd.ExcelFile(BASE / r"data\附件2.xlsx")
    load_df = pd.read_excel(xl, sheet_name=0, header=0)
    pv_df = pd.read_excel(xl, sheet_name=1, header=0)
    dates = pd.to_datetime(load_df.iloc[:, 0])
    load_act = load_df.iloc[:, 1:145].to_numpy(float) * DT
    pv_act = pv_df.iloc[:, 1:145].to_numpy(float) * DT
    net_act = load_act - pv_act
    return price, net_fc, dates, load_act, pv_act, net_act


# ---------------- 通用 LP ----------------
def lp_dispatch(price_h, N_hat, E_start, e_end_min=None, e_end_eq=None,
                b_fixed=None, emerg_mult=None):
    """
    单/多区间调度LP。
    price_h: (H,) 电价; N_hat: (H,) 净负载预测(kWh); E_start: 初始SOC
    e_end_eq: 末端SOC等式约束; e_end_min: 末端SOC下界
    b_fixed: (H,) 固定购电（执行层重优化用），此时购电不再是变量，紧急购电e为变量
    emerg_mult: 紧急购电价格倍数（b_fixed模式必给）
    返回 dict(g=, c=, d=, w=, e=, E=, cost=)  (b_fixed模式 cost=紧急费用)
    """
    H = len(N_hat)
    if b_fixed is None:
        # 变量: g,c,d,w,E[H+1]
        n = 4 * H + (H + 1)
        ig, ic, id_, iw, iE = 0, H, 2*H, 3*H, 4*H
        cobj = np.zeros(n); cobj[ig:ig+H] = price_h
        neq = 2*H
    else:
        # 变量: c,d,e,w,E[H+1]
        n = 4 * H + (H + 1)
        ic, id_, ie, iw, iE = 0, H, 2*H, 3*H, 4*H
        cobj = np.zeros(n); cobj[ie:ie+H] = emerg_mult * price_h
        neq = 2*H
    Aeq = lil_matrix((neq + 2, n))
    beq = np.zeros(neq + 2)
    for t in range(H):
        Aeq[t, iE+t+1] = 1.0; Aeq[t, iE+t] = -1.0
        Aeq[t, ic+t] = -ETA_C; Aeq[t, id_+t] = 1.0/ETA_D
        r = H + t
        if b_fixed is None:
            Aeq[r, ig+t] = 1.0
        else:
            Aeq[r, ie+t] = 1.0
        Aeq[r, id_+t] = 1.0; Aeq[r, ic+t] = -1.0; Aeq[r, iw+t] = -1.0
        beq[r] = N_hat[t] - (0.0 if b_fixed is None else b_fixed[t])
    Aeq[neq, iE] = 1.0; beq[neq] = E_start
    nterm = neq
    if e_end_eq is not None:
        nterm += 1
        Aeq[nterm, iE+H] = 1.0; beq[nterm] = e_end_eq
    Aeq = csr_matrix(Aeq[:nterm+1]); beq = beq[:nterm+1]

    if b_fixed is None:
        bounds = ([(0.0, None)]*H + [(0.0, P_MAX)]*H + [(0.0, P_MAX)]*H
                  + [(0.0, None)]*H + [(E_MIN, E_MAX)]*(H+1))
    else:
        e_max_end = E_MAX if e_end_min is None else None
        E_bounds = [(E_MIN, E_MAX)]*H + [(max(E_MIN, e_end_min or E_MIN), E_MAX)]
        bounds = ([(0.0, P_MAX)]*H + [(0.0, P_MAX)]*H + [(0.0, None)]*H
                  + [(0.0, None)]*H + E_bounds)
    res = linprog(cobj, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    assert res.status == 0, f"LP失败: {res.message}"
    x = res.x
    if b_fixed is None:
        return {"g": x[ig:ig+H], "c": x[ic:ic+H], "d": x[id_:id_+H],
                "w": x[iw:iw+H], "e": np.zeros(H), "E": x[iE:iE+H+1], "cost": float(res.fun)}
    return {"g": b_fixed.copy(), "c": x[ic:ic+H], "d": x[id_:id_+H],
            "w": x[iw:iw+H], "e": x[ie:ie+H], "E": x[iE:iE+H+1], "cost": float(res.fun)}


# ---------------- 预测（因果） ----------------
def forecast_net(day, net_fc, net_act, beta=None):
    """day d 的净负载预测。只用 [d-W_HIST, d-1] 的实测残差。
    beta: 在B校准基础上再加 分时段beta分位数 的风险预留（C策略）。"""
    h0 = max(0, day - W_HIST)
    hist = net_act[h0:day]                    # (m,144)
    if hist.shape[0] < MIN_HIST:
        base = net_fc.copy()
        resid_hist = None
    else:
        resid = hist - net_fc                 # 相对代表日的残差
        base = net_fc + np.median(resid, axis=0)
        resid_hist = hist - base              # 相对校准预测的残差
    if beta is not None and resid_hist is not None:
        margin = np.quantile(resid_hist, beta, axis=0)
        margin = np.maximum(margin, 0.0)      # 只加上尾预留
        base = base + margin
    return base


# ---------------- 执行器 ----------------
def execute_greedy(b, c_plan, net_act_d, E_start):
    """A/B/C 用：贪心因果执行。net_act_d: 当天实测净负载"""
    c = np.zeros(T); d = np.zeros(T); e = np.zeros(T); w = np.zeros(T)
    E = np.empty(T+1); E[0] = E_start
    for t in range(T):
        net = b[t] - net_act_d[t]
        if net >= 0:
            c_t = min(c_plan[t], net, P_MAX, max(0.0, (E_MAX - E[t]) / ETA_C))
            c[t] = c_t; w[t] = net - c_t
        else:
            d_t = min(-net, P_MAX, max(0.0, (E[t] - E_MIN) * ETA_D))
            d[t] = d_t; e[t] = -net - d_t
        E[t+1] = E[t] + ETA_C*c[t] - d[t]/ETA_D
    return c, d, e, w, E


def execute_reopt(b, N_hat, net_act_d, price, E_start):
    """D 用：每4小时（24个10分钟区间）以b固定重优化剩余时域，执行时按实测物理截断。
    重优化的预测轨迹末端SOC不低于日初值；实际执行经截断后末端可低于日初值，
    实际储能跨日衔接（评价期有137天实际末端低于日初，属正常反馈，不违反题意）。
    允许执行层在低价时段主动保留电量（重优化自行决定e）。"""
    c = np.zeros(T); d = np.zeros(T); e = np.zeros(T); w = np.zeros(T)
    E = np.empty(T+1); E[0] = E_start
    plan_cd = {}
    for t0 in range(0, T, 24):
        sol = lp_dispatch(price[t0:], N_hat[t0:], E[t0], b_fixed=b[t0:],
                          emerg_mult=EMERG_MULT, e_end_min=E_START_DAY[0])
        for k in range(24):
            plan_cd[t0+k] = (sol["c"][k], sol["d"][k])
        for t in range(t0, t0+24):
            c_star, d_star = plan_cd[t]
            net = b[t] - net_act_d[t]
            if net >= 0:
                c_t = min(c_star, net, P_MAX, max(0.0, (E_MAX - E[t]) / ETA_C))
                c[t] = c_t; w[t] = net - c_t
            else:
                deficit = -net
                # 遵循重优化的放电意图（可为保留电量而小于缺口），但做物理截断
                d_t = min(d_star, deficit, P_MAX, max(0.0, (E[t] - E_MIN) * ETA_D))
                d[t] = d_t; e[t] = deficit - d_t
            E[t+1] = E[t] + ETA_C*c[t] - d[t]/ETA_D
    return c, d, e, w, E


E_START_DAY = [E_INIT]  # D 执行末端下界（逐日更新）


# ---------------- 全年回放 ----------------
def replay(price, dates, net_fc, net_act, load_act, pv_act, strategy, beta=None):
    ndays = len(dates)
    B = np.zeros((ndays, T)); C = np.zeros((ndays, T)); Dm = np.zeros((ndays, T))
    EM = np.zeros((ndays, T)); Wm = np.zeros((ndays, T))
    SOC0 = np.zeros(ndays); SOC1 = np.zeros(ndays)
    E_start = E_INIT
    for day in range(ndays):
        if strategy == "A":
            N_hat = net_fc
        elif strategy in ("B", "B48", "C", "D"):
            N_hat = forecast_net(day, net_fc, net_act, beta if strategy in ("C", "D") else None)
        elif strategy == "PI":
            N_hat = net_act[day]
        if strategy == "B48":
            # 48h窗口：信息截止日 = 当天0时。次日预测与当天相同（两天均只用 day-1 及以前数据），
            # 避免 forecast_net(day+1) 把当天未发生的实测混入历史（评审指出的泄漏）。
            N_next = N_hat
            sol = lp_dispatch(np.tile(price, 2), np.concatenate([N_hat, N_next]),
                              E_start, e_end_eq=E_start)
            b = sol["g"][:T]; c_plan = sol["c"][:T]
        else:
            sol = lp_dispatch(price, N_hat, E_start, e_end_eq=E_start)
            b = sol["g"]; c_plan = sol["c"]
        E_START_DAY[0] = E_start
        if strategy == "D":
            c, d, e, w, E = execute_reopt(b, N_hat, net_act[day], price, E_start)
        else:
            c, d, e, w, E = execute_greedy(b, c_plan, net_act[day], E_start)
        B[day], C[day], Dm[day], EM[day], Wm[day] = b, c, d, e, w
        SOC0[day] = E[0]; SOC1[day] = E[-1]
        E_start = E[-1]
    return {"B": B, "C": C, "D": Dm, "EM": EM, "W": Wm, "SOC0": SOC0, "SOC1": SOC1}


def metrics(R, price, eval_idx):
    plan_cost = float((R["B"][eval_idx] * price).sum())
    em_cost = float(EMERG_MULT * (R["EM"][eval_idx] * price).sum())
    return {"计划费用(万元)": plan_cost/1e4, "紧急费用(万元)": em_cost/1e4,
            "总费用(万元)": (plan_cost+em_cost)/1e4,
            "紧急电量(kWh)": float(R["EM"][eval_idx].sum()),
            "紧急日数": int((R["EM"][eval_idx].sum(axis=1) > TOL).sum()),
            "余量(kWh)": float(R["W"][eval_idx].sum())}


def merge_events(e):
    events = []; t = 0
    while t < T:
        if e[t] > TOL:
            s = t; tot = 0.0
            while t < T and e[t] > TOL:
                tot += e[t]; t += 1
            events.append((s, t, tot))
        else:
            t += 1
    return events


def event_label(s, e_):
    def fmt(i):
        m = i * 10
        return f"{(m-1440)//60}:{(m-1440)%60:02d}+1" if m >= 1440 else f"{m//60}:{m%60:02d}"
    return f"{fmt(s)}-{fmt(e_)}"


def main():
    price, net_fc, dates, load_act, pv_act, net_act = load_inputs()
    ndays = len(dates)
    eval_idx = np.where(dates >= pd.Timestamp("2025-02-01"))[0]

    # ---- 0. 执行器自洽性检查：实测=预测时应复现LP计划、零紧急 ----
    sol0 = lp_dispatch(price, net_fc, E_INIT, e_end_eq=E_INIT)
    c0, d0, e0, w0, E0 = execute_greedy(sol0["g"], sol0["c"], net_fc, E_INIT)
    rep = {"紧急总量": float(e0.sum()),
           "充电偏差": float(np.max(np.abs(c0 - sol0["c"]))),
           "放电偏差": float(np.max(np.abs(d0 - sol0["d"]))),
           "末端SOC偏差": float(abs(E0[-1] - E_INIT))}
    print("执行器自洽检查:", rep)
    assert rep["紧急总量"] < TOL and rep["充电偏差"] < 1e-6 and rep["放电偏差"] < 1e-6

    results = {}
    # ---- A ----
    print("回放 A(基线) ...")
    results["A"] = replay(price, dates, net_fc, net_act, load_act, pv_act, "A")
    # ---- B / B48 ----
    print("回放 B(因果校准) ...")
    results["B"] = replay(price, dates, net_fc, net_act, load_act, pv_act, "B")
    print("回放 B48(48h窗口) ...")
    results["B48"] = replay(price, dates, net_fc, net_act, load_act, pv_act, "B48")
    # ---- C：β 在预热期（1月）选定 ----
    print("预热期选择β ...")
    beta_scores = {}
    warm_idx = np.arange(0, WARMUP_END)
    for beta in BETA_GRID:
        R = replay(price, dates[:WARMUP_END], net_fc, net_act[:WARMUP_END],
                   load_act[:WARMUP_END], pv_act[:WARMUP_END], "C", beta=beta)
        m = metrics(R, price, np.arange(0, WARMUP_END))
        beta_scores[beta] = m["总费用(万元)"]
        print(f"  β={beta}: 1月总费用 {m['总费用(万元)']:.2f} 万元 (紧急 {m['紧急费用(万元)']:.2f})")
    best_beta = min(beta_scores, key=beta_scores.get)
    print(f"  选定 β* = {best_beta}（2月起冻结）")
    print("回放 C(风险预留) ...")
    results["C"] = replay(price, dates, net_fc, net_act, load_act, pv_act, "C", beta=best_beta)
    # ---- D ----
    print("回放 D(执行重优化) ...")
    results["D"] = replay(price, dates, net_fc, net_act, load_act, pv_act, "D", beta=best_beta)
    # ---- PI ----
    print("回放 PI(完美信息下界) ...")
    results["PI"] = replay(price, dates, net_fc, net_act, load_act, pv_act, "PI")

    # ---- 对比表 ----
    print("\n=== 策略对比（评价期 2.1-12.31, 334天） ===")
    table = {}
    for name in ("A", "B", "B48", "C", "D", "PI"):
        table[name] = metrics(results[name], price, eval_idx)
        m = table[name]
        print(f"  {name:4s} 总 {m['总费用(万元)']:9.2f} 万 | 计划 {m['计划费用(万元)']:9.2f} | "
              f"紧急 {m['紧急费用(万元)']:8.2f} | 紧急电量 {m['紧急电量(kWh)']:12.1f} | "
              f"紧急日 {m['紧急日数']:3d} | 余量 {m['余量(kWh)']:12.1f}")

    # ---- 校验（净负载口径: b + d + e = N + c + w） ----
    checks_detail = {}
    for name in ("A", "B", "B48", "C", "D", "PI"):
        R = results[name]
        r = R["B"] + R["D"] + R["EM"] - net_act - R["C"] - R["W"]
        checks_detail[name] = {
            "能量平衡最大残差": float(np.max(np.abs(r))),
            "SOC最小": float(min(R["SOC0"].min(), R["SOC1"].min())),
            "SOC最大": float(max(R["SOC0"].max(), R["SOC1"].max())),
            "跨日断裂": float(np.max(np.abs(R["SOC1"][:-1] - R["SOC0"][1:]))),
        }
        print(f"  校验 {name}: {checks_detail[name]}")

    # ---- 指定日期（C策略） ----
    print("\n=== 指定日期紧急购电（C策略） ===")
    for ds in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
        day = int(np.where(dates == pd.Timestamp(ds))[0][0])
        evs = merge_events(results["C"]["EM"][day])
        txt = "; ".join(f"{event_label(s,e_)}: {tot:.2f} kWh" for s, e_, tot in evs) or "无"
        print(f"  {ds}: {txt}")

    # ---- 保存 ----
    save = {"beta_star": best_beta, "beta_scores": beta_scores,
            "executor_selfcheck": rep, "table": table, "checks": checks_detail}
    for name in ("A", "B", "B48", "C", "D", "PI"):
        R = results[name]
        np.savez_compressed(BASE / rf"results\q2_strategy_{name}.npz",
                            B=R["B"], C=R["C"], D=R["D"], EM=R["EM"], W=R["W"],
                            SOC0=R["SOC0"], SOC1=R["SOC1"])
    with open(BASE / r"results\q2_strategies.json", "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print("已写出 results/q2_strategies.json 与 q2_strategy_*.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
