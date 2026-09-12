# -*- coding: utf-8 -*-
"""
Q2 通用模型骨架（依题目原文构建，供后续迭代优化）
========================================================
本文件把"问题二"的模型按题面表述拆成 4 个可独立替换的模块，
每个模块的口径由 Policy 显式开关控制：

  ①  预测层 forecast      : 0:00 可用的净负载预测 N̂_{d,t}
  ②  计划层 plan_terminal : 日计划 LP 的边界/窗口处理
  ③  执行层 exec_*        : 日内储能的充/放动作自由度
  ④  结算 settle          : 计划电量按 p_t 计费、紧急电量按 5p_t 计费（题面给定，不设开关）

设计目的：让"改模型"= 只改 Policy 的某个字段，而不是改算法代码。
自检：`python q2_model.py` 会复现论文已核验的 A/B/C/D/PI/B48 数值并逐项比对。

（本文件只读输入 data/附件*.xlsx，不写入任何结果；结果对比目标为 results/q2_strategies.json）
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent

# ============================ 题面给定常数 ============================
T = 144                    # 每天 144 个 10 分钟区间
DT = 1.0 / 6.0             # 区间长度（h）
ETA_C = 0.9                # 充电效率（附录：充放电效率 90%）
ETA_D = 0.9                # 放电效率
E_MIN, E_MAX = 1200.0, 10800.0     # 电量运行区间（附录）
P_MAX = 5000.0 * DT        # 单区间最大充/放电量（5000 kW × 1/6 h）
E_INIT = 6000.0            # 2025-01-01 0:00 电量（附录）
MULT = 5.0                 # 紧急购电电价倍数（题面）
TOL = 1e-6                 # 事件判定阈值（kWh）
EVAL_START = pd.Timestamp("2025-02-01")   # result2.xlsx 要求的输出区间


# ============================ 数据层 ============================
def load_data():
    """读取附件1（代表日：电价/负载/光伏预测）与附件2（全年实测：负载/光伏）。

    时间口径（已定论）：附件1 时间列 144 行、首行 0:10、末行 '0:00+1'；附件2 数据列 144 列、
    首列 0:10、末列 '0:00+1'。故标签是区间【末端时刻】，区间 t 覆盖 (t·10min, (t+1)·10min]，
    t = 0..143 恰好铺满 (0:00, 24:00]。本函数按此口径把功率(kW)折算为区间电量(kWh)。
    """
    a1 = pd.read_excel(BASE / "data" / "附件1.xlsx", sheet_name=0, header=0)
    price = a1.iloc[:, 1].to_numpy(float)                      # 元/kWh
    net_ref = (a1.iloc[:, 2].to_numpy(float)
               - a1.iloc[:, 3].to_numpy(float)) * DT           # 代表日净负载 kWh

    xl = pd.ExcelFile(BASE / "data" / "附件2.xlsx")
    load_df = pd.read_excel(xl, sheet_name=0, header=0)
    pv_df = pd.read_excel(xl, sheet_name=1, header=0)
    dates = pd.to_datetime(load_df.iloc[:, 0]).to_numpy()
    load = load_df.iloc[:, 1:145].to_numpy(float) * DT          # (365,144) kWh
    pv = pv_df.iloc[:, 1:145].to_numpy(float) * DT
    net = load - pv

    assert price.shape[0] == T and net_ref.shape[0] == T
    assert load.shape[1] == T
    return dict(price=price, net_ref=net_ref, dates=dates, load=load, pv=pv, net=net)


# ============================ ① 预测层 ============================
def forecast_net(pol: "Policy", d: int, D: dict) -> np.ndarray:
    """第 d 天 0:00 可用的净负载预测（因果性由各分支自身保证）。"""
    if pol.forecast == "ref":                 # 只有代表日数据可用
        return D["net_ref"].copy()
    if pol.forecast == "oracle":              # 完美信息（不可实施，仅作参考）
        return D["net"][d].copy()

    hist = D["net"][max(0, d - pol.W):d]      # 仅 d 及以前之外：严格 d-1 及以前
    if hist.shape[0] < pol.min_hist:          # 冷启动
        return D["net_ref"].copy()

    base = np.median(hist, axis=0)            # 分时段中位数（= 代表日 + 残差中位数）
    if pol.forecast == "cmedian":
        return base
    if pol.forecast == "cquant":              # 上尾经验分位数预留
        margin = np.quantile(hist - base, pol.beta, axis=0)
        return base + np.maximum(margin, 0.0)
    raise ValueError(pol.forecast)


# ============================ ② 计划层 LP ============================
def solve_lp(price, N_hat, E_start, *, mode, terminal, e_end_min=None, b_fixed=None):
    """统一 LP。

    mode='plan'   : 变量 g,c,d,w,E；目标 min Σ p_t g_t；平衡 g + d = N̂ + c + w
    mode='recourse': 变量 c,d,e,w,E；目标 min Σ 5p_t e_t；平衡 b + d + e = N̂ + c + w
    terminal='cycle' : E_H = E_start
    terminal='geq'   : E_H ≥ e_end_min
    """
    H = len(N_hat)
    n = 4 * H + (H + 1)
    cobj = np.zeros(n)
    if mode == "plan":
        ig, ic, id_, iw, iE = 0, H, 2 * H, 3 * H, 4 * H
        cobj[ig:ig + H] = price
    else:
        ic, id_, ie, iw, iE = 0, H, 2 * H, 3 * H, 4 * H
        cobj[ie:ie + H] = MULT * price

    Aeq = lil_matrix((2 * H + 2, n))
    beq = np.zeros(2 * H + 2)
    for t in range(H):
        Aeq[t, iE + t + 1] = 1.0
        Aeq[t, iE + t] = -1.0
        Aeq[t, ic + t] = -ETA_C
        Aeq[t, id_ + t] = 1.0 / ETA_D
        r = H + t
        if mode == "plan":
            Aeq[r, ig + t] = 1.0
        else:
            Aeq[r, ie + t] = 1.0
        Aeq[r, id_ + t] = 1.0
        Aeq[r, ic + t] = -1.0
        Aeq[r, iw + t] = -1.0
        beq[r] = N_hat[t] - (0.0 if b_fixed is None else b_fixed[t])
    Aeq[2 * H, iE] = 1.0
    beq[2 * H] = E_start
    nterm = 2 * H
    if terminal == "cycle":
        nterm += 1
        Aeq[nterm, iE + H] = 1.0
        beq[nterm] = E_start
    Aeq = csr_matrix(Aeq[:nterm + 1])
    beq = beq[:nterm + 1]

    # 期末电量下限：plan 模式原为固定 (E_MIN,E_MAX)，此处改为支持可选下限 e_end_min。
    # 注意 e_end_min=None 时 lo=E_MIN，与改动前的界**完全相同**（向后兼容，不影响任何既有结果）。
    lo = E_MIN if e_end_min is None else max(E_MIN, e_end_min)
    if mode == "plan":
        bounds = ([(0.0, None)] * H + [(0.0, P_MAX)] * H + [(0.0, P_MAX)] * H
                  + [(0.0, None)] * H + [(E_MIN, E_MAX)] * H + [(lo, E_MAX)])
    else:
        bounds = ([(0.0, P_MAX)] * H + [(0.0, P_MAX)] * H + [(0.0, None)] * H
                  + [(0.0, None)] * H + [(E_MIN, E_MAX)] * H + [(lo, E_MAX)])

    res = linprog(cobj, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    assert res.status == 0, f"LP 失败: {res.message}"
    x = res.x
    if mode == "plan":
        return dict(b=x[ig:ig + H], c=x[ic:ic + H], d=x[id_:id_ + H],
                    w=x[iw:iw + H], e=np.zeros(H), E=x[iE:iE + H + 1], obj=float(res.fun))
    return dict(b=b_fixed.copy(), c=x[ic:ic + H], d=x[id_:id_ + H], w=x[iw:iw + H],
                e=x[ie:ie + H], E=x[iE:iE + H + 1], obj=float(res.fun))


def make_plan(pol: "Policy", price, N_hat, E_start):
    """按策略配置生成当日（或 H 天窗口的）计划 b 与充电意图。"""
    if pol.horizon == 1:
        sol = solve_lp(price, N_hat, E_start, mode="plan", terminal="cycle")
        return sol["b"], sol["c"]
    # 多日窗口：第二段预测与第一段相同（信息截止同一时点）
    Hp = np.concatenate([N_hat, N_hat])
    sol = solve_lp(np.tile(price, 2), Hp, E_start, mode="plan", terminal="cycle")
    return sol["b"][:T], sol["c"][:T]


# ============================ ③ 执行层 ============================
def execute(pol: "Policy", b, c_plan, N_hat, net_act_d, price, E_start):
    """区间内按实测物理平衡。两个开关：
      discharge_policy='must'    : 缺口必须由储能按物理上限补足（"有电即用"）
      discharge_policy='planned' : 储能动作是决策（可用重优化意图，允许主动少放电）
      charge_policy='planned'    : 余量按计划充电意图充
      charge_policy='max'        : 余量在物理上限内尽量充
    exec_mode='reopt' 时，每 reopt_step 个区间用同预测重优化剩余时域（b 固定）。
    """
    c = np.zeros(T); d = np.zeros(T); e = np.zeros(T); w = np.zeros(T)
    E = np.empty(T + 1); E[0] = E_start
    c_int = c_plan.copy()
    d_int = np.zeros(T)

    # 重优化与逐区间执行必须按块交错（E[t0] 是块开始的实际电量）
    step = pol.reopt_step if pol.exec_mode == "reopt" else T
    for t0 in range(0, T, step):
        if pol.exec_mode == "reopt":
            sol = solve_lp(price[t0:], N_hat[t0:], E[t0], mode="recourse",
                           terminal="geq", e_end_min=E_start, b_fixed=b[t0:])
            for k in range(min(step, T - t0)):
                c_int[t0 + k] = sol["c"][k]
                d_int[t0 + k] = sol["d"][k]
        for t in range(t0, min(t0 + step, T)):
            net = b[t] - net_act_d[t]          # >0 余量；<0 缺口
            if net >= 0:
                avail = max(0.0, (E_MAX - E[t]) / ETA_C)
                intent = c_int[t] if pol.charge_policy == "planned" else np.inf
                c[t] = min(intent, net, P_MAX, avail)
                w[t] = net - c[t]
            else:
                deficit = -net
                avail = max(0.0, (E[t] - E_MIN) * ETA_D)
                intent = d_int[t] if pol.discharge_policy == "planned" else np.inf
                d[t] = min(intent, deficit, P_MAX, avail)
                e[t] = deficit - d[t]
            E[t + 1] = E[t] + ETA_C * c[t] - d[t] / ETA_D
    return c, d, e, w, E


# ============================ ④ 结算（题面给定） ============================
def settle(b, e, price):
    """J = Σ p_t·b_t + Σ 5p_t·e_t。计划电量全量计费（弃置不退费），紧急量另行结算。"""
    return float((b * price).sum()), float(MULT * (e * price).sum())


# ============================ 策略配置 ============================
@dataclass
class Policy:
    name: str
    forecast: str = "ref"          # ref | cmedian | cquant | oracle
    W: int = 14
    min_hist: int = 7
    beta: float = 0.7
    horizon: int = 1
    exec_mode: str = "greedy"      # greedy | reopt
    reopt_step: int = 24           # 24 个 10 分钟区间 = 4 小时
    discharge_policy: str = "must"
    charge_policy: str = "planned"

    def describe(self):
        return (f"{self.name:10s} 预测={self.forecast:7s} W={self.W} β={self.beta} "
                f"窗口={self.horizon}天 执行={self.exec_mode}({self.reopt_step}) "
                f"放电={self.discharge_policy} 充电={self.charge_policy}")


# ============================ 全年回放 ============================
def replay(pol: Policy, D: dict, start_day: int = 0, E_start0: float | None = None):
    """全年回放。start_day/E_start0 用于"公共预热"口径：所有可实施策略从 2 月 1 日
    的公共实际状态起评（见公共预热等价性检查），预测仍只用 d-1 及以前的实测。"""
    price, dates, net = D["price"], D["dates"], D["net"]
    nd = len(dates)
    B = np.zeros((nd, T)); C = np.zeros((nd, T)); Dm = np.zeros((nd, T))
    EM = np.zeros((nd, T)); Wm = np.zeros((nd, T))
    SOC0 = np.zeros(nd); SOC1 = np.zeros(nd); Eall = np.zeros((nd, T + 1))
    E_start = E_INIT if E_start0 is None else float(E_start0)
    for d in range(start_day, nd):
        N_hat = forecast_net(pol, d, D)
        b, c_plan = make_plan(pol, price, N_hat, E_start)
        c, dd, ee, ww, E = execute(pol, b, c_plan, N_hat, net[d], price, E_start)
        B[d], C[d], Dm[d], EM[d], Wm[d] = b, c, dd, ee, ww
        SOC0[d], SOC1[d] = E[0], E[-1]
        Eall[d] = E
        E_start = E[-1]
    return dict(B=B, C=C, D=Dm, EM=EM, W=Wm, SOC0=SOC0, SOC1=SOC1, E=Eall)


def metrics(R, D: dict, idx):
    price = D["price"]
    plan = float((R["B"][idx] * price).sum())
    em = float(MULT * (R["EM"][idx] * price).sum())
    return dict(计划费用=plan, 紧急费用=em, 总费用=plan + em,
                紧急电量=float(R["EM"][idx].sum()),
                紧急日数=int((R["EM"][idx].sum(axis=1) > TOL).sum()),
                余量=float(R["W"][idx].sum()))


def audit(R, D: dict):
    r = R["B"] + R["D"] + R["EM"] - D["net"] - R["C"] - R["W"]
    return dict(平衡残差=float(np.max(np.abs(r))),
                重叠=float(np.minimum(R["C"], R["D"]).max()),
                跨日断裂=float(np.max(np.abs(R["SOC1"][:-1] - R["SOC0"][1:]))),
                SOC范围=[float(min(R["SOC0"].min(), R["SOC1"].min())),
                        float(max(R["SOC0"].max(), R["SOC1"].max()))])


def violations(R, D: dict, idx):
    """约束违反量（逐项最大越界幅度，kWh）。全部应为 0 或机器精度量级。"""
    E = R["E"][idx]
    neg = -min(0.0, float(min(R["B"][idx].min(), R["C"][idx].min(), R["D"][idx].min(),
                               R["EM"][idx].min(), R["W"][idx].min())))
    items = {
        "SOC下越": float(max(0.0, (E_MIN - E).max())),
        "SOC上越": float(max(0.0, (E - E_MAX).max())),
        "功率越界": float(max(0.0, (R["C"][idx] - P_MAX).max(), (R["D"][idx] - P_MAX).max())),
        "负值": float(neg),
        "充放电重叠": float(np.minimum(R["C"], R["D"])[idx].max()),
        "平衡残差": float(np.max(np.abs(R["B"] + R["D"] + R["EM"] - D["net"] - R["C"] - R["W"])[idx])),
    }
    items["约束违反量"] = max(items.values())
    return items


WARMUP_DAY = 31          # 2025-02-01 的日期索引（1 月 = 31 天）
E_FEB1 = 10800.0         # 公共 A 策略 1 月轨迹的实际末态


def mapping_table(D: dict):
    """O6：内部标准索引 ↔ 真实区间 ↔ 原始数据标签 ↔ 模板位置及标签。只报告事实，不代替决策。"""
    import openpyxl
    a1 = pd.read_excel(BASE / "data" / "附件1.xlsx", header=0)
    raw = [str(x) for x in a1.iloc[:, 0].tolist()]
    ws = openpyxl.load_workbook(BASE / "data" / "附件5" / "result2.xlsx")["计划购电量"]
    tlab = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)][1:145]

    def lab(m):
        return "0:%02d+1" % ((m - 1440) % 60) if m >= 1440 else "%d:%02d" % (m // 60, m % 60)

    print("=== O6 数据—模板映射核对（内部标准：t  ↔  真实区间 (tΔ,(t+1)Δ]）===")
    print(f"原始标签 {len(raw)} 个：首 {raw[0]}，末 {raw[-1]}（前若干为 datetime.time，后段为字符串，类型混杂）")
    print(f"模板时间列 {len(tlab)} 个：首 '{tlab[0]}'，末 '{tlab[-1]}'")
    print()
    print(f"{'内部t':>5} {'真实区间':>15} {'原始数据标签':>12} {'模板列号':>8} {'模板标签':>14}")
    for t in (0, 1, 59, 60, 61, 119, 142, 143):
        print(f"{t:>5} {'(%s,%s]' % (lab(t*10), lab(t*10+10)):>15} {raw[t]:>12}"
              f" {'第%d列' % (t+2):>8} {tlab[t]:>14}")
    print()
    print("事实 1：模板第 j 个时间列的标签 = [原始第 j 个标签, 原始第 j+1 个标签]，末列以 0:10+1 外推收尾。")
    print(f"        ⇒ 模板标签集 = (0:10,0:20], …, (23:50,24:00], (24:00,24:10]")
    print("事实 2：模板【没有】标签 0:00-0:10 的槽位，却【多出】标签 0:00-0:10+1 的槽位")
    print("        ⇒ 模板把源数据的『区间末端标签』当作『区间左端点』使用，标签相对真实区间整体右移一槽位。")
    print("事实 3：标签 10:00-10:10 只存在于第 61 列（= 列序对应 t=59 的位置）。")
    print()
    print("由此产生两种自洽填法（二者的后果不同，本脚本不代为决定）：")
    print("  (甲) 按列序 1:1 填：Excel 列 t+2 ← 内部 t。则第 61 列(标签10:00-10:10)装 t=59 的值 → 标签与内容不符。")
    print("  (乙) 按标签字面填：第 61 列装 t=60 的值 → 则整体错位一列，首区间 (0:00,0:10] 无槽位可放。")
    print("现状（仅记录）：论文表1 用 (乙) 读法（t=60）；results/result2.xlsx 用 (甲) 读法（第61列 ← t=59）。")


# ============================ 与已核验结果对拍 ============================
BASELINE = [
    Policy("A", forecast="ref"),
    Policy("B", forecast="cmedian"),
    Policy("B48", forecast="cmedian", horizon=2),
    Policy("C", forecast="cquant", beta=0.7),
    Policy("D", forecast="cquant", beta=0.7, exec_mode="reopt",
           discharge_policy="planned"),
    Policy("PI", forecast="oracle"),
]


# ============================ 执行口径四组合（固定每 4 小时重优化） ============================
def combos():
    """O1（缺口侧）× O2（余量侧）四组合。其余一切固定：预测 cquant β=0.7、W=14、
    exec_mode='reopt'、reopt_step=24、公共预热起点。
    额外加入 R2*：与 R2 同规则，但**完全关闭执行层 LP**（greedy），用于验证
    "must+max 不使用重优化意图时执行 LP 是否退化为冗余"。"""
    common = dict(forecast="cquant", beta=0.7, W=14, exec_mode="reopt", reopt_step=24)
    return {
        "C  基准":  Policy("C  基准",  forecast="cquant", beta=0.7, W=14,
                          exec_mode="greedy", discharge_policy="must", charge_policy="planned"),
        "R1 must/planned": Policy("R1", **common, discharge_policy="must",    charge_policy="planned"),
        "R2 must/max":     Policy("R2", **common, discharge_policy="must",    charge_policy="max"),
        "R3 planned/planned": Policy("R3", **common, discharge_policy="planned", charge_policy="planned"),
        "R4 planned/max":  Policy("R4", **common, discharge_policy="planned", charge_policy="max"),
        "R2* 无执行LP":     Policy("R2*", forecast="cquant", beta=0.7, W=14,
                                 exec_mode="greedy", discharge_policy="must", charge_policy="max"),
    }


def verify_no_reopt(R2reopt, R2nore, idx):
    """验证：must+max 规则下，执行层重优化产生的 c*/d* 均未被使用，故 R2 应逐元素等于 R2*。"""
    print("\n=== 3b. R2 中执行层 LP 是否冗余（must+max 不使用 c*/d*）===")
    dmax = {}
    for k in ("B", "C", "D", "EM", "W"):
        dmax[k] = float(np.abs(R2reopt[k][idx] - R2nore[k][idx]).max())
    dmax["SOC0"] = float(np.abs(R2reopt["SOC0"][idx] - R2nore["SOC0"][idx]).max())
    dmax["SOC1"] = float(np.abs(R2reopt["SOC1"][idx] - R2nore["SOC1"][idx]).max())
    print("  逐元素最大差（R2 含执行LP  vs  R2* 无执行LP）：")
    print("    " + " | ".join(f"{k} {v:.2e}" for k, v in dmax.items()))
    ok = max(dmax.values()) < 1e-9
    print(f"  → {'完全一致 ✔：主方案无需执行层 LP，可简化为「分位数日前计划 + 尽限实时反馈」' if ok else '**不一致，需排查**'}")
    return ok


def verify_o2_proposition(D, pol, idx):
    """命题（固定合同下的执行占优）：同一初始电量、同一购电序列 b、同一实测净负载，
    且缺口侧均用 must 时，余量侧 max 相对 planned 有 E_t^max ≥ E_t^planned、e_t^max ≤ e_t^planned。
    证明见 Q2模型.md §9；此处按日逐区间数值核验（b 与 E_start 对两种规则完全相同）。"""
    print("\n=== 5. 固定合同下 O2 命题的逐区间核验（同一 b、同一 E_start、缺口侧均为 must）===")
    price, net = D["price"], D["net"]
    Epol = Policy("E", forecast=pol.forecast, beta=pol.beta, W=pol.W,
                  exec_mode="greedy", discharge_policy="must", charge_policy="planned")
    Emax = Policy("M", forecast=pol.forecast, beta=pol.beta, W=pol.W,
                  exec_mode="greedy", discharge_policy="must", charge_policy="max")
    Rref = replay(pol, D, start_day=WARMUP_DAY, E_start0=E_FEB1)
    bad_E = bad_e = 0
    worst_E = worst_e = 0.0
    for d in idx:
        N_hat = forecast_net(pol, d, D)
        E0 = float(Rref["SOC0"][d])
        b, c_plan = make_plan(pol, price, N_hat, E0)
        _, _, em_p, _, EE_p = execute(Epol, b, c_plan, N_hat, net[d], price, E0)
        _, _, em_m, _, EE_m = execute(Emax, b, c_plan, N_hat, net[d], price, E0)
        vE = float((EE_p[1:] - EE_m[1:]).max())      # max(planned - max) 应为 ≤0
        ve = float((em_m - em_p).max())              # max(max - planned) 应为 ≤0
        if vE > 1e-9:
            bad_E += 1; worst_E = max(worst_E, vE)
        if ve > 1e-9:
            bad_e += 1; worst_e = max(worst_e, ve)
    print(f"  违反 E_t^max ≥ E_t^planned 的天数 = {bad_E}/{len(idx)}，最大违反 {worst_E:.3e} kWh")
    print(f"  违反 e_t^max ≤ e_t^planned 的天数 = {bad_e}/{len(idx)}，最大违反 {worst_e:.3e} kWh")
    print(f"  → {'逐区间命题成立 ✔（固定合同下的数学命题，不等于全年费用更低）' if bad_E == 0 and bad_e == 0 else '**命题被反例推翻**'}")


def diagnose_mechanism(D, R1, R2, idx):
    """机制诊断：O2 增加当前可储存的能量，O1 改变已有能量的释放时机。
    检验"更充分地吸收早先余量，是否减少了后来主动留电的必要性"。"""
    price = D["price"]
    p90 = float(np.quantile(price, 0.90))
    hi = price >= p90
    print(f"\n=== 6. R1 与 R2 的机制诊断（高价阈值 p ≥ 90 分位 = {p90:.4f} 元/kWh，共 {int(hi.sum())} 个区间/天）===")
    e1 = R1["EM"][idx][:, hi].sum()
    e2 = R2["EM"][idx][:, hi].sum()
    print(f"  高价区间内的紧急电量：R1 {e1:.1f} kWh  →  R2 {e2:.1f} kWh（变化 {e2-e1:+.1f}）")
    E1 = R1["E"][idx][:, :T]
    E2 = R2["E"][idx][:, :T]
    diff = E2 - E1
    print(f"  高价区间开始时刻的储能差 (R2 − R1)：均值 {diff[:, hi].mean():+.2f} kWh，"
          f"中位数 {np.median(diff[:, hi]):+.2f} kWh，为正的比例 {(diff[:, hi] > 0).mean()*100:.1f}%")
    print(f"  全时域储能差 (R2 − R1)：均值 {diff.mean():+.2f} kWh，"
          f"为正的比例 {(diff > 0).mean()*100:.1f}%")
    # 只在高价且实际出现缺口的区间上看
    deficit = (D["net"][idx][:, :T] - R2["B"][idx]) > 0
    sel = hi[None, :] & deficit
    if sel.sum() > 0:
        print(f"  高价且出现缺口的区间（{int(sel.sum())} 个）上，储能差均值 {diff[sel].mean():+.2f} kWh，"
              f"紧急电量 R1 {R1['EM'][idx][:, :T][sel].sum():.1f} → R2 {R2['EM'][idx][:, :T][sel].sum():.1f} kWh")
    print("  说明：R2 在多数区间储能不低于 R1，故进入高价缺口时可用电量更多、需要的紧急购电更少。")


def main():
    D = load_data()
    idx = np.where(D["dates"] >= EVAL_START.to_datetime64())[0]
    ref = json.loads((BASE / "results" / "q2_strategies.json").read_text(encoding="utf-8"))["table"]

    mapping_table(D)

    # ---------- 1. 基线对拍：证明骨架忠实于已核验结果 ----------
    print(f"\n=== 1. 基线对拍（评价期 {len(idx)} 天，各策略按原始口径从 1/1 起跑）===")
    worst = 0.0
    for pol in BASELINE:
        R = replay(pol, D)
        m = metrics(R, D, idx)
        r0 = ref[pol.name]
        au = audit(R, D)
        worst = max(worst, abs(m["总费用"] / 1e4 - r0["总费用(万元)"]))
        print(f"  {pol.name:5s} 总 {m['总费用']/1e4:9.4f} 万 (论文 {r0['总费用(万元)']:9.4f}) | "
              f"计划 {m['计划费用']/1e4:9.4f} | 紧急 {m['紧急费用']/1e4:8.4f} | "
              f"紧急电量 {m['紧急电量']:11.1f} | 紧急日 {m['紧急日数']:3d} | 平衡残差 {au['平衡残差']:.1e}")
    print(f"  → 最大总费用偏差 {worst:.3e} 万元 "
          f"{'（一致 ✔）' if worst < 1e-3 else '（**需排查**）'}")

    # ---------- 2. 公共预热等价性：A 的 1 月轨迹 vs 直接 2/1 从 10800 起评 ----------
    print("\n=== 2. 公共预热等价性检查（口径 O7）===")
    print(f"  口径：所有可实施策略共用 A 的 1 月实际轨迹（6000 → {E_FEB1:.0f} kWh），"
          f"2 月 1 日起从 {E_FEB1:.0f} kWh 起评。")
    for name in ("A", "B", "B48", "C", "D"):
        pol = next(p for p in BASELINE if p.name == name)
        Rf = replay(pol, D)                                   # 从 1/1 起（含 1 月自跑）
        Rp = replay(pol, D, start_day=WARMUP_DAY, E_start0=E_FEB1)
        dmax = max(np.abs(Rf[k][idx] - Rp[k][idx]).max() for k in ("B", "C", "D", "EM", "W"))
        print(f"  {name:5s} 2–12 月逐元素最大差 {dmax:.2e} | 1 月自跑末态 {Rf['SOC1'][WARMUP_DAY-1]:.4f} kWh"
              f" | {'等价 ✔' if dmax < 1e-9 else '**不等价**'}")

    # ---------- 3. O1 × O2 四组合 ----------
    print("\n=== 3. O1×O2 四组合（固定每 4 小时重优化；公共预热起评）===")
    print("  固定：预测 cquant β=0.70、W=14、reopt_step=24、2/1 起评 E=10800 kWh、无期末约束")
    res, met = {}, {}
    for tag, pol in combos().items():
        R = replay(pol, D, start_day=WARMUP_DAY, E_start0=E_FEB1)
        res[tag] = R
        met[tag] = metrics(R, D, idx)
        met[tag]["_vio"] = violations(R, D, idx)
    print()
    print(f"{'组合':<20}{'计划费用':>10}{'紧急费用':>10}{'总费用':>10}{'紧急电量':>12}"
          f"{'供能余量':>12}{'年末电量':>12}{'约束违反量':>13}")
    print(f"{'':<20}{'/万元':>10}{'/万元':>10}{'/万元':>10}{'/kWh':>12}{'/kWh':>12}{'/kWh':>12}{'/kWh':>13}")
    for tag, m in met.items():
        R = res[tag]
        print(f"{tag:<20}{m['计划费用']/1e4:>10.4f}{m['紧急费用']/1e4:>10.4f}{m['总费用']/1e4:>10.4f}"
              f"{m['紧急电量']:>12.1f}{m['余量']:>12.1f}{R['SOC1'][-1]:>12.4f}"
              f"{m['_vio']['约束违反量']:>13.2e}")
    print()
    print("  约束违反量明细（kWh）：")
    for tag, m in met.items():
        v = m["_vio"]
        print(f"    {tag:<20} SOC下越 {v['SOC下越']:.2e} | SOC上越 {v['SOC上越']:.2e} | "
              f"功率越界 {v['功率越界']:.2e} | 负值 {v['负值']:.2e} | "
              f"重叠 {v['充放电重叠']:.2e} | 平衡残差 {v['平衡残差']:.2e}")

    # ---------- 3b. R2 的执行层 LP 是否冗余 ----------
    verify_no_reopt(res["R2 must/max"], res["R2* 无执行LP"], idx)

    # ---------- 4. 两开关的交互作用 ----------
    J = {k: v["总费用"] / 1e4 for k, v in met.items()}
    o1_under_planned = J["R1 must/planned"] - J["R3 planned/planned"]
    o1_under_max = J["R2 must/max"] - J["R4 planned/max"]
    o2_under_must = J["R1 must/planned"] - J["R2 must/max"]
    o2_under_planned = J["R3 planned/planned"] - J["R4 planned/max"]
    I = o1_under_planned - o1_under_max
    print("\n=== 4. 两开关的交互作用（万元；正 = 前者更贵）===")
    print(f"  O1 增益（余量侧=planned）：J(R1) − J(R3) = {o1_under_planned:+.4f}")
    print(f"  O1 增益（余量侧=max）    ：J(R2) − J(R4) = {o1_under_max:+.4f}")
    print(f"  O2 增益（缺口侧=must）   ：J(R1) − J(R2) = {o2_under_must:+.4f}")
    print(f"  O2 增益（缺口侧=planned）：J(R3) − J(R4) = {o2_under_planned:+.4f}")
    print(f"  交互作用 I = (J_R1−J_R3) − (J_R2−J_R4) = {I:+.4f}")
    print(f"  → 余量充电规则由 planned 改为 max 后，『允许保留电量』的收益"
          f"{'基本不变' if abs(I) < 1.0 else '发生明显变化'}"
          f"（变化量 {abs(I):.4f} 万元，占 O1 增益的 {abs(I)/abs(o1_under_planned)*100:.2f}%）")
    print(f"  基准 C（greedy, must/planned）总费用 = {J['C  基准']:.4f} 万元；"
          f"四组合相对 C 的变化区间 = [{J['R4 planned/max']-J['C  基准']:+.4f}, "
          f"{max(J['R1 must/planned'],J['R2 must/max'])-J['C  基准']:+.4f}] 万元")
    print("  说明：I 只表示『余量充分入储后，主动保留放电能力的边际降费收益缩小了多少』，"
          "不表示两开关物理作用重叠。")

    # ---------- 5. 固定合同下 O2 命题的逐区间核验 ----------
    verify_o2_proposition(D, combos()["R2 must/max"], idx)

    # ---------- 6. R1 与 R2 的机制诊断 ----------
    diagnose_mechanism(D, res["R1 must/planned"], res["R2 must/max"], idx)

    print("\n备注：PI 未参与本轮比较（不可实施，且 2 月初为 6000 kWh 与其余策略不同），"
          "本轮不将任何量称为下界。β 与 W 均未调整。")

    # ---------- 结果落盘（便于复现） ----------
    out = {
        "固定口径": {"预测": "cquant", "beta": 0.7, "W": 14, "exec_mode": "reopt",
                     "reopt_step": 24, "起点": "2025-02-01, E=10800 kWh(公共A预热)",
                     "期末约束": "无", "评价期天数": int(len(idx))},
        "组合": {tag: {**{k: float(v) for k, v in m.items() if not k.startswith("_")},
                       **{"违反_" + k: float(v) for k, v in m["_vio"].items()},
                       "年末电量": float(res[tag]["SOC1"][-1])}
                 for tag, m in met.items()},
        "效应": {"执行模式_greedy→reopt(均must/planned)": float(J["C  基准"] - J["R1 must/planned"]),
                 "O1_增益(余量侧=planned)": float(o1_under_planned),
                 "O1_增益(余量侧=max)": float(o1_under_max),
                 "O2_增益(缺口侧=must)": float(o2_under_must),
                 "O2_增益(缺口侧=planned)": float(o2_under_planned),
                 "交互作用I": float(I)},
    }
    (BASE / "results" / "q2_combos.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写出 results/q2_combos.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
