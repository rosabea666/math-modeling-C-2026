# -*- coding: utf-8 -*-
r"""
§25 认证随机 LP 采购（乐观界）：分离 §24 的"求解质量"混杂（只读，不改模型）
================================================================================
§24 的非前瞻采购用的是**分块坐标下降**——用户指出这是三个未分离混杂之一（**求解质量**：
只搜索了有限候选、无最优性差距证书）。本脚本把求解换成 **HiGHS 认证的两阶段随机 LP**：

  min_{b, {x_s}, θ}   Σ_t p_t b_t  +  Σ_s w_s [ 5 Σ_t p_t e_{s,t} + θ_s ]
  s.t.  每场景 s：E 动态、区间平衡 b_t + d + e = net_s + c + w、E_{s,0}=E_start
        θ_s ≥ Φ(E_{s,T})（Φ 为凸分段线性，用割平面/epigraph 表示）

  · **共享日前购电 b**（每天只在 0:00 定一次）——这是唯一的第一阶段决策；
  · **每场景储能自由调度（anticipative）** ⇒ 这是**信息放松**（场景内完美日内信息），
    故 b* 是**乐观**采购：给它"日内先知"优势 + 认证全局最优，看它在**因果部署**下能否赢 V5；
  · **Φ(E) 终端**（逐月 FVI 估计，复用 §24）替代固定期末下限。

部署（**因果、非前瞻**）：把 LP 给出的 b* 直接在实测日上用 must/max 或价值反馈执行。
判定逻辑：若**给了先知优势 + 认证最优**的采购在因果部署下仍输给 V5 ⇒ 采购结构**确实**
不是瓶颈（强否定，且求解质量混杂已被排除）；若赢 ⇒ 存在真实空间，值得做非前瞻训练。

效率：A_eq 全常数（只建一次）；A_ub/b_ub 只依赖 Φ（按月重建）；每日仅更新目标权重与右端。
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_MIN, E_MAX, P_MAX, ETA_C, ETA_D, E_FEB1,  # noqa: E402
                      WARMUP_DAY, load_data)
import q2_value_adp as va  # noqa: E402

NS = None          # 场景数（运行期由池决定，取固定 S_FIX 以便矩阵只建一次）
S_FIX = 12         # 固定场景数（条件池取前 12 个）
V5_TOTAL = 1583.7354
V6_TOTAL = 1581.6655


# ============================ LP 结构（A_eq 一次建；A_ub/b_ub 按月建） ============================
def build_layout(S):
    ns = 5 * T + 2
    n = T + S * ns
    colB = lambda s: T + s * ns
    return ns, n, colB


def build_Aeq(S, ns, colB):
    """A_eq 全常数：SOC 动态 + 区间平衡 + 初始电量。右端每日更新，故只建系数。"""
    rows, cols, vals = [], [], []
    # SOC 动态块：row = s*T + t
    for s in range(S):
        cB = colB(s)
        cE = cB + 4 * T
        cc, cd = cB + 0, cB + T
        for t in range(T):
            r = s * T + t
            rows += [r, r, r, r]
            cols += [cE + t + 1, cE + t, cc + t, cd + t]
            vals += [1.0, -1.0, -ETA_C, 1.0 / ETA_D]
    # 平衡块：row = S*T + s*T + t ;  b_t + d + e - c - w = net_s[t]
    for s in range(S):
        cB = colB(s)
        cc, cd, ce, cw = cB + 0, cB + T, cB + 2 * T, cB + 3 * T
        for t in range(T):
            r = S * T + s * T + t
            rows += [r, r, r, r, r]
            cols += [t, cd + t, ce + t, cc + t, cw + t]
            vals += [1.0, 1.0, 1.0, -1.0, -1.0]
    # 初始块：row = 2*S*T + s ;  E_s[0] = E_start
    for s in range(S):
        r = 2 * S * T + s
        rows.append(r); cols.append(colB(s) + 4 * T); vals.append(1.0)
    A = coo_matrix((vals, (rows, cols)), shape=(2 * S * T + S, T + S * ns)).tocsr()
    return A


def build_Aub(S, ns, colB, Phi):
    """θ_s ≥ Φ(E_T)：slope_i·E_T − θ_s ≤ slope_i·E_GRID[i] − Φ[i]，i=0..G-2。"""
    G = len(va.E_GRID)
    slopes = np.diff(Phi) / np.diff(va.E_GRID)
    rows, cols, vals, bub = [], [], [], []
    for s in range(S):
        cE_T = colB(s) + 4 * T + T       # E_s[T]
        cth = colB(s) + 5 * T + 1        # θ_s
        for i in range(G - 1):
            r = s * (G - 1) + i
            rows += [r, r]
            cols += [cE_T, cth]
            vals += [slopes[i], -1.0]
            bub.append(slopes[i] * va.E_GRID[i] - Phi[i])
    A = coo_matrix((vals, (rows, cols)), shape=(S * (G - 1), T + S * ns)).tocsr()
    return A, np.array(bub)


def build_bounds(S, ns, colB, n):
    b = [(0.0, None)] * n
    for s in range(S):
        cB = colB(s)
        for t in range(T):
            b[cB + t] = (0.0, P_MAX)          # c
            b[cB + T + t] = (0.0, P_MAX)      # d
            b[cB + 2 * T + t] = (0.0, None)   # e
            b[cB + 3 * T + t] = (0.0, None)   # w
        for t in range(T + 1):
            b[cB + 4 * T + t] = (E_MIN, E_MAX)  # E
        b[cB + 5 * T + 1] = (None, None)        # θ
    return b


def solve_day(price, scen, wsc, E_start, Phi, cache, b_fixed=None):
    """解一天的 anticipative 两阶段 LP，返回 (b*, 训练目标值)。
    b_fixed 不为 None 时：把 b 固定为该合同，只优化储能 ⇒ 该合同的 anticipative（放松）费用，
    用于"政策差距"分解（同一合同下 因果政策 vs 自由储能）。"""
    S = scen.shape[0]
    if S not in cache:
        ns, n, colB = build_layout(S)
        cache[S] = dict(ns=ns, n=n, colB=colB, Aeq=build_Aeq(S, ns, colB),
                        bounds=build_bounds(S, ns, colB, n), Aub={}, bub={})
    ck = cache[S]
    ns, n, colB = ck["ns"], ck["n"], ck["colB"]
    # Φ 终端按月变化 → A_ub/b_ub 缓存（以 Φ 的标识为键）
    key = id(Phi)
    if key not in ck["Aub"]:
        A_ub, b_ub = build_Aub(S, ns, colB, Phi)
        ck["Aub"][key], ck["bub"][key] = A_ub, b_ub
    A_ub, b_ub = ck["Aub"][key], ck["bub"][key]
    # 目标
    c = np.zeros(n)
    c[:T] = price
    for s in range(S):
        cB = colB(s)
        c[cB + 2 * T: cB + 3 * T] = wsc[s] * MULT * price      # e_s
        c[cB + 5 * T + 1] = wsc[s]                             # θ_s
    # 右端（SOC 块=0；平衡块=net_s；初始块=E_start）
    beq = np.zeros(2 * S * T + S)
    for s in range(S):
        beq[S * T + s * T: S * T + (s + 1) * T] = scen[s]
        beq[2 * S * T + s] = E_start
    bounds = ck["bounds"]
    if b_fixed is not None:
        bounds = list(bounds)
        for t in range(T):
            bounds[t] = (float(b_fixed[t]), float(b_fixed[t]))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=ck["Aeq"], b_eq=beq,
                  bounds=bounds, method="highs")
    assert res.status == 0, f"LP 失败: {res.message}"
    return res.x[:T].copy(), float(res.fun)


# ============================ 全年：乐观采购 + 因果部署 ============================
def run(D, idx, Estar_of, Phi_of, month_start):
    price, net, months = D["price"], D["net"], D["_month"]
    nd = len(D["dates"])
    cache = {}
    # 三条部署轨迹共享同一份 b*（每天只解一次 LP）
    dep = {k: dict(dp=np.zeros(nd), de=np.zeros(nd), dk=np.zeros(nd),
                   soc0=np.zeros(nd), soc1=np.zeros(nd), socmin=np.zeros(nd))
           for k in ("must", "value")}
    Emust = float(E_FEB1); Evalue = float(E_FEB1)
    train_obj = np.zeros(nd)
    t0 = time.perf_counter()
    for d in range(WARMUP_DAY, nd):
        m = months[d]
        scen, wsc = va.scenario_pool(d, D, K=S_FIX)
        # must 与 value 用各自的 SOC 起点解 b*？—— 共享同一天起点会破坏两轨独立性。
        # 这里 b* 用 must 轨迹的起点解一次，value 复用同一 b*（合同共享），仅执行不同。
        b, tobj = solve_day(price, scen, wsc, Emust, Phi_of[m], cache)
        train_obj[d] = tobj
        c, dd, e, w, Etraj = va.execute_value(b, net[d], price, Emust, np.full(T, E_MIN))
        dep["must"]["dp"][d] = float((b * price).sum()); dep["must"]["de"][d] = float(MULT * (e * price).sum())
        dep["must"]["dk"][d] = float(e.sum()); dep["must"]["soc0"][d] = Emust
        dep["must"]["soc1"][d] = Etraj[-1]; dep["must"]["socmin"][d] = Etraj.min()
        Emust = Etraj[-1]
        c, dd, e, w, Etraj = va.execute_value(b, net[d], price, Evalue, Estar_of[m])
        dep["value"]["dp"][d] = float((b * price).sum()); dep["value"]["de"][d] = float(MULT * (e * price).sum())
        dep["value"]["dk"][d] = float(e.sum()); dep["value"]["soc0"][d] = Evalue
        dep["value"]["soc1"][d] = Etraj[-1]; dep["value"]["socmin"][d] = Etraj.min()
        Evalue = Etraj[-1]
        if (d - WARMUP_DAY) % 40 == 0:
            print(f"    d={d} ({str(D['dates'][d])[:10]}) 累计 {time.perf_counter()-t0:.0f}s", flush=True)
    return dep, train_obj


def summ(name, r, idx, price):
    p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum())
    dE = max(0.0, E_FEB1 - float(r["soc1"][idx][-1]))
    return dict(name=name, plan=p / 1e4, em=e / 1e4, total=(p + e) / 1e4,
                P99=float(np.quantile(r["de"][idx] / 1e4, 0.99)),
                大紧急=int((r["dk"][idx] > 1000).sum()),
                起始SOC=float(r["soc0"][idx].mean()),
                触底=int((r["socmin"][idx] <= E_MIN + 1e-3).sum()),
                折算中=(p + e) / 1e4 + float(price.mean()) * dE / ETA_C / 1e4)


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates = D["price"], D["dates"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    month_start = {m: int(np.min(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]
    quick = "--quick" in sys.argv

    print("=" * 104)
    print("§25 认证随机 LP 采购（anticipative 乐观界）+ 因果部署")
    print("=" * 104)
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）"
          f"{' [QUICK]' if quick else ''}；场景数固定 S={S_FIX}；公共预热 E={E_FEB1:.0f}")

    print("\n[0] 逐月价值函数（取 Φ 与 E*，复用 §24）")
    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        scen = va.fvi_pool(month_start[m], D)
        _, Estar, Phi = va.estimate_value(price, scen)
        Estar_of[m], Phi_of[m] = Estar, Phi
    print("    完成")

    print("\n[1] 乐观采购 LP（每天一次认证全局最优）+ 因果部署（must / value）")
    dep, train_obj = run(D, idx, Estar_of, Phi_of, month_start)

    print("\n" + "=" * 104)
    print("汇总（评价期，万元）")
    print("=" * 104)
    rows = [summ("LP采购+must", dep["must"], idx, price),
            summ("LP采购+value", dep["value"], idx, price)]
    print(f"  {'方案':<18}{'计划':>10}{'紧急':>10}{'总费用':>11}{'相对V5':>10}{'相对V6':>10}"
          f"{'P99':>8}{'起始SOC':>9}{'折算中':>10}")
    for r in rows:
        print(f"  {r['name']:<18}{r['plan']:>10.4f}{r['em']:>10.4f}{r['total']:>11.4f}"
              f"{r['total']-V5_TOTAL:>+10.4f}{r['total']-V6_TOTAL:>+10.4f}{r['P99']:>8.3f}"
              f"{r['起始SOC']:>9.0f}{r['折算中']:>10.4f}")
    print(f"\n  [参照] V5 = {V5_TOTAL:.4f}｜V6 = {V6_TOTAL:.4f}")
    print(f"  [乐观训练目标均值] anticipative LP 每日最优目标 = {train_obj[idx].mean()/1e4:.4f} 万元/天"
          f"（含 Φ 终端；是**不可实施**的信息放松参考）")

    out = dict(口径=dict(S=S_FIX, quick=quick, 评价期天数=int(len(idx))),
               LP_must=rows[0]["total"], LP_value=rows[1]["total"],
               V5=V5_TOTAL, V6=V6_TOTAL,
               detail={r["name"]: r for r in rows})
    tag = "_quick" if quick else ""
    (BASE / "results" / f"q2_saa_lp{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_saa_lp{tag}.json   总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
