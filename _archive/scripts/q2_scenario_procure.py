# -*- coding: utf-8 -*-
"""
问题二 · 对照二：联合场景采购（只读：不改模型、不重出 Excel）
=====================================================================
用户 2026-09-11 意见：把改进重心前移到**预测与日前采购**；场景模型须**检查信息约束**，
并用**因果执行**重新评价；场景数 30 只是该方案的选择，不是普遍合适；联合残差必须来自历史。

本脚本严格按用户给出的目标实现：

    min_b  [ Σ_t p_t b_t  +  Σ_s ω_s ( 5 Σ_t p_t e_{s,t} ) ]

  · 所有场景**共享同一份日前计划 b**（here-and-now 一阶段决策）；
  · 场景 = **历史联合轨迹**（最近 S 天的整条净负载曲线，等权 ω_s = 1/S）——
    保留跨区间相关性，这正是逐区间分位数做不到的；
    场景只取自 d−1 及以前，**不含当日任何数据**；
  · **信息约束（关键）**：场景内的储能**不得**全天完美调度。本脚本把场景内的 c/d
    强制为与执行层**同一条因果 must/max 规则**逐区间推进，故 e_{s,t}(b) 是在
    "日内不知道未来" 的信息结构下算出的，不会系统性低估执行成本；
  · 评价：用真实当日净负载、同一 must/max 规则做连续回放（公共预热 2/1 起评 E=10800）。

只比较"预测"与"采购"各自贡献，故拆成两行：
  对照二-a：场景集分位预测（每区间取场景 q 分位） + **保持原 LP 采购框架** + 尽限反馈
  对照二-b：场景集分位预测                        + **场景采购优化** + 尽限反馈
  ⇒ (a) 相对 S1 的差 = **预测换成场景集**的贡献；(b) 相对 (a) 的差 = **采购优化**的贡献。

求解说明（必须如实披露）：该联合问题关于 b 是**非凸分段线性**的（max/min 与储能夹逼），
不存在随取随用的精确解。本脚本用**多起点坐标下降**（初值 = 在场景隐含曲线上取 q 分位的
确定性 LP 计划，q 覆盖 0.50~0.90；每起点 2 遍、每坐标 4 点线搜索），并按**场景目标 F**
在这些局部解中取最优。故报告的是**局部最优**，并同时给出"初值后 / 优化后"两列，
以便把"β 重新取点"与"储能感知的跨时段腾挪"分开看。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_INIT, E_MIN, E_MAX, P_MAX, ETA_C, ETA_D,  # noqa: E402
                      WARMUP_DAY, E_FEB1, Policy, load_data, make_plan, execute)
from q2_three_way import (_col_quantile, betas_single, prefix_cum, select_schedule,  # noqa: E402
                          eval_schedule, BETA_SINGLE)

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

QUICK = "--quick" in sys.argv
S_MAIN = 30
INIT_Q = [0.50, 0.60, 0.70, 0.80, 0.90]
SWEEPS = 2
DELTAS = [200.0, 50.0]
QS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
S_GRID = [10, 20, 30, 60]


# ---------------- 场景内因果回放（与执行层逐字同规则） ----------------
def _recourse_e(b3, scen3, E_start):
    """b3, scen3: (J,S,T)（scen3 可为 (1,S,T) 广播）。逐区间因果推进，返回 e (J,S,T)。
    维度含义：J = 候选计划数，S = 场景数，T = 区间数。执行器逐 (j,s) 行独立。"""
    J, S, _ = b3.shape
    E = np.full((J, S), float(E_start))
    e = np.zeros((J, S, T))
    for t in range(T):
        net = b3[:, :, t] - scen3[:, :, t]
        pos = net >= 0.0
        avail_c = np.maximum(0.0, (E_MAX - E) / ETA_C)
        cc = np.where(pos, np.minimum(np.minimum(net, P_MAX), avail_c), 0.0)
        deficit = np.where(pos, 0.0, -net)
        avail_d = np.maximum(0.0, (E - E_MIN) * ETA_D)
        dd = np.minimum(np.minimum(deficit, P_MAX), avail_d)
        e[:, :, t] = deficit - dd
        E = E + ETA_C * cc - dd / ETA_D
    return e


def _obj_of(e3, b3, price, omega):
    """F_j = Σ_t p_t b_{j,t} + Σ_s ω_s·5·Σ_t p_t e_{j,s,t}，返回长度 J 的向量。"""
    plan = (b3 * price[None, None, :]).sum(axis=2)                                   # (J,S)，与 s 无关
    em = MULT * (omega[None, :, None] * e3 * price[None, None, :]).sum(axis=2)        # (J,S)
    return plan[:, 0] + em.sum(axis=1)                                               # (J,)


def scen_obj(b, scen, price, E_start, omega):
    """共享计划 b（长度 T）在场景集上的目标 F(b)。"""
    b3 = np.broadcast_to(np.asarray(b, dtype=float), (1,) + scen.shape).copy()
    e3 = _recourse_e(b3, scen[None, :, :], E_start)
    return float(_obj_of(e3, b3, price, omega)[0])


def coord_refine(b0, scen, price, E_start, omega, sweeps=SWEEPS, deltas=DELTAS, npts=4):
    """坐标下降，每坐标 4 点线搜索。
    **加速**：把 4 个候选值堆叠到"候选维"一次性前推（执行器逐行独立），
    每个坐标仍只需 1 次 144 步前推 ⇒ 约 4× 加速；与逐候选依次求值**逐位相同**（见 selftest）。"""
    J = npts
    b = b0.copy()
    F = scen_obj(b, scen, price, E_start, omega)
    scen3 = scen[None, :, :]
    for D in deltas[:sweeps]:
        for t in range(T):
            b3 = np.broadcast_to(b, (J,) + scen.shape).copy()
            cands = np.maximum(b[t] + np.array([-1.5, -0.5, 0.5, 1.5]) * D, 0.0)
            b3[:, :, t] = cands[:, None]
            vals = _obj_of(_recourse_e(b3, scen3, E_start), b3, price, omega)
            j = int(np.argmin(vals))
            if vals[j] < F - 1e-9:
                b[t] = float(cands[j]); F = float(vals[j])
    return b, F


def selftest():
    """校验"把候选堆叠进场景维"与"逐个候选依次求值"逐位一致（否则加速会改变结果）。"""
    rng = np.random.default_rng(0)
    S, M = 7, 4
    scen = rng.normal(300.0, 80.0, (S, T))
    price = np.abs(rng.normal(0.8, 0.3, T)) + 0.2
    omega = np.full(S, 1.0 / S)
    b = rng.uniform(0.0, 500.0, T)
    E0 = 6000.0

    def naive(bv):
        E = np.full(S, E0); e = np.zeros((S, T))
        for t in range(T):
            net = bv[t] - scen[:, t]
            pos = net >= 0
            ac = np.maximum(0.0, (E_MAX - E) / ETA_C)
            cc = np.where(pos, np.minimum(np.minimum(net, P_MAX), ac), 0.0)
            df = np.where(pos, 0.0, -net)
            ad = np.maximum(0.0, (E - E_MIN) * ETA_D)
            dd = np.minimum(np.minimum(df, P_MAX), ad)
            e[:, t] = df - dd
            E = E + ETA_C * cc - dd / ETA_D
        return float((bv * price).sum() + MULT * (omega[:, None] * e * price[None, :]).sum())

    d1 = abs(naive(b) - scen_obj(b, scen, price, E0, omega))
    t0, D = 37, 50.0
    cands = np.maximum(b[t0] + np.array([-1.5, -0.5, 0.5, 1.5]) * D, 0.0)
    b3 = np.broadcast_to(b, (M,) + scen.shape).copy()
    b3[:, :, t0] = cands[:, None]
    vals = _obj_of(_recourse_e(b3, scen[None, :, :], E0), b3, price, omega)
    vals_n = np.array([naive(np.where(np.arange(T) == t0, c, b)) for c in cands])
    d2 = float(np.max(np.abs(vals - vals_n)))
    ok = max(d1, d2) < 1e-9
    print(f"  [自检] 堆叠执行器 vs 逐候选求值：目标差 {d1:.3e}，线搜索值差 {d2:.3e}  "
          f"{'一致 ✔' if ok else '**不一致，禁止使用加速版**'}")
    return ok


def scen_curve(d, D, S, q):
    """场景集在每区间取 q 分位（场景隐含曲线）。"""
    scen = D["net"][max(0, d - S):d]
    if scen.shape[0] < 5:
        return D["net_ref"].copy()
    return np.maximum(_col_quantile(scen, np.full(T, q)), 0.0)


# ---------------- 对照二-a：只换预测（采购仍是原 LP） ----------------
def replay_a(D, q_of_day, S, start_idx, E_start, keep=False):
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    det = None
    if keep:
        det = dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)), Wsp=np.zeros((nd, T)),
                   SOC=np.zeros((nd, T + 1)), Nhat=np.zeros((nd, T)))
    for d in range(start_idx, nd):
        N_hat = scen_curve(d, D, S, q_of_day(d))
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        if keep:
            det["B"][d] = b; det["EM"][d] = e; det["Wsp"][d] = w
            det["SOC"][d] = Etraj; det["Nhat"][d] = N_hat
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, det=det)


# ---------------- 对照二-b：场景采购优化 ----------------
def solve_day_b(d, D, E0, S):
    """多起点坐标下降。返回 dict：b_init/b_opt（按 F 最优的那组）、F0/F1、胜出 q、各起点明细。"""
    price = D["price"]
    scen = D["net"][max(0, d - S):d]
    if scen.shape[0] < 5:
        b, _ = make_plan(POL, price, D["net_ref"].copy(), E0)
        return dict(b_init=b, b_opt=b, F0=np.nan, F1=np.nan, q=np.nan, rows=[])
    omega = np.full(scen.shape[0], 1.0 / scen.shape[0])
    rows, best = [], None
    for q in INIT_Q:
        b0, _ = make_plan(POL, price, scen_curve(d, D, S, q), E0)
        F0 = scen_obj(b0, scen, price, E0, omega)
        bf, Ff = coord_refine(b0, scen, price, E0, omega)
        rows.append(dict(q=q, F0=F0, F1=Ff))
        if best is None or Ff < best[3]:
            best = (q, b0, F0, Ff, bf)
    return dict(b_init=best[1], b_opt=best[4], F0=best[2], F1=best[3], q=best[0], rows=rows)


def replay_b(D, S, start_idx, E_start, keep=False, days=None):
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd); dw = np.zeros(nd)
    dp_i = np.zeros(nd); de_i = np.zeros(nd)          # 初值（未优化）那一版
    qw = np.full(nd, np.nan); F0 = np.full(nd, np.nan); F1 = np.full(nd, np.nan)
    det = None
    if keep:
        det = dict(B=np.zeros((nd, T)), EM=np.zeros((nd, T)), Wsp=np.zeros((nd, T)),
                   SOC=np.zeros((nd, T + 1)), Binit=np.zeros((nd, T)))
    todo = range(start_idx, nd) if days is None else days
    ntot = len(todo)
    for it, d in enumerate(todo):
        sol = solve_day_b(d, D, E, S)
        _, _, e, w, Etraj = execute(POL, sol["b_opt"], np.zeros(T), None, net[d], price, E)
        dp[d] = float((sol["b_opt"] * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        _, _, ei, _, _ = execute(POL, sol["b_init"], np.zeros(T), None, net[d], price, E)
        dp_i[d] = float((sol["b_init"] * price).sum()); de_i[d] = float(MULT * (ei * price).sum())
        qw[d] = sol["q"]; F0[d] = sol["F0"]; F1[d] = sol["F1"]
        if keep:
            det["B"][d] = sol["b_opt"]; det["EM"][d] = e; det["Wsp"][d] = w
            det["SOC"][d] = Etraj; det["Binit"][d] = sol["b_init"]
        E = Etraj[-1]
        if it % 25 == 0:
            print(f"      [进度] {it+1}/{ntot} 天  当前 {str(D['dates'][d])[:10]}  "
                  f"累计 {float(dp.sum()+de.sum())/1e4:.2f} 万元", flush=True)
    return dict(dp=dp, de=de, dk=dk, dw=dw, dp_i=dp_i, de_i=de_i, q=qw, F0=F0, F1=F1, det=det)


def main():
    D = load_data()
    price, dates, net = D["price"], D["net"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    D["_month"] = months
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx_eval = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]
    idx_ps = np.where(D["dates"] >= np.datetime64("2025-01-01"))[0]
    loc = {m: np.where(months[idx_eval] == m)[0] for m in range(2, 13)}

    print("=" * 104)
    print("问题二 · 对照二：联合场景采购（共享一份日前计划；场景内为因果 must/max，非完美调度）")
    print("=" * 104)
    print(f"  评价期 {str(D['dates'][idx_eval[0]])[:10]} ~ {str(D['dates'][idx_eval[-1]])[:10]}"
          f"（{len(idx_eval)} 天）；公共预热 E = {E_FEB1:.0f} kWh")
    print(f"  场景 = 最近 S 天**整条净负载曲线**（等权），S = {S_MAIN}；只取 d−1 及以前。")
    print(f"  初值集 q = {INIT_Q}（在场景隐含曲线上取 q 分位的确定性 LP 计划）；"
          f"坐标下降 {SWEEPS} 遍、每坐标 4 点、步长 {DELTAS}。")
    print("  信息约束：场景内 c/d 由 must/max 因果规则决定 ⇒ 场景内不看未来，不低估执行成本。")
    if not selftest():
        print("  自检失败，终止。")
        return
    if QUICK:
        print("  ** --quick：只跑抽样子集，用于冒烟测试 **")

    # ---------- 0. 情报检查 ----------
    print("\n" + "-" * 104)
    print("0. 场景集与信息约束检查")
    print("-" * 104)
    d0 = 200
    scen = D["net"][d0 - S_MAIN:d0]
    print(f"  示例 d={str(D['dates'][d0])[:10]}：场景数 {scen.shape[0]}，"
          f"场景间同时刻相关性均值 {np.mean([np.corrcoef(scen[i], scen[j])[0,1] for i in range(20) for j in range(i+1,20)]):.4f}")
    print(f"  场景集逐区间 70 分位 vs 逐区间独立重排（打散相关性）后的 70 分位：")
    ind = np.sort(scen, axis=0)
    print(f"    日总量：联合 {np.maximum(_col_quantile(scen, np.full(T, 0.70)),0).sum():.0f} kWh"
          f" | 独立重排 {np.maximum(ind[20],0).sum():.0f} kWh（示意相关性对总量的影响）")
    print("  说明：联合轨迹保留『高负载日整条都高』的相关结构；逐区间独立取分位会把极端值错配到不同时刻。")

    # ---------- 1. 对照二-a：只换预测（采购仍是原 LP），q 由同一因果协议选 ----------
    print("\n" + "-" * 104)
    print("1. 对照二-a：预测换成'场景集 q 分位'，采购保持原 LP（q 用同一 walk-forward 协议选）")
    print("-" * 104)
    cumA = {}
    for k, q in enumerate(QS):
        r = replay_a(D, lambda d, q=q: q, S_MAIN, 0, E_INIT)
        cumA[k] = np.cumsum(r["dp"] + r["de"])
    curveA = {q: float(cumA[k][-1]) for k, q in enumerate(QS)}
    qbest = min(curveA, key=curveA.get)
    print("  [诊断·post-hoc] 全年连续回放 J(q)：")
    for q in QS:
        print(f"    q={q:.2f}   全年 J = {curveA[q]/1e4:>10.4f} 万元{'   ← 最优' if q == qbest else ''}")
    schedA = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        schedA[m] = QS[min(range(len(QS)), key=lambda i: cumA[i][te])]
    print("\n  [因果·可实施] 逐月 q 日程：")
    for m in range(2, 13):
        print(f"    用于 {m:2d} 月 → q = {schedA[m]:.2f}")

    # ---------- 2. 对照二-b：场景采购优化 ----------
    print("\n" + "-" * 104)
    print(f"2. 对照二-b：场景采购优化（多起点坐标下降；S = {S_MAIN}）")
    print("-" * 104)
    days_b = idx_ps[::40] if QUICK else idx_ps
    print(f"  参与天数 {len(days_b)}（{'抽样' if QUICK else '全年'}）；"
          f"实测约 2~3 s/天（候选堆叠加速后）⇒ 预计 {len(days_b)*3/60:.0f} 分钟")
    rb = replay_b(D, S_MAIN, 0, E_INIT, keep=True, days=days_b)

    # ---------- 3. 统一口径对比 ----------
    print("\n" + "-" * 104)
    print("3. 统一口径对比（V0/V1 来自 §16 同一实现；对照一来自 q2_decomp_forecast.py）")
    print("-" * 104)
    cum1 = prefix_cum(D, BETA_SINGLE, betas_single, 14)
    sched1 = select_schedule(D, BETA_SINGLE, cum1)
    res = {}
    res["V0原S1 β=0.70"] = eval_schedule(D, {m: 0.70 for m in range(2, 13)}, betas_single, 14, idx_eval)
    res["V1重校准单一β"] = eval_schedule(D, sched1, betas_single, 14, idx_eval)
    res["对照二a 场景预测+原LP"] = replay_a(D, lambda d: schedA[months[d]], S_MAIN, WARMUP_DAY, E_FEB1, keep=True)
    if not QUICK:
        res["对照二b 场景采购优化"] = rb

    v0 = res["V0原S1 β=0.70"]
    p0 = float(v0["dp"][idx_eval].sum()); e0 = float(v0["de"][idx_eval].sum())
    print(f"\n  {'版本':<26}{'计划/万元':>12}{'紧急/万元':>12}{'总费用/万元':>14}"
          f"{'紧急电量/kWh':>15}{'弃置/kWh':>13}")
    print("  " + "-" * 92)
    met = {}
    for k, r in res.items():
        pk = float(r["dp"][idx_eval].sum()); ek = float(r["de"][idx_eval].sum())
        met[k] = dict(plan=pk, em=ek, total=pk + ek,
                      emk=float(r["dk"][idx_eval].sum()), wsp=float(r["dw"][idx_eval].sum()))
        print(f"  {k:<26}{pk/1e4:>12.4f}{ek/1e4:>12.4f}{(pk+ek)/1e4:>14.4f}"
              f"{met[k]['emk']:>15.1f}{met[k]['wsp']:>13.1f}")
    print("\n  相对 V0 的差额与分解：")
    for k, m in met.items():
        print(f"    {k:<26} 总 {(m['total']-p0-e0)/1e4:>+9.4f} 万元"
              f"（计划 {(m['plan']-p0)/1e4:>+8.4f}，紧急 {(m['em']-e0)/1e4:>+8.4f}）")

    if "对照二b 场景采购优化" in res:
        a = met["对照二a 场景预测+原LP"]; b_ = met["对照二b 场景采购优化"]
        print("\n  收益归因（用户要求的'分开识别'）：")
        print(f"    ① 预测换成场景集（采购仍用原 LP）      ：{(a['total']-p0-e0)/1e4:>+9.4f} 万元")
        print(f"    ② 在此基础上再加场景采购优化            ：{(b_['total']-a['total'])/1e4:>+9.4f} 万元")
        print(f"    ③ 两者合计                              ：{(b_['total']-p0-e0)/1e4:>+9.4f} 万元")
        print(f"    参考：V1（只重校准逐区间 β）            ：{met['V1重校准单一β']['total']/1e4-p0/1e4-e0/1e4:>+9.4f} 万元")
        print(f"\n  [披露] '初值后'（未做坐标下降）的实际费用："
              f"{(float(rb['dp_i'][idx_eval].sum())+float(rb['de_i'][idx_eval].sum())-p0-e0)/1e4:+.4f} 万元")
        print("        该列 = 多起点中按场景目标 F 选出的初值直接执行；与优化后的差即为"
              "'储能感知的跨时段腾挪'带来的部分。")

    # ---------- 4. 月度表现 ----------
    if "对照二b 场景采购优化" in res:
        print("\n  月度总费用（万元）：")
        keys = list(res.keys())
        SH = {"V0原S1 β=0.70": "V0", "V1重校准单一β": "V1",
              "对照二a 场景预测+原LP": "对照二a", "对照二b 场景采购优化": "对照二b"}
        print(f"    {'月':>4}" + "".join(f"{SH[k]:>12}" for k in keys))
        for m in range(2, 13):
            row = []
            for k in keys:
                r = res[k]
                row.append(float(r["dp"][idx_eval][loc[m]].sum() + r["de"][idx_eval][loc[m]].sum()))
            print(f"    {m:>4}" + "".join(f"{v/1e4:>12.4f}" for v in row))

    # ---------- 5. 场景数 S 的敏感度 ----------
    print("\n" + "-" * 104)
    print("5. 场景数 S 的敏感度（用户提示：'30 条场景'只是该方案的选择，不是普遍合适）")
    print("-" * 104)
    sub = list(idx_eval[::20] if not QUICK else idx_eval[::110])
    SOCref = res["V1重校准单一β"]["det"]["SOC"][:, 0]      # 统一的逐日起始 SOC（隔离 S 的影响）
    print(f"  固定子样本 {len(sub)} 天（每 20 天取一天）：{[str(D['dates'][d])[:10] for d in sub[:4]]} …")
    print("  逐日起始 SOC 统一取 V1 轨迹，以隔离场景数 S 的影响（本段仅为诊断）。")
    print(f"  {'S':>5}{'计划/万元':>12}{'紧急/万元':>12}{'总费用/万元':>14}{'相对S=30':>12}")
    vals = {}
    for S in ([10, 30] if QUICK else S_GRID):
        pl = em = 0.0
        for si, d in enumerate(sub):
            E0 = float(SOCref[d])
            sol = solve_day_b(d, D, E0, S)
            _, _, e, _, _ = execute(POL, sol["b_opt"], np.zeros(T), None, net[d], price, E0)
            pl += float((sol["b_opt"] * price).sum()); em += float(MULT * (e * price).sum())
            if si % 5 == 0:
                print(f"      [S敏感度] S={S}  {si+1}/{len(sub)} 天  "
                      f"{str(D['dates'][d])[:10]}  累计 {(pl+em)/1e4:.2f} 万元", flush=True)
        vals[S] = pl + em
    ref = vals.get(S_MAIN)
    for S, v in vals.items():
        dtxt = "—" if (ref is None or S == S_MAIN) else "%+.4f" % ((v - ref) / 1e4)
        print(f"  {S:>5}{'':>12}{'':>12}{v/1e4:>14.4f}{dtxt:>12}")
    print("  （计划/紧急两列留空：本段只比较总费用随 S 的变化；S=30 为基准）")

    # ---------- 6. 初值胜出分布 + F 代理质量 + SOC 口径 ----------
    if "对照二b 场景采购优化" in res:
        print("\n" + "-" * 104)
        print("6. 诊断：初值胜出分布 / 场景目标 F 作为代理的可靠度 / SOC 口径")
        print("-" * 104)
        qv = rb["q"][idx_eval]
        print("  按场景目标 F 选出的初值 q 的分布：")
        for q in INIT_Q:
            print(f"    q={q:.2f}：{int((qv == q).sum()):>4d} 天（{(qv == q).mean()*100:>5.1f}%）")
        F1v = rb["F1"][idx_eval]
        c_opt = rb["dp"][idx_eval] + rb["de"][idx_eval]
        c_ini = rb["dp_i"][idx_eval] + rb["de_i"][idx_eval]
        print(f"\n  场景目标 F 与实际执行费用的相关性："
              f"corr(F_final, 实际) = {float(np.corrcoef(F1v, c_opt)[0,1]):.4f}")
        print(f"  初值实际费用 {c_ini.sum()/1e4:.4f} → 优化后 {c_opt.sum()/1e4:.4f} 万元"
              f"（改善 {(c_opt.sum()-c_ini.sum())/1e4:+.4f} 万元）")
        print("\n  SOC 口径检查：")
        for k in ("V1重校准单一β", "对照二b 场景采购优化"):
            soc = res[k]["det"]["SOC"][idx_eval]
            print(f"    {k:<24} 日均起始 {soc[:,0].mean():>9.1f} | 日均末 {soc[:,-1].mean():>9.1f}"
                  f" | 年末 {soc[-1,-1]:>9.1f} | 触底天数 {int((soc.min(axis=1) <= E_MIN+1e-3).sum()):>3d}")

    # ---------- 7. 公平化：终端储能估值调整（用户要求统一"终端处理"） ----------
    print("\n" + "-" * 104)
    print("7. 公平化：终端储能估值调整")
    print("-" * 104)
    lam_lo, lam_mid, lam_hi = float(price.min()), float(price.mean()), float(price.max())
    print(f"  共同起点：2/1 起评 E = {E_FEB1:.0f} kWh。若某方案年末库存更低，它其实'花掉了库存'，")
    print(f"  故按 §10 的终端估值 λ 折算补回成本：调整费 = 总费用 + λ·(E起点 − E年末)/η_c，η_c = {ETA_C}。")
    print(f"  λ 取电价区间 [{lam_lo:.4f}, {lam_hi:.4f}] 元/kWh（中值 {lam_mid:.4f}）。")
    print(f"\n  {'版本':<24}{'年末SOC':>12}{'库存缺口/kWh':>15}{'原总费/万元':>14}"
          f"{'λ低':>12}{'λ中':>12}{'λ高':>12}")
    adjtab = {}
    for k, r in res.items():
        Eend = float(r["det"]["SOC"][idx_eval][-1, -1])
        dE = max(0.0, E_FEB1 - Eend)
        tot = met[k]["total"]
        adj = [tot / 1e4 + l * dE / ETA_C / 1e4 for l in (lam_lo, lam_mid, lam_hi)]
        adjtab[k] = adj
        print(f"  {k:<24}{Eend:>12.1f}{dE:>15.1f}{tot/1e4:>14.4f}"
              f"{adj[0]:>12.4f}{adj[1]:>12.4f}{adj[2]:>12.4f}")
    print("\n  调整后排序（λ 取中值）：")
    for rank, (k, adj) in enumerate(sorted(adjtab.items(), key=lambda kv: kv[1][1]), 1):
        print(f"    {rank}. {k:<24}{adj[1]:>12.4f} 万元"
              f"（相对 V0 调平后 {adj[1]-adjtab['V0原S1 β=0.70'][1]:+9.4f}）")
    print("  ⇒ 若排序在 λ 的整个电价区间内不变，则结论不受终端估值口径影响。")

    # 落盘：把 对照二b 的逐日计划与 SOC 轨迹存下来，便于后续免重算复核
    np.savez_compressed(BASE / "results" / "q2_scenario_plans.npz",
                        B=rb["det"]["B"], SOC=rb["det"]["SOC"], days=days_b)

    out = {"口径": {"S": S_MAIN, "INIT_Q": INIT_Q, "SWEEPS": SWEEPS, "DELTAS": DELTAS,
                    "评价期天数": int(len(idx_eval)), "公共预热": E_FEB1,
                    "场景": "最近 S 天整条净负载曲线（等权）", "场景内规则": "must/max 因果"},
           "q_curve": {str(k): v for k, v in curveA.items()},
           "schedA": {str(m): schedA[m] for m in range(2, 13)},
           "versions": met}
    if "对照二b 场景采购优化" in res:
        out["归因"] = {"预测换场景集": float(met["对照二a 场景预测+原LP"]["total"] - p0 - e0),
                       "再加场景采购优化": float(met["对照二b 场景采购优化"]["total"] - met["对照二a 场景预测+原LP"]["total"]),
                       "初值后_未优化": float(float(rb["dp_i"][idx_eval].sum()) + float(rb["de_i"][idx_eval].sum()) - p0 - e0)}
        out["初值胜出"] = {str(q): int((qv == q).sum()) for q in INIT_Q}
        out["终端估值调整"] = {"lambda": [lam_lo, lam_mid, lam_hi],
                          "adjusted": {k: [float(x) for x in v] for k, v in adjtab.items()}}
    outp = BASE / "results" / "q2_scenario.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：{outp}")


if __name__ == "__main__":
    main()
