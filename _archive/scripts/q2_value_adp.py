# -*- coding: utf-8 -*-
r"""
§24 结构性挑战：库存价值函数 + 条件场景 + 非前瞻采购/储能联合优化（只读，不改模型）
================================================================================
背景（用户 2026-09-11 深夜意见）：此前把"现有策略族内调参收益变小"过早等同于"应停止探索"，
并错误地得出"剩余差距只能靠外部数据"。该结论**撤回**。真正受限的是**决策结构**：

    历史分位数曲线  →  确定性购电 LP  →  尽限充放电（must/max）

三重简化：① 把不确定性压成一条分位曲线；② 缺口一来就放到底（must），不比较"现在放"与
"留给更贵时段"的价值；③ 用固定期末下限（E_H≥2400）代替跨日库存价值。

本脚本做一件**结构性**的事（不是再调 β/W/L）：

  · **库存价值函数** V_t(E)：在 SOC 网格上做**拟合值迭代**（backward FVI，跨日 Phi 不动点），
    得到边际价值 λ_t(E) = −∂V_t/∂E，并导出**逐时目标水位** E_star(t)——
    "电价低时放电价高时放"的非短视释放曲线。must/max 只是 E_star≡E_MIN 的特例。
  · **价值反馈执行**：放电 = min(缺口, 功率, η_d·max(E − E_star(t),0))，**因果、非前瞻**
    （只用当前实测与历史估计的 V，不用未来实现）。
  · **条件场景**：近期 30 天按"新近度 × 近期净负荷水平相似度"加权（取代 §19 的最近30天等权）。
  · **非前瞻采购**：min_b Σp·b + E_s[5Σp·e(s) + Φ(E_144(s))]，储能与执行层**同一策略类**，
    用分块坐标下降 + 候选堆叠前推（§19 的 13× 加速）。

消融（用户指定的 2×2，全部同一 2/1 起点 E=10800、连续回放、评价期只用于报告）：
  V5            = 现状基线（分位计划 + geq2400 + must/max）          —— 已有 1583.7354
  P2 固定合同+价值反馈 = **沿用 V5 的购电序列 b**，只把执行换成价值反馈  → "主动留电值多少钱"
  P3 新采购+尽限       = 条件场景随机采购 b，执行仍 must/max           → "采购结构是否限制费用"
  P4 新采购+价值反馈   = 两者联合                                       → "是否有协同"

纪律：选参/估值只用当期之前的数据（逐月 V 用该月之前的数据估计；场景池只用 d 之前的天）；
普通购电每天只在 0:00 定一次；日内只更新储能动作，**不得偷加普通价补购**。
"""
from __future__ import annotations

import io
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_MIN, E_MAX, P_MAX, ETA_C, ETA_D, E_INIT, E_FEB1,  # noqa: E402
                      WARMUP_DAY, Policy, load_data, solve_lp, execute)
from q2_three_way import betas_single, betas_grouped, forecast_custom, GNAMES  # noqa: E402
import q2_grouped_v4 as g4  # noqa: E402

POL = Policy("S1", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

# SOC 网格（"少量库存节点的分段线性近似"）
G_STEP = 240.0
E_GRID = np.arange(E_MIN, E_MAX + 0.5 * G_STEP, G_STEP)   # 41 节点
NPHI = 3                      # 跨日 Phi 不动点迭代次数
FVI_LOOKBACK = 60             # 逐月估值用的近期天数
POOL_LOOKBACK = 30            # 条件场景池的近期天数
POOL_K = 16                   # 条件场景池取前 K 个
RHO = 0.94                    # 新近度权重

V5_TOTAL = 1583.7354          # §23 已核验基线（对拍用）


# ============================ 价值函数：拟合值迭代 ============================
def interp_grid(x, V):
    """在 E_GRID 上插值 V（端点夹紧）。x 任意形状。"""
    return np.interp(np.asarray(x, dtype=float).ravel(), E_GRID, V).reshape(np.shape(x))


def pava_decreasing(y):
    """等权 isotonic：把 y 投影为**非增**序列（用于 λ ≥ 0 且随 E 非增）。"""
    lv, lw = [], []
    for v in np.asarray(y, dtype=float):
        lv.append(float(v)); lw.append(1.0)
        while len(lv) >= 2 and lv[-2] < lv[-1]:
            v2, w2 = lv.pop(), lw.pop()
            v1, w1 = lv.pop(), lw.pop()
            lv.append((v1 * w1 + v2 * w2) / (w1 + w2)); lw.append(w1 + w2)
    return np.concatenate([np.full(int(round(w)), v) for v, w in zip(lv, lw)])


def lambda_from_V(Vnext):
    """由成本-到-去 V（随 E 递减）取边际价值 λ = −dV/dE，投影为 ≥0 且非增。"""
    slopes = np.diff(Vnext) / np.diff(E_GRID)
    lam = np.maximum(-slopes, 0.0)
    return pava_decreasing(lam)


def estar_from_lambda(lam, thresh):
    """目标水位：λ 随 E 非增，自顶向下累计 λ<thresh 的高水位区 ⇒ E_star。"""
    acc = 0.0
    for i in range(len(lam) - 1, -1, -1):
        if lam[i] < thresh:
            acc += E_GRID[i + 1] - E_GRID[i]
        else:
            break
    return float(E_GRID[-1] - acc)


def estar_curve(V):
    """由完整 V（T+1, G）导出逐时目标水位 E_star(t)，t=0..T-1。"""
    out = np.empty(T)
    for t in range(T):
        lam = lambda_from_V(V[t + 1])
        out[t] = estar_from_lambda(lam, MULT * price_g[t] * ETA_D)
    return out


def estimate_value(price, scen, n_phi=NPHI):
    """在给定场景池（S,T）上做 FVI，返回 (V, E_star 曲线, Phi)。
    V 为成本-到-去（随 E 递减、凸）；Phi(E)=V[0](E) 为跨日终端成本。"""
    global price_g
    price_g = price
    S = scen.shape[0]
    b_ref = solve_lp(price, np.median(scen, axis=0), 6000.0,
                     mode="plan", terminal="geq", e_end_min=2400.0)["b"]
    Eg = E_GRID[None, :]                                   # (1,G)
    Phi = float(price.mean()) * np.maximum(E_GRID[-1] - E_GRID, 0.0)  # 递减初始化
    V = None
    for _ in range(n_phi):
        V = np.zeros((T + 1, len(E_GRID)))
        V[T] = Phi
        for t in range(T - 1, -1, -1):
            bt = b_ref[t]
            gap = np.maximum(scen[:, t] - bt, 0.0)[:, None]        # (S,1)
            sur = np.maximum(bt - scen[:, t], 0.0)[:, None]
            lam = lambda_from_V(V[t + 1])
            Estar = estar_from_lambda(lam, MULT * price[t] * ETA_D)
            d = np.minimum(np.minimum(gap, P_MAX), ETA_D * np.maximum(Eg - Estar, 0.0))
            e = gap - d
            Eafter_d = Eg - d / ETA_D
            cost_gap = MULT * price[t] * e + interp_grid(Eafter_d, V[t + 1])
            c = np.minimum(np.minimum(sur, P_MAX), (E_MAX - Eg) / ETA_C)
            Eafter_c = Eg + ETA_C * c
            cost_sur = interp_grid(Eafter_c, V[t + 1])
            is_gap = (scen[:, t] > bt)[:, None]
            V[t] = np.where(is_gap, cost_gap, cost_sur).mean(axis=0)
        Phi = V[0].copy()
    Estar = estar_curve(V)
    return V, Estar, Phi


# ============================ 价值反馈执行（因果、非前瞻） ============================
def execute_value(b, net_act, price, E_start, Estar):
    """放电 = min(缺口, 功率, η_d·max(E − E_star(t),0))；充电 = 余量尽量充。
    E_star≡E_MIN 时退化为 must/max（自检用）。"""
    c = np.zeros(T); d = np.zeros(T); e = np.zeros(T); w = np.zeros(T)
    E = np.empty(T + 1); E[0] = E_start
    for t in range(T):
        netb = b[t] - net_act[t]
        if netb >= 0:
            avail = (E_MAX - E[t]) / ETA_C
            c[t] = min(netb, P_MAX, avail); w[t] = netb - c[t]
        else:
            gap = -netb
            d[t] = min(gap, P_MAX, ETA_D * max(E[t] - Estar[t], 0.0))
            e[t] = gap - d[t]
        E[t + 1] = E[t] + ETA_C * c[t] - d[t] / ETA_D
    return c, d, e, w, E


# ============================ 条件场景池 ============================
def _ctx(net, dd):
    s = max(0, dd - 3)
    return float(net[s:dd].sum() / max(1, dd - s))


def scenario_pool(d, D, lookback=POOL_LOOKBACK, K=POOL_K, rho=RHO):
    net = D["net"]
    lo = max(0, d - lookback)
    days = np.arange(lo, d)
    if len(days) == 0:
        return D["net"][d:d + 1], np.array([1.0])
    age = (d - 1) - days
    cd = _ctx(net, d)
    sim = np.abs(np.array([_ctx(net, k) for k in days]) - cd)
    scale = np.median(sim) + 1e-9
    w = (rho ** age) * np.exp(-sim / scale)
    top = np.argsort(-w)[:K]
    days = days[top]; w = w[top]; w = w / w.sum()
    return net[days], w


def fvi_pool(m_start, D, lookback=FVI_LOOKBACK):
    net = D["net"]
    lo = max(0, m_start - lookback)
    return net[lo:m_start]


# ============================ 非前瞻采购（分块坐标下降 + 候选堆叠） ============================
def sim_cost(Bcand, E0, scen, wsc, price, Estar, Phi):
    """对 J 个候选计划，在 S 个场景上前推同一策略类，返回 J 个总成本。
    目标 = Σp·b + E_s[5Σp·e + Φ(E_144)]。向量化 over (J,S)。"""
    J = Bcand.shape[0]; S = scen.shape[0]
    E = np.full((J, S), float(E0))
    em = np.zeros((J, S))
    for t in range(T):
        netb = Bcand[:, t][:, None] - scen[:, t][None, :]          # (J,S)
        sur = np.maximum(netb, 0.0); gap = np.maximum(-netb, 0.0)
        c = np.minimum(np.minimum(sur, P_MAX), (E_MAX - E) / ETA_C)
        d = np.minimum(np.minimum(gap, P_MAX), ETA_D * np.maximum(E - Estar[t], 0.0))
        e = gap - d
        E = E + ETA_C * c - d / ETA_D
        em += MULT * price[t] * e
    term = interp_grid(E, Phi)
    scen_cost = em + term                                   # (J,S)
    exp_scen = (scen_cost * wsc[None, :]).sum(axis=1)       # (J,)
    plan = (Bcand * price[None, :]).sum(axis=1)             # (J,)
    return plan + exp_scen


def procure(b0, E0, scen, wsc, price, Estar, Phi, blocks=12,
            scales=(0.85, 0.925, 1.0, 1.075, 1.15), sweeps=2):
    """非前瞻采购：分块坐标下降。储能策略类由 Estar 决定（=E_MIN 即 must，=V 曲线即价值反馈）。
    起始 b0（用 V5 合同作起点，保证落在好盆地）。不证明全局最优（同 §19 的求解质量限制）。"""
    b = b0.copy()
    edges = np.linspace(0, T, blocks + 1).astype(int)
    for _ in range(sweeps):
        for bi in range(blocks):
            t0, t1 = edges[bi], edges[bi + 1]
            J = len(scales)
            Bcand = np.tile(b, (J, 1))
            for j, sc in enumerate(scales):
                Bcand[j, t0:t1] = b[t0:t1] * sc
            costs = sim_cost(Bcand, E0, scen, wsc, price, Estar, Phi)
            b = Bcand[int(np.argmin(costs))]
    return b


# ============================ V5 基线复现（取 b 序列与对拍） ============================
def _parse_v5_sched():
    """从 §23 的 q2_grouped_v4.json 解析逐月分组日程 → {月: ("G", vec4, W)}。
    标签形如 '分组 [早峰=0.80,日间=0.70,晚峰=0.90,其他=0.70], W=7'，顺序 = GNAMES = 组 0..3。"""
    import re
    js = json.loads((BASE / "results" / "q2_grouped_v4.json").read_text(encoding="utf-8"))
    sched = {}
    for m, lab in js["armB"]["sched"].items():
        nums = re.findall(r"=([0-9.]+)", lab)          # 4 个 β + 最后的 W
        betas = tuple(float(x) for x in nums[:4])
        W = int(float(nums[4]))
        sched[int(m)] = ("G", betas, W)
    return sched


def reproduce_v5(D, months):
    """取 V5 的逐日购电序列 B 与对拍总费用。优先解析 §23 已核验日程（快）；
    解析失败则回退到重跑 arm B 因果选择（慢）。"""
    try:
        sched = _parse_v5_sched()
        r = g4.replay(D, lambda d: sched[months[d]], WARMUP_DAY, E_FEB1, keep=True)
        tot = float((r["dp"][np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]]
                     + r["de"][np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]]).sum()) / 1e4
        if abs(tot - V5_TOTAL) < 0.01:
            return sched, r
        print(f"    [warn] 解析日程回放 = {tot:.4f}，与 {V5_TOTAL} 不符，回退到重选")
    except Exception as ex:  # noqa: BLE001
        print(f"    [warn] 日程解析失败（{ex}），回退到重选")
    cum = {}
    for k, c_ in enumerate(g4.CAND_B):
        r = g4.replay(D, lambda d, c_=c_: c_, 0, E_INIT)
        cum[k] = np.cumsum(r["dp"] + r["de"])
    month_end = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    sched = {m: min(range(len(g4.CAND_B)), key=lambda i: cum[i][month_end[m - 1]])
             for m in range(2, 13)}
    r = g4.replay(D, lambda d: g4.CAND_B[sched[months[d]]], WARMUP_DAY, E_FEB1, keep=True)
    return sched, r


# ============================ 消融回放 ============================
def replay_p2_fixed_contract(D, idx, V5B, Estar_of):
    """P2：固定 V5 购电序列，只换价值反馈执行。从 2/1 E=10800 连续回放。"""
    price, net = D["price"], D["net"]
    months = D["_month"]
    E = float(E_FEB1)
    dp = np.zeros(len(D["dates"])); de = np.zeros(len(D["dates"]))
    dk = np.zeros(len(D["dates"])); soc0 = np.zeros(len(D["dates"])); soc1 = np.zeros(len(D["dates"]))
    socmin = np.zeros(len(D["dates"]))
    for d in range(WARMUP_DAY, len(D["dates"])):
        Estar = Estar_of[months[d]]
        c, dd, e, w, Etraj = execute_value(V5B[d], net[d], price, E, Estar)
        dp[d] = float((V5B[d] * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); soc0[d] = E; soc1[d] = Etraj[-1]; socmin[d] = Etraj.min()
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, soc0=soc0, soc1=soc1, socmin=socmin)


def replay_p34(D, idx, V5B, Estar_of, Phi_of, use_value):
    """P3/P4：每日 0:00 用条件场景 + Φ 重订 b（坐标下降），再用同一策略类执行。
    use_value=False → 采购与执行都 must/max（P3）；True → 都价值反馈（P4）。"""
    price, net = D["price"], D["net"]
    months = D["_month"]
    E = float(E_FEB1)
    nd = len(D["dates"])
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd)
    soc0 = np.zeros(nd); soc1 = np.zeros(nd); socmin = np.zeros(nd)
    t0 = time.perf_counter()
    for d in range(WARMUP_DAY, nd):
        m = months[d]
        scen, wsc = scenario_pool(d, D)
        Estar = Estar_of[m] if use_value else np.full(T, E_MIN)
        Phi = Phi_of[m]
        b = procure(V5B[d], E, scen, wsc, price, Estar, Phi)
        c, dd, e, w, Etraj = execute_value(b, net[d], price, E, Estar)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); soc0[d] = E; soc1[d] = Etraj[-1]; socmin[d] = Etraj.min()
        E = Etraj[-1]
        if (d - WARMUP_DAY) % 40 == 0:
            print(f"    d={d} ({str(D['dates'][d])[:10]}) 累计 {time.perf_counter()-t0:.0f}s",
                  flush=True)
    return dict(dp=dp, de=de, dk=dk, soc0=soc0, soc1=soc1, socmin=socmin)


def summarize(name, r, idx, price):
    p = float(r["dp"][idx].sum()); e = float(r["de"][idx].sum())
    dE = max(0.0, E_FEB1 - float(r["soc1"][idx][-1]))
    return dict(name=name, plan=p / 1e4, em=e / 1e4, total=(p + e) / 1e4,
                emk=float(r["dk"][idx].sum()),
                大紧急天数=int((r["dk"][idx] > 1000).sum()),
                P99=float(np.quantile(r["de"][idx] / 1e4, 0.99)),
                日均起始SOC=float(r["soc0"][idx].mean()),
                触底天数=int((r["socmin"][idx] <= E_MIN + 1e-3).sum()),
                年末SOC=float(r["soc1"][idx][-1]), 库存缺口=dE,
                折算中=(p + e) / 1e4 + float(price.mean()) * dE / ETA_C / 1e4)


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    month_start = {m: int(np.min(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]
    quick = "--quick" in sys.argv
    if quick:
        idx = idx[(months[idx] == 2) | (months[idx] == 8)]

    print("=" * 104)
    print("§24 结构性挑战：库存价值函数 + 条件场景 + 非前瞻采购/储能联合优化")
    print("=" * 104)
    print(f"  评价期 {str(dates[idx[0]])[:10]} ~ {str(dates[idx[-1]])[:10]}（{len(idx)} 天）"
          f"{' [QUICK]' if quick else ''}；公共预热 E = {E_FEB1:.0f} kWh")

    # ---- 0. 逐月价值函数（因果：第 m 月只用该月之前的数据）----
    print("\n[0] 逐月库存价值函数（FVI；场景=该月之前最近 ≤60 天）")
    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        scen = fvi_pool(month_start[m], D)
        V, Estar, Phi = estimate_value(price, scen)
        Estar_of[m] = Estar; Phi_of[m] = Phi
        lam_hi = estar_curve_hi = None
        print(f"    {m:>2} 月  E_star: min {Estar.min():7.0f}  max {Estar.max():7.0f}"
              f"  均值 {Estar.mean():7.0f} kWh   Φ(E_MIN)={interp_grid(E_MIN, Phi):.1f}"
              f"  Φ(E_MAX)={interp_grid(E_MAX, Phi):.1f}")
    mshow = 8
    print(f"    [示例 {mshow} 月逐时 E_star（kWh），每 2h 取一点]")
    Es = Estar_of[mshow]
    print("    " + "  ".join(f"{10*t//60:02d}h:{Es[t]:5.0f}" for t in range(0, T, 12)))

    # ---- 1. 复现 V5 并取其购电序列 ----
    print("\n[1] 复现 V5（§23 arm B 因果日程），取购电序列 b")
    sched, rV5 = reproduce_v5(D, months)
    V5B = rV5["det"]["B"]
    cV5 = float((rV5["dp"][idx] + rV5["de"][idx]).sum()) / 1e4
    print(f"    V5 对拍 = {cV5:.4f} 万元（§23 公布 {V5_TOTAL}，差 {cV5 - V5_TOTAL:+.4f}）")
    # 统一成自定义回放格式（供 summarize）
    SOC = rV5["det"]["SOC"]
    rV5 = dict(dp=rV5["dp"], de=rV5["de"], dk=rV5["dk"],
               soc0=SOC[:, 0], soc1=SOC[:, -1], socmin=SOC.min(axis=1))

    # ---- 自检：价值反馈执行器在 E_star≡E_MIN 时必须复现 must/max 的 V5 ----
    Emin_of = {m: np.full(T, E_MIN) for m in range(2, 13)}
    rself = replay_p2_fixed_contract(D, idx, V5B, Emin_of)
    cself = float((rself["dp"][idx] + rself["de"][idx]).sum()) / 1e4
    print(f"    [自检] 价值反馈(E_star≡E_MIN) 复现 must/max V5 = {cself:.4f}"
          f"（差 {cself - cV5:+.2e}，应≈0）")

    # ---- 2. P2 固定合同 + 价值反馈 ----
    print("\n[2] P2 固定 V5 合同 + 价值反馈执行（隔离'主动留电'的价值）")
    rP2 = replay_p2_fixed_contract(D, idx, V5B, Estar_of)
    cP2 = float((rP2["dp"][idx] + rP2["de"][idx]).sum()) / 1e4
    print(f"    P2 = {cP2:.4f} 万元   相对 V5 {cP2 - cV5:+.4f}")

    # ---- 3. P3 新采购 + 尽限 ----
    print("\n[3] P3 条件场景新采购 + must/max 执行（隔离'采购结构'）")
    rP3 = replay_p34(D, idx, V5B, Estar_of, Phi_of, use_value=False)
    cP3 = float((rP3["dp"][idx] + rP3["de"][idx]).sum()) / 1e4
    print(f"    P3 = {cP3:.4f} 万元   相对 V5 {cP3 - cV5:+.4f}")

    # ---- 4. P4 新采购 + 价值反馈 ----
    print("\n[4] P4 条件场景新采购 + 价值反馈执行（联合）")
    rP4 = replay_p34(D, idx, V5B, Estar_of, Phi_of, use_value=True)
    cP4 = float((rP4["dp"][idx] + rP4["de"][idx]).sum()) / 1e4
    print(f"    P4 = {cP4:.4f} 万元   相对 V5 {cP4 - cV5:+.4f}")

    # ---- 汇总 ----
    print("\n" + "=" * 104)
    print("消融汇总（评价期，万元）")
    print("=" * 104)
    rows = [summarize("V5 基线", rV5, idx, price),
            summarize("P2 固定合同+价值反馈", rP2, idx, price),
            summarize("P3 新采购+尽限", rP3, idx, price),
            summarize("P4 新采购+价值反馈", rP4, idx, price)]
    print(f"  {'方案':<22}{'计划':>10}{'紧急':>10}{'总费用':>11}{'相对V5':>10}"
          f"{'P99':>8}{'大紧急':>7}{'起始SOC':>9}{'触底':>6}{'折算中':>10}")
    for r in rows:
        print(f"  {r['name']:<22}{r['plan']:>10.4f}{r['em']:>10.4f}{r['total']:>11.4f}"
              f"{r['total']-cV5:>+10.4f}{r['P99']:>8.3f}{r['大紧急天数']:>7d}"
              f"{r['日均起始SOC']:>9.0f}{r['触底天数']:>6d}{r['折算中']:>10.4f}")
    # 协同 = (P2-V5) + (P3-V5) - (P4-V5)
    syn = (cP2 - cV5) + (cP3 - cV5) - (cP4 - cV5)
    print(f"\n  单独效应：价值反馈 {cP2-cV5:+.4f}｜采购 {cP3-cV5:+.4f}｜联合 {cP4-cV5:+.4f}"
          f"｜交互(协同) {syn:+.4f}（>0 表示联合比各自之和更省）")

    out = dict(口径=dict(G_STEP=G_STEP, NPHI=NPHI, POOL_K=POOL_K, RHO=RHO,
                        quick=quick, 评价期天数=int(len(idx))),
               V5=cV5, 自检must复现=cself,
               P2=cP2, P3=cP3, P4=cP4, 协同=syn,
               detail={r["name"]: r for r in rows})
    tag = "_quick" if quick else ""
    (BASE / "results" / f"q2_value_adp{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_value_adp{tag}.json   总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()