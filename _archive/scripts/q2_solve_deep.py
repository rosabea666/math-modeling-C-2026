# -*- coding: utf-8 -*-
"""§29 求解强度对照（只读）：判别 CD 是否解到位。
同一因果政策、同一场景、同一初始状态，只改变"优化强度"：
  S0 原 CD（blocks=12, 5 scales, sweeps=2，起点 V5B）
  S1 渐进细化 CD（12 -> 48 -> 144 块）
  S2 S1 + 多起点（V5B / 0.90V5B / 1.10V5B / LP 的 b*）
判据：
  场景内：B 能压到多低（相对 A=2041.60 的 anticipative 下界）⇒ 求解是否不足
  实测：  各自 SOC 连续回放，压低后变好=求解为主因；变坏=过拟合、场景为主因
"""
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

import numpy as np  # noqa: E402
import q2_value_adp as va  # noqa: E402
import q2_saa_lp as sa  # noqa: E402
from q2_model import (T, MULT, E_FEB1, WARMUP_DAY, load_data)  # noqa: E402

V6_TOTAL = 1581.6655
SCALES5 = (0.85, 0.925, 1.0, 1.075, 1.15)
SCALES7 = (0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20)


def procure_gen(b0, E0, scen, wsc, price, Estar, Phi,
                blocks_seq=(12,), scales=SCALES5, sweeps=2):
    """通用分块坐标下降；blocks_seq 支持由粗到细渐进。"""
    b = b0.copy()
    for blocks in blocks_seq:
        edges = np.linspace(0, T, blocks + 1).astype(int)
        for _ in range(sweeps):
            for bi in range(blocks):
                t0, t1 = edges[bi], edges[bi + 1]
                if t1 <= t0:
                    continue
                Bcand = np.tile(b, (len(scales), 1))
                for j, s in enumerate(scales):
                    Bcand[j, t0:t1] = b[t0:t1] * s
                cost = va.sim_cost(Bcand, E0, scen, wsc, price, Estar, Phi)
                b = Bcand[int(np.argmin(cost))]
    return b


def day_solve(d, E0, scen, wsc, price, Estar, Phi, V5B, cache, arms):
    """返回 {arm: (b, 场景费用)}。"""
    out = {}
    bLP = None
    if "S2" in arms:
        bLP, _ = sa.solve_day(price, scen, wsc, E0, Phi, cache)
    # S0
    if "S0" in arms:
        b = va.procure(V5B[d], E0, scen, wsc, price, Estar, Phi)
        out["S0"] = (b, float(va.sim_cost(b[None, :], E0, scen, wsc, price, Estar, Phi)[0]))
    # S1
    if "S1" in arms:
        b = procure_gen(V5B[d], E0, scen, wsc, price, Estar, Phi,
                        blocks_seq=(12, 48, 144), scales=SCALES7, sweeps=2)
        out["S1"] = (b, float(va.sim_cost(b[None, :], E0, scen, wsc, price, Estar, Phi)[0]))
    # S2：多起点 + S1 的细化
    if "S2" in arms:
        best_b, best_c = None, np.inf
        for s0 in (V5B[d], V5B[d] * 0.90, V5B[d] * 1.10, bLP):
            bb = procure_gen(s0, E0, scen, wsc, price, Estar, Phi,
                             blocks_seq=(12, 48, 144), scales=SCALES7, sweeps=2)
            cc = float(va.sim_cost(bb[None, :], E0, scen, wsc, price, Estar, Phi)[0])
            if cc < best_c:
                best_b, best_c = bb, cc
        out["S2"] = (best_b, best_c)
    return out


def replay_continuous(arm_solver, D, Estar_of, Phi_of, idx, V5B, quick=None):
    """各 arm 自身 SOC 连续回放。arm_solver(d, E0, scen, wsc, Estar, Phi) -> b。"""
    price, dates, net = D["price"], D["dates"], D["net"]
    months = D["_month"]
    E = float(E_FEB1)
    tot = np.zeros(len(dates))
    days = range(WARMUP_DAY, len(dates))
    if quick:
        days = list(days)[:quick]
    cache = {}
    for d in days:
        m = months[d]
        Estar, Phi = Estar_of[m], Phi_of[m]
        scen, wsc = va.scenario_pool(d, D)
        b = arm_solver(d, E, scen, wsc, price, Estar, Phi, cache)
        _, _, e, _, Etraj = va.execute_value(b, net[d], price, E, Estar)
        tot[d] = float((b * price).sum() + MULT * (e * price).sum())
        E = Etraj[-1]
    return tot[idx].sum() / 1e4, E


def main(quick=None, arms=("S0", "S1", "S2")):
    t_all = time.perf_counter()
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([dt.astype("datetime64[M]").astype(int) % 12 + 1 for dt in dates])
    D["_month"] = months
    month_start = {m: int(np.min(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§29 求解强度对照：判别 CD 是否解到位（同一因果政策，只改优化强度）")
    print("=" * 104)
    print(f"  arms = {arms}" + (f"；quick 模式（仅前 {quick} 天）" if quick else ""))

    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        sc_ = va.fvi_pool(month_start[m], D)
        _, Estar, Phi = va.estimate_value(price, sc_)
        Estar_of[m], Phi_of[m] = Estar, Phi
    _, rV5 = va.reproduce_v5(D, months)
    V5B = rV5["det"]["B"]

    eval_days = list(range(WARMUP_DAY, len(dates)))
    if quick:
        eval_days = eval_days[:quick]

    # ---------- 场景内：同一 V6 起点，比较各 arm 的场景费用 ----------
    print("\n[场景内] 同一 V6 实际起点轨迹、同一场景、同一 Φ")
    soc0 = np.zeros(len(dates))
    Eprev = float(E_FEB1)
    for d in range(WARMUP_DAY, len(dates)):
        soc0[d] = Eprev
        _, _, _, _, Etraj = va.execute_value(V5B[d], net[d], price, Eprev, Estar_of[months[d]])
        Eprev = Etraj[-1]

    cache = {}
    scen_cost = {a: np.zeros(len(dates)) for a in arms}
    A = np.zeros(len(dates))
    C = np.zeros(len(dates))
    t0 = time.perf_counter()
    for i, d in enumerate(eval_days):
        m = months[d]
        E0 = float(soc0[d])
        scen, wsc = va.scenario_pool(d, D)
        Phi, Estar = Phi_of[m], Estar_of[m]
        bLB, A[d] = sa.solve_day(price, scen, wsc, E0, Phi, cache)
        C[d] = float(va.sim_cost(V5B[d][None, :], E0, scen, wsc, price, Estar, Phi)[0])
        res = day_solve(d, E0, scen, wsc, price, Estar, Phi, V5B, cache, arms)
        for a in arms:
            scen_cost[a][d] = res[a][1]
        if i % 20 == 0:
            print(f"    d={d} 累计 {time.perf_counter()-t0:.0f}s", flush=True)

    sub = np.array(eval_days)
    print(f"\n  {'arm':<6}{'场景费用':>12}{'相对S0':>10}{'距下界A':>10}")
    sA, sC = A[sub].sum() / 1e4, C[sub].sum() / 1e4
    base = None
    for a in arms:
        s = scen_cost[a][sub].sum() / 1e4
        if base is None:
            base = s
        print(f"  {a:<6}{s:>12.4f}{s-base:>+10.4f}{s-sA:>+10.4f}")
    print(f"  {'A(下界)':<6}{sA:>12.4f}")
    print(f"  {'C(V6合同)':<6}{sC:>12.4f}")

    out = dict(口径=dict(arms=list(arms), 评价天数=len(sub), quick=quick),
               场景内={a: float(scen_cost[a][sub].sum() / 1e4) for a in arms},
               场景内_A下界=float(sA), 场景内_C_V6合同=float(sC))

    # ---------- 实测：各 arm 自身 SOC 连续回放 ----------
    print("\n[实测] 各 arm 自身 SOC 连续回放（不借 V6 起点）")

    def mk(a):
        def f(d, E0, scen, wsc, price, Estar, Phi, cache):
            return day_solve(d, E0, scen, wsc, price, Estar, Phi, V5B, cache, (a,))[a][0]
        return f

    replay = {}
    for a in arms:
        t1 = time.perf_counter()
        tt, ee = replay_continuous(mk(a), D, Estar_of, Phi_of, idx, V5B, quick=quick)
        replay[a] = dict(总费=tt, 年末SOC=float(ee))
        print(f"    {a} 连续回放 = {tt:9.4f}（相对 V6 {tt-V6_TOTAL:+.4f}）  年末SOC {ee:.0f}"
              f"  [{time.perf_counter()-t1:.0f}s]", flush=True)
    out["实测连续回放"] = replay

    (BASE / "results" / ("q2_solve_deep_quick.json" if quick else "q2_solve_deep.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘   总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    q = None
    aa = ("S0", "S1", "S2")
    for arg in sys.argv[1:]:
        if arg.isdigit():
            q = int(arg)
        else:
            aa = tuple(arg.split(","))
    main(quick=q, arms=aa)
