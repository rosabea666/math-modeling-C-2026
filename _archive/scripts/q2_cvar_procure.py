# -*- coding: utf-8 -*-
r"""
§27b 机理收口：风险厌恶（CVaR）能否让场景采购兑现 §26 的 74~84 万余量？（只读，不改模型）
================================================================================
§26 证明采购合同在场景上确有 74~84 万余量、但期望成本采购迁移比 −1.05；§27 证明三种场景构造
（含尾部增强）迁移比仍全为负，且瓶颈是**峰值上尾覆盖**（≈0.50）——期望成本对重尾（5 倍价晚峰）
欠保护。本脚本检验最后一个候选：**把目标从"期望成本"换成"CVaR 风险厌恶"**（聚焦最坏尾场景，
与 V6 的上尾分位哲学同源）。若 CVaR 能让迁移比回到 +1 附近 ⇒ 采购确有可兑现空间（只需换目标）；
若仍为负 ⇒ 场景-SAA 这条路在本数据上对日前采购**基本走不通**，分位数计划已是"对采购有用的预测"。

CVaR_β 取 β=0.90（与 V6 的 β 同档，**非在评价期调参**）；求解仍用同一分块坐标下降。
对照：G3 因果残差、G4 尾部增强残差 × {期望成本, CVaR(0.90)}。
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_MIN, E_MAX, P_MAX, ETA_C, ETA_D, E_FEB1, WARMUP_DAY, load_data)  # noqa: E402
import q2_value_adp as va  # noqa: E402
import q2_scenario_quality as sq  # noqa: E402

V6_TOTAL = 1581.6655
BETA_CVAR = 0.90


def scen_cost_matrix(Bcand, E0, scen, price, Estar, Phi):
    """返回 (plan(J,), scen_cost(J,S))，不加权（供 CVaR 用）。"""
    J = Bcand.shape[0]; S = scen.shape[0]
    E = np.full((J, S), float(E0)); em = np.zeros((J, S))
    for t in range(T):
        netb = Bcand[:, t][:, None] - scen[:, t][None, :]
        sur = np.maximum(netb, 0.0); gap = np.maximum(-netb, 0.0)
        c = np.minimum(np.minimum(sur, P_MAX), (E_MAX - E) / ETA_C)
        d = np.minimum(np.minimum(gap, P_MAX), ETA_D * np.maximum(E - Estar[t], 0.0))
        e = gap - d
        E = E + ETA_C * c - d / ETA_D
        em += MULT * price[t] * e
    term = va.interp_grid(E, Phi)
    plan = (Bcand * price[None, :]).sum(axis=1)
    return plan, em + term


def cvar_obj(plan, sc, wsc, beta):
    J = sc.shape[0]; out = np.empty(J)
    for j in range(J):
        order = np.argsort(sc[j]); cs = sc[j][order]; ws = wsc[order]
        cw = np.cumsum(ws); k = int(np.searchsorted(cw, beta))
        tail, tw = cs[k:], ws[k:]
        out[j] = plan[j] + (tail * tw).sum() / max(tw.sum(), 1e-12)
    return out


def procure_cvar(b0, E0, scen, wsc, price, Estar, Phi, beta=BETA_CVAR,
                 blocks=12, scales=(0.85, 0.925, 1.0, 1.075, 1.15), sweeps=2):
    b = b0.copy(); edges = np.linspace(0, T, blocks + 1).astype(int)
    for _ in range(sweeps):
        for bi in range(blocks):
            t0, t1 = edges[bi], edges[bi + 1]
            Jn = len(scales); Bcand = np.tile(b, (Jn, 1))
            for j, s in enumerate(scales):
                Bcand[j, t0:t1] = b[t0:t1] * s
            plan, sc = scen_cost_matrix(Bcand, E0, scen, price, Estar, Phi)
            obj = cvar_obj(plan, sc, wsc, beta)
            b = Bcand[int(np.argmin(obj))]
    return b


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    month_start = {m: int(np.min(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]
    pre = sq.precompute(D)

    print("=" * 104)
    print("§27b CVaR 风险厌恶能否兑现采购余量（β=0.90，与 V6 同档）")
    print("=" * 104)
    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        scen = va.fvi_pool(month_start[m], D)
        _, Estar, Phi = va.estimate_value(price, scen)
        Estar_of[m], Phi_of[m] = Estar, Phi
    sched, rV5 = va.reproduce_v5(D, months)
    V5B = rV5["det"]["B"]
    soc0 = np.zeros(len(dates)); actV6 = np.zeros(len(dates))
    E = float(E_FEB1)
    for d in range(WARMUP_DAY, len(dates)):
        Estar = Estar_of[months[d]]
        _, _, e, _, Etraj = va.execute_value(V5B[d], net[d], price, E, Estar)
        soc0[d] = E; actV6[d] = float((V5B[d] * price).sum() + MULT * (e * price).sum())
        E = Etraj[-1]
    V6 = actV6[idx].sum()
    print(f"  V6 基准 = {V6/1e4:.4f} 万元")

    GENS = {"G3因果残差": sq.gen_residual, "G4尾部增强残差": sq.gen_tail}
    rows = {}
    for gname, gen in GENS.items():
        for mode in ("期望成本", "CVaR0.90"):
            aG = np.zeros(len(dates))
            t0 = time.perf_counter()
            for d in range(WARMUP_DAY, len(dates)):
                m = months[d]; E0 = soc0[d]
                scen, wsc = gen(d, D, pre)
                Phi, Estar = Phi_of[m], Estar_of[m]
                if mode == "期望成本":
                    bG = va.procure(V5B[d], E0, scen, wsc, price, Estar, Phi)
                else:
                    bG = procure_cvar(V5B[d], E0, scen, wsc, price, Estar, Phi)
                _, _, e, _, _ = va.execute_value(bG, net[d], price, E0, Estar)
                aG[d] = float((bG * price).sum() + MULT * (e * price).sum())
                if (d - WARMUP_DAY) % 90 == 0:
                    print(f"    [{gname}/{mode}] d={d} 累计 {time.perf_counter()-t0:.0f}s", flush=True)
            A = aG[idx].sum()
            rows[f"{gname}/{mode}"] = A / 1e4
            print(f"  {gname:<14}{mode:<10} 实测总费 {A/1e4:9.4f}  相对 V6 {(A-V6)/1e4:+9.4f}")

    print("\n" + "=" * 104)
    print("CVaR 对照（相对 V6 越负越好）")
    print("=" * 104)
    print(f"  {'构造/目标':<24}{'实测总费/万元':>14}{'相对 V6':>12}")
    for k, v in rows.items():
        print(f"  {k:<24}{v:>14.4f}{v - V6_TOTAL:>+12.4f}")
    print(f"\n  参照 V6 = {V6_TOTAL:.4f}")
    out = dict(口径=dict(BETA_CVAR=BETA_CVAR, 评价期天数=int(len(idx)), V6=V6_TOTAL), 结果=rows)
    (BASE / "results" / "q2_cvar_procure.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_cvar_procure.json   总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
