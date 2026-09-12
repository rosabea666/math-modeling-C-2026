# -*- coding: utf-8 -*-
"""§28 诊断修正（只读，不改模型）：
1) 覆盖指标重定义：包络超出率 / 加权 PIT 校准 / 超出量与其费用 / 峰值幅值-时刻误差
2) CVaR 离散权重修正（标准部分概率质量式 + Rockafellar-Uryasev 自检）
3) 求解残差的"同合同"合法分解（证伪"政策差距为常数"，给出合法上界）
4) 各候选自身 SOC 连续回放（替代逐日反事实）
5) newsvendor 检验：期望成本最优是否已给出上分位采购
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
import q2_scenario_quality as sq  # noqa: E402
import q2_saa_lp as sa  # noqa: E402
from q2_model import (T, MULT, E_MIN, E_MAX, P_MAX, ETA_C, ETA_D,  # noqa: E402
                      E_FEB1, WARMUP_DAY, load_data)

V6_TOTAL = 1581.6655


# ==================== Part 1：覆盖指标重定义 ====================
def pit_of(scen, wsc, act):
    """加权 PIT：实际值在加权场景经验分布中的分位水平。
    校准良好的预测分布 ⇒ PIT ≈ U(0,1)。"""
    out = np.empty(T)
    for t in range(T):
        o = np.argsort(scen[:, t])
        xs = scen[o, t]
        cw = np.cumsum(wsc[o])
        # 节点：(x_1, 0), (x_1, F_1), ..., (x_S, F_S)；右侧超出取 1
        xx = np.concatenate([[xs[0]], xs])
        ff = np.concatenate([[0.0], cw])
        out[t] = float(np.clip(np.interp(act[t], xx, ff), 0.0, 1.0))
    return out


def ks_uniform(u):
    u = np.sort(np.asarray(u))
    n = len(u)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def coverage_v2(scen, wsc, act, price, hi):
    smax = scen.max(axis=0)
    smean = (scen * wsc[:, None]).sum(axis=0)
    exc = np.maximum(act - smax, 0.0)                # 超出场景包络的量
    r = dict(
        env_exc_all=float(np.mean(act > smax)),                     # 包络超出率（全天）
        env_exc_hi=float(np.mean(act[hi] > smax[hi])) if hi.sum() else np.nan,  # 高价窗口
        exc_kwh_all=float(exc.sum()),                               # 包络外电量 kWh/天
        exc_kwh_hi=float(exc[hi].sum()),
        exc_cost_hi=float(MULT * (price[hi] * exc[hi]).sum()),      # 5倍价费用上界（储能零贡献假设）
        pit_mean=float(np.nanmean(pit_of(scen, wsc, act))),         # 旧"晚峰覆盖"的真含义
        pit_exc90=float(np.mean(pit_of(scen, wsc, act) > 0.90)),    # 理想 0.10
        pit_exc50=float(np.mean(pit_of(scen, wsc, act) > 0.50)),    # 理想 0.50
    )
    pit = pit_of(scen, wsc, act)
    r["pit_ks"] = ks_uniform(pit)
    r["pit_hi_mean"] = float(pit[hi].mean()) if hi.sum() else np.nan
    # 峰值：幅值与时刻（形状相关高不代表峰值对）
    r["peak_amp_err"] = float(act.max() - smean.max())
    r["peak_time_err"] = float(abs(int(np.argmax(act)) - int(np.argmax(smean))))
    r["old_peak_cov"] = float(np.mean([(scen[:, t] >= act[t]).mean() for t in np.where(hi)[0]]))
    return r


# ==================== Part 2：CVaR 离散修正 ====================
def cvar_discrete(z, w, alpha):
    """标准离散 CVaR_alpha：分位点处只取部分概率质量，使尾部概率恰为 1-alpha。
    CVaR = [(F_k - alpha) z_k + sum_{i>k} w_i z_i] / (1 - alpha)，k = min{i: F_i >= alpha}。"""
    o = np.argsort(z)
    xs = np.asarray(z)[o]
    ws = np.asarray(w)[o]
    cw = np.cumsum(ws)
    k = int(np.searchsorted(cw, alpha, side="left"))
    k = min(k, len(xs) - 1)
    val = (cw[k] - alpha) * xs[k] + float(np.dot(ws[k + 1:], xs[k + 1:]))
    return val / (1.0 - alpha)


def cvar_ru(z, w, alpha):
    """Rockafellar-Uryasev：min_zeta zeta + 1/(1-alpha) sum w (z-zeta)^+（用于自检）。"""
    z = np.asarray(z)
    w = np.asarray(w)
    best = np.inf
    for zeta in np.unique(z):
        v = zeta + float(np.sum(w * np.maximum(z - zeta, 0.0))) / (1.0 - alpha)
        best = min(best, v)
    return best


def cvar_obj_new(plan, sc, wsc, alpha):
    return np.array([plan[j] + cvar_discrete(sc[j], wsc, alpha) for j in range(sc.shape[0])])


def procure_cvar_new(b0, E0, scen, wsc, price, Estar, Phi, alpha=0.90,
                     blocks=12, scales=(0.85, 0.925, 1.0, 1.075, 1.15), sweeps=2):
    b = b0.copy()
    edges = np.linspace(0, T, blocks + 1).astype(int)
    for _ in range(sweeps):
        for bi in range(blocks):
            t0, t1 = edges[bi], edges[bi + 1]
            Bcand = np.tile(b, (len(scales), 1))
            for j, s in enumerate(scales):
                Bcand[j, t0:t1] = b[t0:t1] * s
            plan, sc = sq_scen_cost(Bcand, E0, scen, price, Estar, Phi)
            obj = cvar_obj_new(plan, sc, wsc, alpha)
            b = Bcand[int(np.argmin(obj))]
    return b


def sq_scen_cost(Bcand, E0, scen, price, Estar, Phi):
    """(plan(J,), scen_cost(J,S))，不加权。"""
    J = Bcand.shape[0]
    E = np.full((J, scen.shape[0]), float(E0))
    em = np.zeros((J, scen.shape[0]))
    for t in range(T):
        netb = Bcand[:, t][:, None] - scen[:, t][None, :]
        sur = np.maximum(netb, 0.0)
        gap = np.maximum(-netb, 0.0)
        c = np.minimum(np.minimum(sur, P_MAX), (E_MAX - E) / ETA_C)
        d = np.minimum(np.minimum(gap, P_MAX), ETA_D * np.maximum(E - Estar[t], 0.0))
        e = gap - d
        E = E + ETA_C * c - d / ETA_D
        em += MULT * price[t] * e
    term = va.interp_grid(E, Phi)
    return (Bcand * price[None, :]).sum(axis=1), em + term


# ==================== Part 5：newsvendor 检验 ====================
def newsvendor_check(net, idx, W=7):
    """无储能单层：min_b p·b + 5p·E[(N-b)^+] ⇒ 内部最优满足 F(b*)=1-1/5=0.8。
    用前 W 天经验分布逐日求 b*，报告其经验分位水平。"""
    qs = []
    for d in idx:
        hist = net[max(0, d - W):d]
        if len(hist) < 3:
            continue
        x = np.sort(hist.ravel())
        n = len(x)
        grid = np.linspace(x[0], x[-1], 400)
        obj = grid + MULT * np.mean(np.maximum(x[None, :] - grid[:, None], 0.0), axis=1)
        bstar = grid[int(np.argmin(obj))]
        qs.append(float(np.mean(x <= bstar)))
    qs = np.array(qs)
    return dict(理论值=0.8, 实测均值=float(qs.mean()), 中位数=float(np.median(qs)),
                P10=float(np.quantile(qs, 0.10)), P90=float(np.quantile(qs, 0.90)),
                天数=int(len(qs)))


# ==================== 主流程 ====================
def main(parts=(1, 2, 3, 4, 5)):
    t_all = time.perf_counter()
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    month_start = {m: int(np.min(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]
    hi = price >= np.quantile(price, 0.90)

    print("=" * 108)
    print("§28 诊断修正（只读）：覆盖指标 / CVaR 离散权重 / 同合同求解残差 / 连续回放 / newsvendor")
    print("=" * 108)
    print(f"  评价期 {len(idx)} 天；高价窗口 {int(hi.sum())} 区间/天")

    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        sc_ = va.fvi_pool(month_start[m], D)
        _, Estar, Phi = va.estimate_value(price, sc_)
        Estar_of[m], Phi_of[m] = Estar, Phi
    sched, rV5 = va.reproduce_v5(D, months)
    V5B = rV5["det"]["B"]
    out = {}

    # ---------- Part 5：newsvendor（快，先跑） ----------
    if 5 in parts:
        print("\n[Part 5] newsvendor 检验：期望成本最优是否已给出上分位采购")
        nv = newsvendor_check(net, idx)
        out["Part5_newsvendor"] = nv
        print(f"  无储能单层 min p·b + 5p·E[(N-b)^+] ⇒ 理论 F(b*)={nv['理论值']}")
        print(f"  实测（前7天经验分布，{nv['天数']} 天）：均值 {nv['实测均值']:.4f}｜"
              f"中位 {nv['中位数']:.4f}｜P10 {nv['P10']:.4f}｜P90 {nv['P90']:.4f}")

    # ---------- Part 1：覆盖指标重定义 ----------
    if 1 in parts:
        print("\n[Part 1] 覆盖指标重定义（包络超出率 / 加权 PIT / 超出量 / 峰值误差）")
        pre = sq.precompute(D)
        res = {}
        for gname, gen in sq.GENS.items():
            acc = {k: [] for k in ("env_exc_all", "env_exc_hi", "exc_kwh_all", "exc_kwh_hi",
                                   "exc_cost_hi", "pit_mean", "pit_exc90", "pit_exc50",
                                   "pit_ks", "pit_hi_mean", "peak_amp_err", "peak_time_err",
                                   "old_peak_cov")}
            t0 = time.perf_counter()
            for d in idx:
                scen, wsc = gen(d, D, pre)
                r = coverage_v2(scen, wsc, net[d], price, hi)
                for k in acc:
                    acc[k].append(r[k])
            res[gname] = {k: float(np.nanmean(v)) for k, v in acc.items()}
            print(f"    [{gname}] {time.perf_counter()-t0:.0f}s", flush=True)
        out["Part1_coverage"] = res
        print(f"\n  {'构造':<14}{'旧指标':>8}{'PIT均值':>9}{'PIT>0.9':>9}{'PIT>0.5':>9}"
              f"{'KS':>8}{'包络超出':>10}{'高价超出':>10}{'包络外kWh':>11}{'高价外kWh':>11}"
              f"{'峰值幅差':>10}{'峰时差':>8}")
        for g, r in res.items():
            print(f"  {g:<14}{r['old_peak_cov']:>8.3f}{r['pit_mean']:>9.3f}{r['pit_exc90']:>9.3f}"
                  f"{r['pit_exc50']:>9.3f}{r['pit_ks']:>8.3f}{r['env_exc_all']:>10.3f}"
                  f"{r['env_exc_hi']:>10.3f}{r['exc_kwh_all']:>11.1f}{r['exc_kwh_hi']:>11.1f}"
                  f"{r['peak_amp_err']:>10.1f}{r['peak_time_err']:>8.2f}")
        print("  （PIT 校准良好 ⇒ PIT均值≈0.50、PIT>0.9≈0.10、KS 小；包络超出率 = 实测高于所有场景的比例）")

    # ---------- Part 3：同合同求解残差分解 ----------
    if 3 in parts:
        print("\n[Part 3] 求解残差的『同合同』合法分解（证伪政策差距为常数）")
        Acd = np.zeros(len(dates))
        Ac = np.zeros(len(dates))
        Bc_ = np.zeros(len(dates))
        Cc = np.zeros(len(dates))
        Al = np.zeros(len(dates))
        cache = {}
        t0 = time.perf_counter()
        # V6 实际起点轨迹（与 §26 同口径；不得用恒定初值，否则 A 不可比）
        soc0 = np.zeros(len(dates))
        Eprev = float(E_FEB1)
        for d in range(WARMUP_DAY, len(dates)):
            soc0[d] = Eprev
            _, _, _, _, Etraj = va.execute_value(V5B[d], net[d], price, Eprev,
                                                 Estar_of[months[d]])
            Eprev = Etraj[-1]
        for d in range(WARMUP_DAY, len(dates)):
            m = months[d]
            E0 = float(soc0[d])
            scen, wsc = va.scenario_pool(d, D)
            Phi, Estar = Phi_of[m], Estar_of[m]
            bLB, Al[d] = sa.solve_day(price, scen, wsc, E0, Phi, cache)
            _, Ac[d] = sa.solve_day(price, scen, wsc, E0, Phi, cache, b_fixed=V5B[d])
            Cc[d] = va.sim_cost(V5B[d][None, :], E0, scen, wsc, price, Estar, Phi)[0]
            bCD = va.procure(V5B[d], E0, scen, wsc, price, Estar, Phi)
            Bc_[d] = va.sim_cost(bCD[None, :], E0, scen, wsc, price, Estar, Phi)[0]
            _, Acd[d] = sa.solve_day(price, scen, wsc, E0, Phi, cache, b_fixed=bCD)
            if (d - WARMUP_DAY) % 60 == 0:
                print(f"    d={d} 累计 {time.perf_counter()-t0:.0f}s", flush=True)
        sAl, sAc, sB, sC, sAcd = Al[idx].sum(), Ac[idx].sum(), Bc_[idx].sum(), Cc[idx].sum(), Acd[idx].sum()
        pol_v6 = (sC - sAc) / 1e4
        pol_cd = (sB - sAcd) / 1e4
        res3 = dict(A_LB=sAl / 1e4, A_V6=sAc / 1e4, A_CD=sAcd / 1e4, B_CD=sB / 1e4, C_V6=sC / 1e4,
                    政策差距_V6合同=pol_v6, 政策差距_CD合同=pol_cd, 政策差距之差=pol_cd - pol_v6,
                    求解残差上界_唯一合法=(sB - sAl) / 1e4,
                    分解_同合同政策差距=(sB - sAcd) / 1e4,
                    分解_CD合同anticipative次优性=(sAcd - sAl) / 1e4,
                    合同余量_V6合同=(sAc - sAl) / 1e4)
        out["Part3_solve_gap"] = res3
        print(f"    A(松弛下界)      = {sAl/1e4:9.4f}")
        print(f"    A_V6(V6合同自由储能)= {sAc/1e4:9.4f}")
        print(f"    A_CD(CD合同自由储能)= {sAcd/1e4:9.4f}")
        print(f"    B (CD合同因果)   = {sB/1e4:9.4f}")
        print(f"    C (V6合同因果)   = {sC/1e4:9.4f}")
        print(f"  → 政策差距(V6合同) = {pol_v6:+.4f}；政策差距(CD合同) = {pol_cd:+.4f}")
        print(f"  → 两者差 {pol_cd-pol_v6:+.4f} ⇒ 政策差距随合同变化，"
              f"{'减法不合法（原 10 万推导作废）' if abs(pol_cd-pol_v6) > 1.0 else '两者接近'}")
        print(f"  → 唯一合法的求解残差上界：B − J*_causal ≤ B − A = "
              f"{res3['求解残差上界_唯一合法']:+.4f} 万元（不可再收紧）")
        print(f"  → 其分解：同合同政策差距(B−A_CD) {res3['分解_同合同政策差距']:+.4f} ＋ "
              f"CD合同 anticipative 次优性(A_CD−A) {res3['分解_CD合同anticipative次优性']:+.4f}")
        print("    （注意：B−A_CD 是『政策差距』不是求解残差；J*_causal ≥ A 恒成立，故只有 B−A 是上界）")

    # ---------- Part 2 + 4：CVaR 修正 + 连续回放 ----------
    if 2 in parts or 4 in parts:
        pre = sq.precompute(D) if 1 not in parts else pre
        print("\n[Part 2/4] CVaR 离散修正 + 各候选自身 SOC 连续回放")
        # CVaR 自检
        rng = np.random.default_rng(0)
        zz = rng.normal(size=12) * 100
        ww = rng.random(12)
        ww = ww / ww.sum()
        for al in (0.7, 0.9, 0.95):
            d1, d2 = cvar_discrete(zz, ww, al), cvar_ru(zz, ww, al)
            print(f"    CVaR 自检 α={al}: 标准式 {d1:.6f}｜R-U {d2:.6f}｜差 {abs(d1-d2):.2e}")
        out["Part2_cvar_selftest"] = {str(a): float(abs(cvar_discrete(zz, ww, a) - cvar_ru(zz, ww, a)))
                                      for a in (0.7, 0.9, 0.95)}

        # 连续回放：V6 基准 + 各生成器 CD 合同
        def replay(gen, use_cvar=False, alpha=0.90, name=""):
            E = float(E_FEB1)
            tot = np.zeros(len(dates))
            for d in range(WARMUP_DAY, len(dates)):
                m = months[d]
                Estar, Phi = Estar_of[m], Phi_of[m]
                if gen is None:
                    b = V5B[d]
                else:
                    scen, wsc = gen(d, D, pre)
                    b = (procure_cvar_new(V5B[d], E, scen, wsc, price, Estar, Phi, alpha)
                         if use_cvar else va.procure(V5B[d], E, scen, wsc, price, Estar, Phi))
                _, _, e, _, Etraj = va.execute_value(b, net[d], price, E, Estar)
                tot[d] = float((b * price).sum() + MULT * (e * price).sum())
                E = Etraj[-1]
            return tot[idx].sum() / 1e4, E

        rows = {}
        v6t, e6 = replay(None)
        rows["V6(连续,对拍)"] = dict(总费=v6t, 年末SOC=float(e6))
        print(f"    V6 连续回放 = {v6t:.4f}（对拍 {V6_TOTAL}，差 {v6t-V6_TOTAL:+.4f}）")
        for gname, gen in sq.GENS.items():
            t0 = time.perf_counter()
            tt, ee = replay(gen)
            rows[f"{gname}/期望/连续"] = dict(总费=tt, 年末SOC=float(ee))
            print(f"    {gname} 期望·连续 = {tt:9.4f}（相对 V6 {tt-v6t:+.4f}） 年末SOC {ee:.0f}"
                  f"  [{time.perf_counter()-t0:.0f}s]", flush=True)
        if 2 in parts:
            for gname in ("G3因果残差", "G4尾部增强残差"):
                gen = sq.GENS[gname]
                t0 = time.perf_counter()
                tt, ee = replay(gen, use_cvar=True)
                rows[f"{gname}/CVaR修正/连续"] = dict(总费=tt, 年末SOC=float(ee))
                print(f"    {gname} CVaR(修正)·连续 = {tt:9.4f}（相对 V6 {tt-v6t:+.4f}）"
                      f"  [{time.perf_counter()-t0:.0f}s]", flush=True)
        out["Part4_continuous"] = rows

    (BASE / "results" / "q2_diag_fix.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_diag_fix.json   总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    parts = tuple(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else (1, 2, 3, 4, 5)
    main(parts)
