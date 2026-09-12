# -*- coding: utf-8 -*-
r"""
§27 对采购有用的联合预测：三种场景构造的对照（只读，不改模型）
================================================================================
承接 §26 的判别结论（**主因是场景失配，不是没解好**），本轮研究"对采购有用的联合预测"。
日前采购需要场景回答三件事：① 明天总净需求大致多少；② 缺口主要出现在哪些**连续**时段；
③ 缺口出现前有没有可入储的富余。据此对照三种**严格因果**的场景构造：

  G1 近期整日轨迹（基准）：最近 30 天、纯新近度加权；
  G2 相似日匹配：按"近 3 日负载水平 + 近 3 日光伏水平 + 星期类型 + 月份"匹配历史日
     —— 特征**只用该日开始前可得的信息**（第 k 天的特征用 k 之前的数据算），**不碰该日实际总量**；
  G3 因果残差构造：当天因果基线 base_d（前 14 天中位数）+ 历史**残差轨迹**（保留日内连续相关），
     —— 保留第 d 天自身的水平，只借历史的"形状/连续缺口"。

评价（同一 CD 求解器、同一 V6 起点、因果价值反馈部署，差异只归因于场景构造）：
  · **迁移**：场景上 CD 优化了多少（C−B） vs 实测兑现多少（V6−实际）；迁移比越接近 +1 越好；
  · **覆盖**：晚峰覆盖（场景达到实测峰值的频率）、总需求偏差、形状相关（缺口时段/富余位置）。
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

from q2_model import (T, MULT, E_MIN, E_MAX, ETA_C, E_FEB1, WARMUP_DAY, load_data)  # noqa: E402
import q2_value_adp as va  # noqa: E402

V6_TOTAL = 1581.6655
K = 16
RHO = 0.94
LOOKBACK = 30
BW = 14           # 基线/残差用的中位数窗口


# ============================ 预计算（全因果） ============================
def precompute(D):
    net, load, pv, dates = D["net"], D["load"], D["pv"], D["dates"]
    nd = len(dates)
    base_all = np.zeros_like(net)
    for d in range(nd):
        base_all[d] = np.median(net[max(0, d - BW):d], axis=0) if d > 0 else D["net_ref"]
    dl = load.sum(axis=1); dpv = pv.sum(axis=1)
    ctxL = np.zeros(nd); ctxP = np.zeros(nd)
    for d in range(nd):
        s = max(0, d - 3)
        ctxL[d] = dl[s:d].mean() if d > s else dl[d]
        ctxP[d] = dpv[s:d].mean() if d > s else dpv[d]
    import pandas as pd
    wknd = np.asarray(pd.DatetimeIndex(dates).weekday >= 5)
    month = D["_month"]
    hi = D["price"] >= np.quantile(D["price"], 0.90)
    peak_resid = np.zeros(nd)
    for d in range(nd):
        peak_resid[d] = float((net[d] - base_all[d])[hi].max()) if d > 0 else 0.0
    return dict(base_all=base_all, ctxL=ctxL, ctxP=ctxP, wknd=wknd, month=month,
                hi=hi, peak_resid=peak_resid)


# ============================ 三种场景构造 ============================
def gen_recent(d, D, pre, K=K):
    """G1 近期整日（纯新近度）。"""
    net = D["net"]
    days = np.arange(max(0, d - LOOKBACK), d)
    if len(days) == 0:
        return net[d:d + 1], np.array([1.0])
    w = RHO ** ((d - 1) - days)
    top = np.argsort(-w)[:K]
    days = days[top]; w = w[top]; w = w / w.sum()
    return net[days], w


def gen_matched(d, D, pre, K=K):
    """G2 相似日匹配（特征只用该日开始前的信息）。"""
    net = D["net"]
    days = np.arange(0, d)
    if len(days) == 0:
        return net[d:d + 1], np.array([1.0])
    sL = np.std(pre["ctxL"][:d + 1]) + 1e-9
    sP = np.std(pre["ctxP"][:d + 1]) + 1e-9
    dist = (np.abs(pre["ctxL"][days] - pre["ctxL"][d]) / sL
            + np.abs(pre["ctxP"][days] - pre["ctxP"][d]) / sP
            + 1.5 * (pre["wknd"][days] != pre["wknd"][d]).astype(float)
            + 0.5 * np.abs(pre["month"][days] - pre["month"][d]))
    top = np.argsort(dist)[:K]
    days = days[top]; dist = dist[top]
    w = 1.0 / (1.0 + dist); w = w / w.sum()
    return net[days], w


def gen_residual(d, D, pre, K=K):
    """G3 因果残差构造：base_d + 历史残差轨迹（保留日内连续相关）。"""
    net = D["net"]; base_all = pre["base_all"]
    days = np.arange(max(0, d - LOOKBACK), d)
    if len(days) == 0:
        return net[d:d + 1], np.array([1.0])
    w = RHO ** ((d - 1) - days)
    top = np.argsort(-w)[:K]
    days = days[top]; w = w[top]; w = w / w.sum()
    scen = base_all[d][None, :] + (net[days] - base_all[days])
    return scen, w


def gen_tail(d, D, pre, K=K, Ktail=5, wtail=0.25):
    """G4 尾部增强残差：近期残差（75% 权重）＋ 历史**最高峰残差**日（25% 权重）。
    显式让场景池覆盖峰值上尾（针对 §27 诊断出的'晚峰覆盖≈0.50'）。"""
    net = D["net"]; base_all = pre["base_all"]
    days = np.arange(max(0, d - LOOKBACK), d)
    if len(days) == 0 or d < 1:
        return net[d:d + 1], np.array([1.0])
    w_rec = RHO ** ((d - 1) - days)
    K1 = K - Ktail
    top_rec = np.argsort(-w_rec)[:K1]
    past = np.arange(0, d)
    top_tail = past[np.argsort(-pre["peak_resid"][past])[:Ktail]]
    scen_rec = base_all[d][None, :] + (net[days[top_rec]] - base_all[days[top_rec]])
    scen_tail = base_all[d][None, :] + (net[top_tail] - base_all[top_tail])
    scen = np.concatenate([scen_rec, scen_tail], axis=0)
    w = np.concatenate([0.75 * (w_rec[top_rec] / w_rec[top_rec].sum()),
                        np.full(Ktail, wtail / Ktail)])
    return scen, w


GENS = {"G1近期整日": gen_recent, "G2相似日匹配": gen_matched,
        "G3因果残差": gen_residual, "G4尾部增强残差": gen_tail}


# ============================ 覆盖指标 ============================
def coverage(scen, wsc, act, base_d, hi):
    """返回 (晚峰覆盖, 总需求偏差, 形状相关)。"""
    smean = (scen * wsc[:, None]).sum(axis=0)
    peak_cov = float(np.mean([(scen[:, t] >= act[t]).mean() for t in np.where(hi)[0]])) if hi.sum() else np.nan
    level_bias = float(smean.sum() - act.sum())
    a = act - act.mean(); s = smean - smean.mean()
    shape_corr = float((a * s).sum() / (np.linalg.norm(a) * np.linalg.norm(s) + 1e-9))
    return peak_cov, level_bias, shape_corr


def main():
    t_all = time.perf_counter()
    D = load_data()
    price, dates, net = D["price"], D["dates"], D["net"]
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    month_start = {m: int(np.min(np.where(months == m)[0])) for m in range(1, 13)}
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]
    pre = precompute(D)
    hi = price >= np.quantile(price, 0.90)          # 高价（晚峰）窗口

    print("=" * 104)
    print("§27 对采购有用的联合预测：三种场景构造对照")
    print("=" * 104)
    print(f"  评价期 {len(idx)} 天；场景数 K={K}；同一 CD 求解器 + 同一 V6 起点 + 因果价值反馈部署")
    print(f"  高价（晚峰）窗口：{int(hi.sum())} 区间/天（p≥p90）")

    # 价值函数 + V6 基准
    Estar_of, Phi_of = {}, {}
    for m in range(2, 13):
        scen = va.fvi_pool(month_start[m], D)
        _, Estar, Phi = va.estimate_value(price, scen)
        Estar_of[m], Phi_of[m] = Estar, Phi
    sched, rV5 = va.reproduce_v5(D, months)
    V5B = rV5["det"]["B"]
    # V6 逐日实际起点 + V6 逐日实际费用（冻结基准）
    soc0 = np.zeros(len(dates)); actV6 = np.zeros(len(dates))
    E = float(E_FEB1)
    for d in range(WARMUP_DAY, len(dates)):
        Estar = Estar_of[months[d]]
        _, _, e, _, Etraj = va.execute_value(V5B[d], net[d], price, E, Estar)
        soc0[d] = E; actV6[d] = float((V5B[d] * price).sum() + MULT * (e * price).sum())
        E = Etraj[-1]
    print(f"  V6 基准 = {actV6[idx].sum()/1e4:.4f} 万元（对拍 {V6_TOTAL}，差 {actV6[idx].sum()/1e4 - V6_TOTAL:+.4f}）")

    # 每个生成器：迁移 + 覆盖
    print("\n[对照] 三种场景构造（同一求解器与部署）")
    results = {}
    for gname, gen in GENS.items():
        sC = np.zeros(len(dates)); sB = np.zeros(len(dates)); aG = np.zeros(len(dates))
        pc = np.zeros(len(dates)); lb = np.zeros(len(dates)); sc_ = np.zeros(len(dates))
        t0 = time.perf_counter()
        for d in range(WARMUP_DAY, len(dates)):
            m = months[d]
            E0 = soc0[d]
            scen, wsc = gen(d, D, pre)
            Phi, Estar = Phi_of[m], Estar_of[m]
            sC[d] = va.sim_cost(V5B[d][None, :], E0, scen, wsc, price, Estar, Phi)[0]
            bG = va.procure(V5B[d], E0, scen, wsc, price, Estar, Phi)
            sB[d] = va.sim_cost(bG[None, :], E0, scen, wsc, price, Estar, Phi)[0]
            _, _, e, _, _ = va.execute_value(bG, net[d], price, E0, Estar)
            aG[d] = float((bG * price).sum() + MULT * (e * price).sum())
            pc[d], lb[d], sc_[d] = coverage(scen, wsc, net[d], pre["base_all"][d], hi)
            if (d - WARMUP_DAY) % 60 == 0:
                print(f"    [{gname}] d={d} 累计 {time.perf_counter()-t0:.0f}s", flush=True)
        C, B, A = sC[idx].sum(), sB[idx].sum(), aG[idx].sum()
        V6 = actV6[idx].sum()
        scen_impr = (C - B) / 1e4
        act_impr = (V6 - A) / 1e4
        ratio = act_impr / scen_impr if abs(scen_impr) > 1e-6 else float("nan")
        results[gname] = dict(scen_impr=scen_impr, act_total=A / 1e4, act_impr=act_impr,
                              ratio=ratio, peak_cov=float(np.nanmean(pc[idx])),
                              level_bias=float(np.mean(lb[idx])), shape_corr=float(np.mean(sc_[idx])))
        print(f"  {gname:<12} 场景改进 {scen_impr:+8.4f}  实测总费 {A/1e4:9.4f}  迁移 {act_impr:+8.4f}"
              f"  迁移比 {ratio:+6.3f}")

    # 汇总表
    print("\n" + "=" * 104)
    print("场景构造对照表（迁移比越接近 +1 越好；覆盖指标越高越好）")
    print("=" * 104)
    print(f"  {'构造':<14}{'场景改进':>10}{'实测总费':>11}{'迁移':>9}{'迁移比':>8}"
          f"{'晚峰覆盖':>10}{'总需求偏差':>12}{'形状相关':>10}")
    for gname, r in results.items():
        print(f"  {gname:<14}{r['scen_impr']:>10.4f}{r['act_total']:>11.4f}{r['act_impr']:>+9.4f}"
              f"{r['ratio']:>8.3f}{r['peak_cov']:>10.3f}{r['level_bias']:>12.1f}{r['shape_corr']:>10.3f}")
    print(f"\n  参照 V6 = {V6_TOTAL:.4f}；迁移指标单位：万元；总需求偏差单位：kWh/天")

    out = dict(口径=dict(K=K, RHO=RHO, LOOKBACK=LOOKBACK, BW=BW, 评价期天数=int(len(idx)), V6=V6_TOTAL),
               结果=results)
    (BASE / "results" / "q2_scenario_quality.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘：results/q2_scenario_quality.json   总耗时 {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
