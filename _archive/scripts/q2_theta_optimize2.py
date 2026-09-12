# -*- coding: utf-8 -*-
"""
Q2 §12-4 后两层：θ 加入"历史时间权重 ρ"与"少量时段分组 β"（**只读**）
=====================================================================
承 §14（θ = β × W 的滚动选参已降 86.67 万元）。本脚本再测两层：
  · 第 3 层 A：历史样本**指数时间权重** ρ —— 近期历史权重更高（w_i ∝ ρ^(age)）；
  · 第 4 层 B：**两个时段分组**的分位数 —— 高价时段 β_hi、其余 β_lo。
               分组依据是**题面给定的电价曲线**（p ≥ p90），属**事前可定义**，
               不使用评价期的缺口/费用信息（避免用评价期信息决定分组）。

方法完全沿用 §14 的**滚动（walk-forward）选参**：
  · 第 m 月的 θ_m 只用 1月~(m−1)月末的数据选出；训练费来自**连续回放**（从 1/1 的 6000 kWh 起、
    状态跨日传递）的前缀累计费用最小化；
  · 评价期 2/1–12/31、公共预热 E=10800，只用于报告；
  · "事后最优固定 θ" 不可实施，仅作参考。
本脚本自带预测/回放实现，**不导入也不修改 q2_model 的模型逻辑**（只借用其 LP 与执行规则）。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from q2_model import (T, MULT, E_INIT, WARMUP_DAY, E_FEB1,  # noqa: E402
                      Policy, load_data, make_plan, execute)

MIN_HIST = 7
POL = Policy("S1c", forecast="cquant", W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")

BETA_GRID = [0.70, 0.80, 0.90]
W_GRID = [7, 14]
RHO_GRID = [1.0, 0.9, 0.8]
THETA0 = dict(beta=0.70, W=14, rho=1.0)          # 现状

# 第 4 层：分组 β（β_hi ≥ β_lo），分组由给定电价定（p ≥ p90）
HI_GRID = [0.80, 0.90, 0.95]
LO_GRID = [0.60, 0.70, 0.80]


def _wquantile(x, q, rho):
    """加权经验分位数。ρ=1 时严格退化为 np.quantile(method='linear')，便于与现有模型对齐。"""
    n = len(x)
    if abs(rho - 1.0) < 1e-12:
        return float(np.quantile(x, q, method="linear"))
    w = rho ** np.arange(n - 1, -1, -1)          # 最新的一天权重最大
    idx = np.argsort(x)
    cw = np.cumsum(w[idx])
    cw = cw / cw[-1]
    return float(np.interp(q, cw, x[idx]))


def forecast_custom(d, D, W, betas, rho):
    """分时段上尾分位数（可分组、可时间加权）。betas 为长度 T 的数组。"""
    hist = D["net"][max(0, d - W):d]
    if hist.shape[0] < MIN_HIST:
        return D["net_ref"].copy()
    base = np.median(hist, axis=0)
    resid = hist - base
    out = np.empty(T)
    for t in range(T):
        out[t] = base[t] + max(_wquantile(resid[:, t], betas[t], rho), 0.0)
    return out


def betas_of(th, price, p90):
    if "beta" in th:
        return np.full(T, float(th["beta"]))
    return np.where(price >= p90, float(th["hi"]), float(th["lo"]))


def rollout(D, theta_by_month, start_idx, E_start, p90):
    """连续回放；theta_by_month: {月份: θ}。返回逐日 (计划费, 紧急费, 紧急量)。"""
    price, net, nd = D["price"], D["net"], len(D["dates"])
    E = float(E_start)
    dp = np.zeros(nd); de = np.zeros(nd); dk = np.zeros(nd)
    for d in range(start_idx, nd):
        th = theta_by_month[D["_month"][d]]
        N_hat = forecast_custom(d, D, th["W"], betas_of(th, price, p90), th["rho"])
        b, c_plan = make_plan(POL, price, N_hat, E)
        _c, _d, e, _w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum())
        de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum())
        E = Etraj[-1]
    return dp, de, dk


def monthly_schedule(D, thetas, p90):
    """滚动选参：第 m 月用 1月~(m−1)月末累计训练费最小的 θ。"""
    cum = {}
    for i, th in enumerate(thetas):
        dp, de, _ = rollout(D, {m: th for m in range(1, 13)}, 0, E_INIT, p90)
        cum[i] = np.cumsum(dp + de)
    sched = {}
    for m in range(2, 13):
        te = D["_month_end"][m - 1]
        i = min(range(len(thetas)), key=lambda k: cum[k][te])
        sched[m] = thetas[i]
    return sched


def run_experiment(name, thetas, D, p90, idx_eval):
    print(f"\n{'='*78}\n### {name}（候选 {len(thetas)} 个）\n{'='*78}")
    sched = monthly_schedule(D, thetas, p90)
    print("滚动选参日程：")
    for m in range(2, 13):
        th = sched[m]
        tag = (f"β={th['beta']:.2f}" if "beta" in th
               else f"β_hi={th['hi']:.2f}/β_lo={th['lo']:.2f}")
        print(f"  用于 {m:2d} 月 → {tag}, W={th['W']:2d}, ρ={th['rho']:.2f}")

    def total(dp, de, idx):
        return float(dp[idx].sum() + de[idx].sum())

    dp0, de0, _ = rollout(D, {m: THETA0 for m in range(2, 13)}, WARMUP_DAY, E_FEB1, p90)
    dpw, dew, dkw = rollout(D, sched, WARMUP_DAY, E_FEB1, p90)
    print(f"\n  现状(β=0.70,W=14,ρ=1)：总 {total(dp0, de0, idx_eval)/1e4:>10.4f} 万元")
    print(f"  滚动 θ            ：总 {total(dpw, dew, idx_eval)/1e4:>10.4f} 万元"
          f"（{total(dpw, dew, idx_eval)/1e4 - total(dp0, de0, idx_eval)/1e4:+.4f}）"
          f" | 计划 {dpw[idx_eval].sum()/1e4:.4f} | 紧急 {dew[idx_eval].sum()/1e4:.4f}"
          f" | 紧急电量 {dkw[idx_eval].sum():.1f} kWh")
    return sched, total(dpw, dew, idx_eval)


def main():
    D = load_data()
    price = D["price"]
    p90 = float(np.quantile(price, 0.90))
    dates = D["dates"]
    nd = len(dates)
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    D["_month"] = months
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    idx_eval = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=== 说明 =====================================================")
    print(f"  高价分组阈值 p90 = {p90:.4f} 元/kWh（由题面给定电价曲线确定，事前可定义）")
    print(f"  高价时段 {int((price >= p90).sum())} 个/天；现状口径 (β=0.70, W=14, ρ=1) 评价期 "
          f"应在 1719.7084 万元附近")

    # ---- 第 3 层：时间权重 ρ ----
    thA = [dict(beta=b, W=w, rho=r) for b in BETA_GRID for w in W_GRID for r in RHO_GRID]
    _, totA = run_experiment("第 3 层 A：θ=(β, W, ρ) 含历史时间权重", thA, D, p90, idx_eval)

    # 归因：同一网格内 ρ 限定为 1 的滚动结果
    thA1 = [dict(beta=b, W=w, rho=1.0) for b in BETA_GRID for w in W_GRID]
    sched1 = monthly_schedule(D, thA1, p90)
    dp1, de1, _ = rollout(D, sched1, WARMUP_DAY, E_FEB1, p90)
    totA1 = float(dp1[idx_eval].sum() + de1[idx_eval].sum())
    print(f"\n  【归因】同一网格内 ρ=1 限定：{totA1/1e4:.4f} 万元"
          f" ⇒ ρ（时间权重）的增量收益 = {(totA1-totA)/1e4:+.4f} 万元")

    # ---- 第 4 层：分组 β ----
    thB = [dict(hi=h, lo=l, W=w, rho=1.0)
           for h in HI_GRID for l in LO_GRID for w in W_GRID if h >= l]
    _, totB = run_experiment("第 4 层 B：θ=(β_hi, β_lo, W) 两时段分组", thB, D, p90, idx_eval)
    print("\n  【归因】分组 β 相对 'ρ=1 且单一 β' 的滚动结果"
          f"（{totA1/1e4:.4f} 万元）： {(totA1-totB)/1e4:+.4f} 万元")

    # ---- 第 4 层 C：分组 β 与时间权重合并（全网格，检验是否互补）----
    thC = [dict(hi=h, lo=l, W=w, rho=r)
           for h in HI_GRID for l in LO_GRID for w in W_GRID for r in RHO_GRID]
    _, totC = run_experiment("第 4 层 C：θ=(β_hi, β_lo, W, ρ) 分组 β + 时间权重",
                             thC, D, p90, idx_eval)
    print(f"\n  【归因】C 相对 A（含时间权重的最好单一 β）：{(totA-totC)/1e4:+.4f} 万元"
          "（≤0 表示分组 β 在并入时间权重后仍无增益）")

    print("\n" + "=" * 78)
    print(f"  现状                       1719.7084 万元")
    print(f"  §14 滚动 (β,W)             1633.0429 万元（−86.6654）")
    print(f"  本层 A 滚动 (β,W,ρ)        {totA/1e4:>10.4f} 万元（(A1−A) 归因 ρ={((totA1-totA)/1e4):+.4f}）")
    print(f"  本层 B 滚动 (β_hi,β_lo,W)  {totB/1e4:>10.4f} 万元（相对 A1 归因分组 β={(totA1-totB)/1e4:+.4f}）")
    print(f"  本层 C 滚动 (β_hi,β_lo,W,ρ) {totC/1e4:>9.4f} 万元（相对 A 归因 {(totA-totC)/1e4:+.4f}）")
    print("  注：A/B/C 网格比 §14 更窄（W 只到 14、β 只到 0.90/0.95）；")
    print("      但 A1（ρ=1 限制）已精确复现 §14 的 1633.0429，故 A 的 ρ 归因是干净可比的。")


if __name__ == "__main__":
    main()
