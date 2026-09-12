# -*- coding: utf-8 -*-
"""
Q2 §12-4：费用驱动的策略参数优化 θ（**只读**：不改模型、不写任何结果文件）
=========================================================================
目标（用户 2026-09-11 评审建议）：
    θ* = argmin_{θ∈Θ} Σ_{k∈D_train} J_k^{S1}(θ)
  θ 逐步包含：分位数 β → 历史窗口 W → （后续）时间权重 / 少量时段分组参数。

本脚本先做 **θ = (β, W)** 两层，并采用**滚动（walk-forward）选参**，
因为上一轮已发现"固定用 1 月选参"的代价达 66 万元量级（§13.1）。

严格纪律（避免任何数据泄漏）：
  · **选参**：第 m 月使用的 θ_m 只用**该月之前**的数据（1 月 ~ m-1 月底）选出；
    且训练费用来自**连续回放**（状态跨日传递，从 1/1 的 6000 kWh 起），
    不是"每天独立初始化后相加"。
  · **评价**：2025-02-01 ~ 12-31，公共预热起评 E = 10800 kWh（与四组合表同口径）；
    评价期**只用于报告**，不参与选参。
  · 参考上界"事后最优固定 θ"**仅作展示**，不可实施，明确标注。
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

from q2_model import (MULT, E_INIT, WARMUP_DAY, E_FEB1,  # noqa: E402
                      Policy, load_data, forecast_net, make_plan, execute, replay)

BETA_GRID = [0.60, 0.70, 0.80, 0.90, 0.95]
W_GRID = [7, 14, 21]
THETA0 = (0.70, 14)


def make_pol(beta: float, W: int) -> Policy:
    """S1 规则：分位数日前计划 + must/max 尽限反馈（greedy，无需执行层 LP）。"""
    return Policy("S1", forecast="cquant", beta=beta, W=W, exec_mode="greedy",
                  discharge_policy="must", charge_policy="max")


def day_cost(R, price):
    """逐日费用 = 计划购电费 + 紧急购电费（元）。"""
    return (R["B"] * price).sum(axis=1) + MULT * (R["EM"] * price).sum(axis=1)


def walk_forward(D, month_pol, start_idx, E_start):
    """按"每月一套 θ"连续回放：状态跨月传递，返回 (计划费, 紧急费, 紧急量, 紧急日数)。"""
    price, net = D["price"], D["net"]
    E = float(E_start)
    plan = em = emk = 0.0
    emd = 0
    for d in range(start_idx, len(D["dates"])):
        pol = month_pol[D["_month"][d]]
        N_hat = forecast_net(pol, d, D)
        b, c_plan = make_plan(pol, price, N_hat, E)
        _c, _d, e, _w, Etraj = execute(pol, b, c_plan, N_hat, net[d], price, E)
        plan += float((b * price).sum())
        em += float(MULT * (e * price).sum())
        emk += float(e.sum())
        emd += int(e.sum() > 1e-6)
        E = Etraj[-1]
    return plan, em, emk, emd


def main():
    D = load_data()
    price, dates = D["price"], D["dates"]
    nd = len(dates)

    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in dates])
    month_end = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}
    D["_month"] = months
    eval_idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    thetas = [(b, w) for b in BETA_GRID for w in W_GRID]
    print(f"=== θ = (β, W) 共 {len(thetas)} 个候选；S1 规则（must/max）===")
    print(f"    β ∈ {BETA_GRID}，W ∈ {W_GRID}\n")

    # ---------- 1. 每个 θ 的连续全年回放（训练用前缀费用）----------
    print("训练：各 θ 从 1/1（E=6000）连续回放，取各月末累计费用（元 → 仅用于选参）")
    cum = {}
    for th in thetas:
        R = replay(make_pol(*th), D)
        cum[th] = np.cumsum(day_cost(R, price))
    print(f"  完成 {len(thetas)} 条全年轨迹（每条 {nd} 天，状态跨日传递）。\n")

    # ---------- 2. 滚动选参：第 m 月只用 <m 的数据 ----------
    print("=== 滚动选参日程（θ_m 由 1月~（m−1）月末的累计费用最小化选出）===")
    schedule = {}
    for m in range(2, 13):
        te = month_end[m - 1]
        best = min(thetas, key=lambda th: cum[th][te])
        schedule[m] = best
        print(f"  用于 {m:2d} 月  →  β = {best[0]:.2f}, W = {best[1]:2d}"
              f"   （训练至 {str(dates[te])[:10]}，累计训练费 {cum[best][te]/1e4:.2f} 万元）")

    # ---------- 3. 评价 ----------
    print("\n=== 评价期（2025-02-01 ~ 12-31，公共预热 E=10800）===")
    month_pol = {m: make_pol(*schedule[m]) for m in range(2, 13)}

    # 3a 现状：固定 θ0 = (0.70, 14)
    p0, e0, k0, d0 = walk_forward(D, {m: make_pol(*THETA0) for m in range(2, 13)},
                                  WARMUP_DAY, E_FEB1)
    # 3b 滚动 θ
    p1, e1, k1, d1 = walk_forward(D, month_pol, WARMUP_DAY, E_FEB1)
    # 3c 参考：事后最优固定 θ（不可实施，仅展示）
    fixed = {}
    for th in thetas:
        R = replay(make_pol(*th), D, start_day=WARMUP_DAY, E_start0=E_FEB1)
        fixed[th] = float((R["B"][eval_idx] * price).sum()
                          + MULT * (R["EM"][eval_idx] * price).sum())
    best_fixed = min(fixed, key=fixed.get)

    print(f"{'方案':<26}{'总费用/万元':>13}{'计划/万元':>12}{'紧急/万元':>12}"
          f"{'紧急电量/kWh':>14}{'紧急日数':>9}")
    print("-" * 88)
    print(f"{'现状 固定 β=0.70, W=14':<26}{(p0+e0)/1e4:>13.4f}{p0/1e4:>12.4f}"
          f"{e0/1e4:>12.4f}{k0:>14.1f}{d0:>9d}")
    print(f"{'滚动 θ（walk-forward）':<26}{(p1+e1)/1e4:>13.4f}{p1/1e4:>12.4f}"
          f"{e1/1e4:>12.4f}{k1:>14.1f}{d1:>9d}")
    print(f"{'【参考·不可实施】事后最优固定':<24}{fixed[best_fixed]/1e4:>13.4f}"
          f"{'':>12}{'':>12}{'':>14}{'':>9}  (β={best_fixed[0]:.2f}, W={best_fixed[1]})")
    print(f"\n  滚动相对现状：{(p1+e1)/1e4 - (p0+e0)/1e4:+.4f} 万元 "
          f"（{((p1+e1)/(p0+e0) - 1) * 100:+.3f}%）")
    print(f"  现状距事后最优：{fixed[best_fixed]/1e4 - (p0+e0)/1e4:+.4f} 万元"
          f"；滚动距事后最优：{fixed[best_fixed]/1e4 - (p1+e1)/1e4:+.4f} 万元")

    # ---------- 4. 分月对比（收益是否稳定）----------
    print("\n=== 分月对比（万元，各月单独从公共预热起跑，仅用于分月展示）===")
    R0 = replay(make_pol(*THETA0), D, start_day=WARMUP_DAY, E_start0=E_FEB1)
    print(f"{'月份':>5}{'现状':>11}{'滚动 θ':>11}{'差(滚动−现状)':>15}")
    print("-" * 44)
    tot = 0.0
    for m in range(2, 13):
        m_idx = eval_idx[months[eval_idx] == m]
        c0 = float((R0["B"][m_idx] * price).sum() + MULT * (R0["EM"][m_idx] * price).sum())
        R1 = replay(make_pol(*schedule[m]), D, start_day=WARMUP_DAY, E_start0=E_FEB1)
        c1 = float((R1["B"][m_idx] * price).sum() + MULT * (R1["EM"][m_idx] * price).sum())
        tot += c1 - c0
        print(f"{m:>5}{c0/1e4:>11.4f}{c1/1e4:>11.4f}{(c1-c0)/1e4:>15.4f}")
    print("-" * 44)
    print(f"  分月差合计 {tot/1e4:+.4f} 万元（此处各月单独从公共预热起跑，未串接跨月状态，")
    print("   故与上表的滚动−现状之差可能略有出入；上表为真实串接结果。）")

    print("\n说明：事后最优固定 θ 用评价期选出，**不可实施**，仅用于显示可达范围；")
    print("      滚动方案每个月只使用该月之前的数据，是因果、可实施的选参流程。")


if __name__ == "__main__":
    main()
