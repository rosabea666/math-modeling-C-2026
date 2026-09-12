# -*- coding: utf-8 -*-
"""
Q2 β 重校准 + 紧急电量"功率/储能"分解（**只读**：不改模型、不写任何结果文件）
=============================================================================
动机（用户 2026-09-11 评审）：
  1. 既有 β=0.7 是在 **C 的 must/planned** 执行规则下于 **1 月**选定的；
     主方案已改为 **S1 的 must/max**。执行方式变了，相同计划冗余的边际价值也变了，
     故 β_C* 未必等于 β_S1*。→ 本脚本在**同一训练期（1 月）**下分别对两种执行规则重选 β。
  2. S1 仍有紧急购电，但未回答：这些缺口主要来自**放电功率不够快**，还是**当时电量不够**？
     → 精确会计分解 e_t = e_t^P + e_t^E。

纪律（严格遵守，避免数据泄漏）：
  · 选参 **只用 2025 年 1 月**（连续回放、状态跨日传递，从 1/1 的 6000 kWh 起）。
  · 评价期 2025-02-01~12-31 **只用于报告**，不参与选参。
  · 评价口径与四组合表一致：公共预热，2/1 起评 E = 10800 kWh。
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

from q2_model import (T, E_MIN, ETA_D, MULT, WARMUP_DAY, E_FEB1,  # noqa: E402
                      Policy, load_data, metrics, replay)

BETA_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
Q_POW = 5000.0 / 6.0            # 单区间最大放电量（kWh）= 5000 kW × 1/6 h
RULES = {"C 规则 must/planned": "planned", "S1 规则 must/max": "max"}


def policy_for(beta: float, charge_policy: str) -> Policy:
    return Policy("x", forecast="cquant", beta=beta, W=14, exec_mode="greedy",
                  discharge_policy="must", charge_policy=charge_policy)


def jan_only(D: dict) -> dict:
    """1 月切片，供训练期选参（连续回放，状态跨日传递）。"""
    k = 31
    return dict(price=D["price"], dates=D["dates"][:k], net=D["net"][:k],
                net_ref=D["net_ref"], load=D["load"][:k], pv=D["pv"][:k])


def main():
    D = load_data()
    Dj = jan_only(D)
    idx_eval = np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]

    print("=== 训练期（2025-01-01 ~ 01-31）β 选择：两种执行规则各选一次 ===")
    print("  连续回放，E(1/1 0:00) = 6000 kWh；只用于选参，不用评价期。\n")
    print(f"{'β':>6} | {'C 规则(must/planned)':>22} | {'S1 规则(must/max)':>22}")
    print(f"{'':>6} | {'1月总费用/万元':>22} | {'1月总费用/万元':>22}")
    print("-" * 60)
    jan = {tag: {} for tag in RULES}
    for beta in BETA_GRID:
        row = []
        for tag, cp in RULES.items():
            R = replay(policy_for(beta, cp), Dj)
            m = metrics(R, D, np.arange(len(Dj["dates"])))
            jan[tag][beta] = m["总费用"] / 1e4
            row.append(m["总费用"] / 1e4)
        print(f"{beta:>6.2f} | {row[0]:>22.4f} | {row[1]:>22.4f}")
    best = {tag: min(jan[tag], key=jan[tag].get) for tag in RULES}
    for tag in RULES:
        print(f"  → {tag}：β* = {best[tag]:.2f}（1月费用 {jan[tag][best[tag]]:.4f} 万元）")

    print("\n=== 评价期（2025-02-01 ~ 12-31，334 天；公共预热 E=10800）===")
    print("  说明：β 只用 1 月选出；此处仅报告，不用于选参。\n")
    print(f"{'执行规则':<20}{'β':>6}{'总费用/万元':>12}{'计划/万元':>11}{'紧急/万元':>11}"
          f"{'紧急电量/kWh':>14}{'紧急日数':>9}")
    print("-" * 84)
    keep = {}
    for tag, cp in RULES.items():
        for beta in ([0.70, best[tag]] if best[tag] != 0.70 else [0.70]):
            R = replay(policy_for(beta, cp), D, start_day=WARMUP_DAY, E_start0=E_FEB1)
            m = metrics(R, D, idx_eval)
            keep[(tag, beta)] = (R, m)
            flag = "  ← 沿用" if abs(beta - 0.70) < 1e-9 else "  ← 本规则重选"
            print(f"{tag:<20}{beta:>6.2f}{m['总费用']/1e4:>12.4f}{m['计划费用']/1e4:>11.4f}"
                  f"{m['紧急费用']/1e4:>11.4f}{m['紧急电量']:>14.1f}{m['紧急日数']:>9d}{flag}")

    # 对照：S1 规则下把整个 β 网格在评价期都算一遍（仅作曲线展示，不用于选参）
    print("\n  附：S1 规则下评价期费用随 β 的曲线（仅展示，不用于选参）")
    curve = []
    for beta in BETA_GRID:
        R = replay(policy_for(beta, "max"), D, start_day=WARMUP_DAY, E_start0=E_FEB1)
        m = metrics(R, D, idx_eval)
        curve.append((beta, m["总费用"] / 1e4))
    for beta, c in curve:
        mark = " ← 1月选出" if abs(beta - best["S1 规则 must/max"]) < 1e-9 else ""
        print(f"    β={beta:.2f}: {c:.4f} 万元{mark}")

    # ---------------- 紧急电量分解（S1，评价期）----------------
    beta_s1 = best["S1 规则 must/max"]
    R, m = keep[("S1 规则 must/max", beta_s1)]
    net = D["net"][idx_eval]                 # 实测净负载（kWh）
    b = R["B"][idx_eval]
    E_prev = R["E"][idx_eval][:, :T]         # 各区间起点的储能
    e_act = R["EM"][idx_eval]

    Dshort = np.maximum(net - b, 0.0)                                   # 缺口 D_t
    A = np.maximum(0.0, (E_prev - E_MIN) * ETA_D)                       # 可用放电量 A_t
    e_P = np.maximum(Dshort - Q_POW, 0.0)                               # 功率受限部分
    e_E = np.minimum(Dshort, Q_POW) - np.minimum(np.minimum(Dshort, Q_POW), A)  # 储能受限部分
    resid = float(np.max(np.abs(e_P + e_E - e_act)))

    price = D["price"]
    kwh_P, kwh_E = float(e_P.sum()), float(e_E.sum())
    cost_P = float((MULT * price[None, :] * e_P).sum())
    cost_E = float((MULT * price[None, :] * e_E).sum())
    kwh_tot, cost_tot = float(e_act.sum()), float((MULT * price[None, :] * e_act).sum())

    print(f"\n=== 紧急电量分解（S1，β={beta_s1:.2f}，评价期）===")
    print(f"  恒等式校验：max|e_P + e_E − e_实际| = {resid:.3e} kWh "
          f"{'✔ 精确成立' if resid < 1e-6 else '**不成立**'}")
    print(f"\n{'分量':<26}{'电量/kWh':>14}{'占比':>9}{'费用/万元':>12}{'占比':>9}")
    print("-" * 72)
    print(f"{'功率受限 e^P = (D−Q)⁺':<26}{kwh_P:>14.1f}{kwh_P/kwh_tot*100:>8.1f}%"
          f"{cost_P/1e4:>12.4f}{cost_P/cost_tot*100:>8.1f}%")
    print(f"{'储能受限 e^E = min(D,Q)−min(D,Q,A)':<26}{kwh_E:>14.1f}{kwh_E/kwh_tot*100:>8.1f}%"
          f"{cost_E/1e4:>12.4f}{cost_E/cost_tot*100:>8.1f}%")
    print(f"{'合计':<26}{kwh_tot:>14.1f}{100.0:>8.1f}%{cost_tot/1e4:>12.4f}{100.0:>8.1f}%")

    # 高价区间内的分解（衔接 §9.4）
    p90 = float(np.quantile(price, 0.90))
    hi = price >= p90
    print(f"\n  其中高价区间（p ≥ {p90:.4f} 元/kWh）："
          f"功率受限 {float(e_P[:, hi].sum()):.1f} kWh、储能受限 {float(e_E[:, hi].sum()):.1f} kWh")

    # 分时段（一天 144 个区间）Top-5，看两类缺口落在哪些时段
    per_slot_P = e_P.sum(axis=0)
    per_slot_E = e_E.sum(axis=0)
    topP = np.argsort(per_slot_P)[::-1][:5]
    topE = np.argsort(per_slot_E)[::-1][:5]

    def slot_label(t: int) -> str:
        a = t * 10
        return f"{a//60:02d}:{a%60:02d}-{(a+10)//60:02d}:{(a+10)%60:02d}"

    print("  功率受限 Top-5 时段：" + "，".join(
        f"{slot_label(int(t))}({per_slot_P[t]:.0f})" for t in topP))
    print("  储能受限 Top-5 时段：" + "，".join(
        f"{slot_label(int(t))}({per_slot_E[t]:.0f})" for t in topE))

    print("\n说明：e^P/e^E 是**有明确顺序的会计分解**（先检查功率上限 Q，再检查可用电量 A），")
    print("      不声称两类物理原因完全独立，也不是扩容收益的因果估计。")


if __name__ == "__main__":
    main()
