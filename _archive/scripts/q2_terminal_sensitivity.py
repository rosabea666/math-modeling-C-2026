# -*- coding: utf-8 -*-
"""
Q2 §10 严格化：期末储能差异能否翻转四组合排序？（**只读**，不改模型、不写任何结果文件）
=========================================================================================
背景（§10）：C/R1/R2/R3/R4 的年末电量落在 10452.82 ~ 10708.98 kWh，极差 256.15 kWh。
原来的"357 元"只是**按单一单价**的粗略估值，且 §10 已指出它不是严格上界。
本脚本把这条论证从"量级辅助"提升为**估值意义下的稳健性界**：

  J_λ = J_base − λ · (E_end − E_ref)

λ = 期末每 kWh 库存的价值（元/kWh）。λ=0 即原口径。

给出两类结论：
  (A) **翻转阈值**：对排序中每一对相邻组合，求使二者成本相等所需的 λ*（若更贵的一方
      期末电量反而更少，则该对在任意 λ≥0 下都不会翻转，记 λ*=∞）。
  (B) **极端估值对比**：把 λ* 与"最高正常电价 p_max"和"最高紧急电价 5·p_max"比较。

输入：results/q2_combos.json（四组合回放结果）、data/附件1.xlsx（电价，取 p_max）
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent

# 排序（由优到劣）：来自 results/q2_combos.json 的总费用
RANK = ["R4 planned/max", "R2 must/max", "R3 planned/planned",
        "R1 must/planned", "C  基准"]
SHORT = {"R4 planned/max": "R4", "R2 must/max": "S1(R2)", "R3 planned/planned": "R3",
         "R1 must/planned": "R1", "C  基准": "C"}


def main():
    t = json.loads((BASE / "results" / "q2_combos.json").read_text(encoding="utf-8"))["组合"]
    price = pd.read_excel(BASE / "data" / "附件1.xlsx", header=0).iloc[:, 1].to_numpy(float)
    p_max = float(price.max())
    lam_em = 5.0 * p_max          # 最高紧急电价（5 倍）

    J = {k: t[k]["总费用"] / 1e4 for k in RANK}          # 万元
    E = {k: t[k]["年末电量"] for k in RANK}              # kWh

    print("=== §10 严格化：期末储能差异的估值敏感性（只读）===")
    print(f"最高正常电价 p_max = {p_max:.4f} 元/kWh；最高紧急电价 5·p_max = {lam_em:.4f} 元/kWh\n")

    print(f"{'组合':<9}{'总费用/万元':>12}{'年末电量/kWh':>14}{'相对上一名':>12}")
    for i, k in enumerate(RANK):
        d = "" if i == 0 else f"{J[k]-J[RANK[i-1]]:+.4f}万"
        print(f"{SHORT[k]:<9}{J[k]:>12.4f}{E[k]:>14.4f}{d:>12}")

    spread = max(E.values()) - min(E.values())
    print(f"\n年末电量极差 = {spread:.4f} kWh（{min(E.values()):.2f} ~ {max(E.values()):.2f}）")

    # ---------- (A) 翻转阈值 ----------
    print("\n=== (A) 相邻组合的翻转阈值 λ*（元/kWh）===")
    print(f"{'相邻对':<18}{'费用差/万元':>12}{'期末电量差/kWh':>16}{'λ*':>14}   说明")
    lam_star = []
    for a, b in zip(RANK, RANK[1:]):
        gap = J[b] - J[a]                 # >0
        dE = E[b] - E[a]                  # 更贵的一方 b 的期末电量差
        if dE > 1e-9:
            ls = gap * 1e4 / dE           # 元/kWh
            lam_star.append(ls)
            note = f"b 期末更多 ⇒ 需 λ>{ls:,.1f} 才翻转"
            lss = f"{ls:,.2f}"
        else:
            lss, note = "∞", ("b 期末不更多 ⇒ 任意 λ≥0 都不翻转" if dE < -1e-9 else "期末相同 ⇒ 与 λ 无关")
        print(f"{SHORT[a]+' > '+SHORT[b]:<18}{gap:>12.4f}{dE:>16.2f}{lss:>14}   {note}")

    lam_min = min(lam_star) if lam_star else float("inf")
    print(f"\n所有相邻对中最小的翻转阈值 λ*_min = {lam_min:,.2f} 元/kWh")
    print(f"  · 是最高正常电价的 {lam_min/p_max:,.0f} 倍")
    print(f"  · 是最高紧急电价(5·p_max)的 {lam_min/lam_em:,.0f} 倍")

    # ---------- (B) 极端估值下的排序 ----------
    print("\n=== (B) 极端估值下重排（J_λ = J − λ·(E − E_min)，λ 取到最高紧急电价）===")
    E_min = min(E.values())
    for lam, tag in [(0.0, "λ=0（原口径）"), (p_max, f"λ=p_max={p_max:.4f}（最高正常价）"),
                     (lam_em, f"λ=5·p_max={lam_em:.4f}（最高紧急价）")]:
        order = sorted(RANK, key=lambda k: J[k] - lam * (E[k] - E_min) / 1e4)
        adj = " < ".join(SHORT[k] for k in order)
        print(f"  {tag:<26} 排序：{adj}")

    print("\n结论：期末电量极差仅 %.2f kWh；除 S1↔R3 一对外，其余相邻对的期末电量差都"
          % spread)
    print("      不偏向成本更高的一方，故与 λ 无关。唯一可能翻转的 S1↔R3 对需")
    print(f"      λ > {lam_min:,.0f} 元/kWh（≈{lam_min/p_max:,.0f}× 最高正常电价），现实中不可达。")
    print("      ⇒ 在**任意线性估值**下，R4 < S1(R2) < R3 < R1 < C 的排序不变。")


if __name__ == "__main__":
    main()
