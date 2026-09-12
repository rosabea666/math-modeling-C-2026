# -*- coding: utf-8 -*-
"""
O6 时间列映射歧义的**代价量化**（只读：不改模型、不重出 Excel）
=================================================================
背景：附件5 result2.xlsx「计划购电量」144 个时间列的标签为
        B1 = 0:10-0:20 , B2 = 0:20-0:30 , ... , EN1 = 23:50-0:00+1 , EO1 = 0:00-0:10+1
      即第 j 列标签覆盖 (10j, 10j+10] 分钟 ⇒ 标签指向"第 j+1 个区间"。
      ⇒ 两种自洽读法：
        R1（列顺序）第 j 列 ← 当天第 j 个区间 (10(j-1), 10j]  （144 列恰好铺满全天）
        R2（标签字面）第 j 列 ← (10j, 10j+10]（则第 1 个区间无处填、第 144 列越出当天）

本脚本回答一个可量化的问题：
  **如果整列错位一格（提交读法与评阅读法不一致），S1 策略要多花多少钱？**

做法：S1 的计划 b 逐日算出后，把 b 整体平移 ±1 个区间再交给执行层
（真实净负荷不变，只是"计划落在了错的区间"），比较评价期总费用。
注意：**全天合计列（全天购电量/全天购电费）对平移不敏感**，
故本脚本同时报告"平移是否改变全天合计"——用来区分
"只是列错位（合计不变）"与"连合计都改（漏填一格）"两种情况。
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

from q2_model import (T, MULT, E_FEB1, WARMUP_DAY, Policy,  # noqa: E402
                      load_data, forecast_net, make_plan, execute)

POL = Policy("S1", forecast="cquant", beta=0.70, W=14, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")


def rollout_shift(D, shift, start=WARMUP_DAY, E0=E_FEB1):
    """shift=0 为基线；+1 表示计划整体推后一个区间（首格空），-1 表示提前（末格空）。"""
    price, net, dates = D["price"], D["net"], D["dates"]
    nd = len(dates)
    E = float(E0)
    plan = em = 0.0
    tot_b = tot_bu = 0.0
    for d in range(start, nd):
        N_hat = forecast_net(POL, d, D)
        b, c_plan = make_plan(POL, price, N_hat, E)
        if shift == 0:
            bu = b
        elif shift > 0:                      # 计划落到晚一个区间
            bu = np.zeros(T)
            bu[shift:] = b[:-shift]
        else:                                # 计划落到早一个区间
            bu = np.zeros(T)
            bu[:shift] = b[-shift:]
        _c, _d, e, _w, Etraj = execute(POL, bu, c_plan, N_hat, net[d], price, E)
        plan += float((bu * price).sum())
        em += float(MULT * (e * price).sum())
        tot_b += float(b.sum())
        tot_bu += float(bu.sum())
        E = Etraj[-1]
    return dict(plan=plan, em=em, total=plan + em,
                tot_b=tot_b, tot_bu=tot_bu)


def main():
    D = load_data()
    dates = D["dates"]
    idx_eval = np.where(dates >= np.datetime64("2025-02-01"))[0]
    n_eval = len(idx_eval)

    print("=== O6 时间列映射歧义的代价（S1 主方案，评价期 2/1–12/31）===")
    print(f"    评价天数 {n_eval} 天；公共预热 E = {E_FEB1:.0f} kWh\n")

    res = {}
    for sh in (0, +1, -1):
        res[sh] = rollout_shift(D, sh)

    base = res[0]
    print(f"{'读法':<34}{'计划费/万元':>13}{'紧急费/万元':>13}{'总费用/万元':>13}{'相对基线/万元':>15}")
    print("-" * 90)
    names = {0: "基线（不错位）",
             +1: "错位 +1 格（计划晚一个区间）",
             -1: "错位 -1 格（计划早一个区间）"}
    for sh in (0, +1, -1):
        r = res[sh]
        d = r["total"] - base["total"]
        tag = "" if sh == 0 else f"{d/1e4:+.4f}"
        print(f"{names[sh]:<34}{r['plan']/1e4:>13.4f}{r['em']/1e4:>13.4f}"
              f"{r['total']/1e4:>13.4f}{tag:>15}")

    print("\n--- 全天合计列是否受影响（诊断：错位是‘纯列错位’还是‘漏填一格’）---")
    for sh in (0, +1, -1):
        r = res[sh]
        diff = r["tot_bu"] - r["tot_b"]
        print(f"  {names[sh]:<34} 平移后计划电量合计相对基线 {diff:+,.1f} kWh"
              f"（{diff/r['tot_b']*100:+.4f}%）")

    print("\n--- 结论口径 ---")
    for sh in (+1, -1):
        r = res[sh]
        print(f"  错位 {sh:+d} 格：总费用 {r['total']/1e4:.4f} 万元，"
              f"相对基线 {r['total']/1e4-base['total']/1e4:+.4f} 万元"
              f"（{r['total']/base['total']-1:+.3%}）；"
              f"其中紧急费 {r['em']/1e4:.4f}（{r['em']/1e4-base['em']/1e4:+.4f}）")
    print("\n注：本量化为‘整列连续错位一格’的情形；若只是首/末格漏填，")
    print("    影响远小于此，但列错位的方向性影响（多买/少买落在错区间）与此同量级。")


if __name__ == "__main__":
    main()
