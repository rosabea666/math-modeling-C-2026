# -*- coding: utf-8 -*-
"""
§12-6 检验：日循环期末结构是否结构性瓶颈（只读，除一处向后兼容的 LP 下限）
==============================================================================
§16.5 提出怀疑：计划 LP 强制"每日期末回到起始电量"（E_H = E_0）可能锁死费用。
本脚本**只改期末/跨日结构，其余完全不变**（β=0.70、W=14、must/max greedy、
公共预热 E_FEB1 = 10800 = E_MAX、评价期 2/1–12/31、同一执行层），对比两类放松：

  ① 跨日结转：把规划窗口拉长到 H 天、期末等式只加在**窗口末端**，只执行第 1 天。
        A1  H=2 ； A2  H=3
     （后续日沿用当日 N̂，**不引入额外信息**，否则等于跨日完美信息，结果不可实施）

  ② 期末回充水平：H=1、期末**不设等式**，只要求 E_H ≥ 下限 L，扫描 L。
        A3  L = 1200 (E_MIN，完全自由) ； A4  L = 3240 ； A5  L = 6480 ； A6  L = 9720

  基线 A0 = H=1 且期末 E_H = E_start（现状），必须复现 1719.7084。

判读：若 ① 与 ② 的费用都几乎不动 ⇒ **期末结构不是瓶颈**，瓶颈在预测（与 §16 的
      490.61 万元 PI 差距一致）；若某变体显著更低 ⇒ 找到新的可实施杠杆。
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

from q2_model import (T, MULT, E_MIN, E_MAX, E_FEB1, WARMUP_DAY,  # noqa: E402
                      Policy, load_data, solve_lp, forecast_net, execute)

POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")


def plan_H(price, N_hat, E_start, H, terminal, floor=None):
    """H 日窗口计划；只返回第 1 天。floor 仅在 terminal='geq' 时生效。"""
    if H == 1:
        sol = solve_lp(price, N_hat, E_start, mode="plan", terminal=terminal,
                       e_end_min=floor)
    else:
        sol = solve_lp(np.tile(price, H), np.tile(N_hat, H), E_start,
                       mode="plan", terminal=terminal, e_end_min=floor)
    return sol["b"][:T], sol["c"][:T]


def run(D, H, terminal, floor, idx, keep=False):
    price, net, dates = D["price"], D["net"], D["dates"]
    E = float(E_FEB1)
    dp = np.zeros(len(dates)); de = np.zeros(len(dates))
    dk = np.zeros(len(dates)); dw = np.zeros(len(dates))
    s_end = np.zeros(len(dates)); s_start = np.zeros(len(dates))
    det = dict(B=np.zeros((len(dates), T))) if keep else None
    for d in range(WARMUP_DAY, len(dates)):
        N_hat = forecast_net(POL, d, D)
        b, c_plan = plan_H(price, N_hat, E, H, terminal, floor)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        s_start[d] = E; s_end[d] = Etraj[-1]
        if keep:
            det["B"][d] = b
        E = Etraj[-1]
    return dict(dp=dp, de=de, dk=dk, dw=dw, s_start=s_start, s_end=s_end, det=det)


def main():
    D = load_data()
    dates = D["dates"]
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§12-6 期末结构检验（只改期末/跨日结构，其余完全不变）")
    print("=" * 104)
    print(f"  评价期 {len(idx)} 天；公共预热 E = {E_FEB1:.0f} kWh；"
          f"区间 [{E_MIN:.0f}, {E_MAX:.0f}]；β=0.70, W=14, must/max")
    print(f"  注意：E_FEB1 = {E_FEB1:.0f} = E_MAX ⇒ 现状等价于"
          f"**每天从满电开始、且必须回到满电**。")

    vars_ = [("A0 日循环 H=1（现状）", 1, "cycle", None),
             ("A1 跨日结转 H=2", 2, "cycle", None),
             ("A2 跨日结转 H=3", 3, "cycle", None),
             ("A3 期末自由 ≥1200", 1, "geq", 1200.0),
             ("A4 期末 ≥3240", 1, "geq", 3240.0),
             ("A5 期末 ≥6480", 1, "geq", 6480.0),
             ("A6 期末 ≥9720", 1, "geq", 9720.0)]
    res = {}
    for nm, H, tm, fl in vars_:
        r = run(D, H, tm, fl, idx, keep=True)
        res[nm] = r
        tot = float(r["dp"][idx].sum() + r["de"][idx].sum())
        print(f"  {nm:<22} 计划 {r['dp'][idx].sum()/1e4:>10.4f}  紧急 "
              f"{r['de'][idx].sum()/1e4:>9.4f}  总 {tot/1e4:>10.4f} 万元  "
              f"紧急电量 {r['dk'][idx].sum():>10.1f}  弃置 {r['dw'][idx].sum():>12.1f}")

    b = res["A0 日循环 H=1（现状）"]
    btot = float(b["dp"][idx].sum() + b["de"][idx].sum())
    print(f"\n  基线校验：A0 = {btot/1e4:.4f} 万元  （应为 1719.7084；"
          f"偏差 {btot/1e4-1719.7084:+.6f}）")

    print("\n  相对 A0 的差额与分解：")
    for nm, H, tm, fl in vars_:
        r = res[nm]
        tot = float(r["dp"][idx].sum() + r["de"][idx].sum())
        dpl = float(r["dp"][idx].sum() - b["dp"][idx].sum())
        dem = float(r["de"][idx].sum() - b["de"][idx].sum())
        print(f"    {nm:<22} 总 {tot/1e4-btot/1e4:>+9.4f} 万元"
              f"（计划 {dpl/1e4:>+8.4f}，紧急 {dem/1e4:>+8.4f}）")

    print("\n  诊断 1：日初/日末 SOC 与计划向量差异")
    print(f"    {'变体':<22}{'日初SOC均值':>12}{'日末SOC均值':>12}"
          f"{'日末标准差':>11}{'日末=E_MAX天数':>14}{'‖b−b_A0‖/‖b_A0‖':>18}")
    for nm, H, tm, fl in vars_:
        r = res[nm]
        se = r["s_end"][idx]; ss = r["s_start"][idx]
        num = float(np.linalg.norm(r["det"]["B"][idx] - b["det"]["B"][idx]))
        den = float(np.linalg.norm(b["det"]["B"][idx]))
        nf = int(np.sum(np.isclose(se, E_MAX, atol=1e-6)))
        print(f"    {nm:<22}{ss.mean():>12.1f}{se.mean():>12.1f}"
              f"{se.std():>11.1f}{nf:>14d}{num/den:>18.6f}")

    print("\n  诊断 2：跨日结转是否真的发生（A1/A2 的日末 SOC 偏离 E_MAX 的天数）")
    for nm in ("A1 跨日结转 H=2", "A2 跨日结转 H=3"):
        se = res[nm]["s_end"][idx]
        n = int(np.sum(np.abs(se - E_MAX) > 1.0))
        print(f"    {nm:<22} 偏离超 1 kWh 的天数 {n:>4d} / {len(idx)}"
              f"；日末 SOC 中位 {np.median(se):.1f}")

    print("\n  诊断 3：期末下限扫描下的弃置量（看是否用‘腾空间’换掉弃置）")
    for nm, H, tm, fl in vars_:
        print(f"    {nm:<22} 弃置 {res[nm]['dw'][idx].sum():>12.1f} kWh"
              f"（{(res[nm]['dw'][idx].sum()-b['dw'][idx].sum())/b['dw'][idx].sum()*100:>+6.1f}%"
              f" vs A0）")

    print("\n  诊断 4：降低期末下限是否**恶化尾部风险**（A0 满电循环 vs 各下限）")
    print(f"    {'变体':<22}{'紧急>1000kWh天数':>16}{'单日最大紧急费/万元':>19}"
          f"{'紧急费P99/万元':>15}{'平均日首储备/kWh':>18}")
    for nm, H, tm, fl in vars_:
        r = res[nm]
        nbig = int((r["dk"][idx] > 1000).sum())
        print(f"    {nm:<22}{nbig:>16d}{r['de'][idx].max()/1e4:>19.4f}"
              f"{np.percentile(r['de'][idx], 99)/1e4:>15.4f}"
              f"{r['s_start'][idx].mean():>18.1f}")
    print("    判读：若下限降低后‘大紧急天数/单日最坏’没有变差，说明"
          "**降低储备并未把风险转移到尾部**；")
    print("    但储备被长期维持在低位，本身就是**对预测误差更少的缓冲**——"
          "须在报告中作为风险取舍说明，不能只报费用下降。")

    print("\n  [参照] §16 公平 PI（完美日前预测）= 1229.1025 万元；"
          f"V0 与 PI 差距 490.6059 万元")
    best = min(vars_, key=lambda v: float(res[v[0]]["dp"][idx].sum()
                                          + res[v[0]]["de"][idx].sum()))
    bt = float(res[best[0]]["dp"][idx].sum() + res[best[0]]["de"][idx].sum())
    print(f"  ⇒ 最好变体 {best[0]} = {bt/1e4:.4f} 万元，"
          f"相对 A0 {bt/1e4-btot/1e4:+.4f} 万元，"
          f"收回 PI 差距的 {(btot-bt)/490.6059e4*100:.1f}%")
    print("  判读：若各变体与 A0 相差都在万元级以下 ⇒ **期末结构不是瓶颈**，"
          "瓶颈在预测（与 §16 的 PI 差距一致）。")


if __name__ == "__main__":
    main()
