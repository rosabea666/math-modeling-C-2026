# -*- coding: utf-8 -*-
"""
§12-7 预测构造阶梯：剩余空间有多少能从"预测侧"收回（只读）
==============================================================================
背景：§16 得 V0 与公平 PI 差 490.6059 万元；§17 证明期末/跨日结构只值 33.9957 万元
      ⇒ 剩余约 456.6 万元只能在"信息（日前预测）"这一侧。本轮把它继续切开：

  ① 可实施（因果）的预测构造变体——看"换一个更好的估计量"能拿回多少；
  ② 不可实施的诊断参照——看"需要多好的信息"才够。

统一不动的东西：策略 must/max、执行 greedy、公共预热 E_FEB1、评价期 2/1–12/31、
      评价边界一致；除注明外 β=0.70、W=14。

**重点怀疑（§16.2(d) 的线索）**：即使剔除分位数裕量，**中位基准本身**也比当日实测高
      （+26 ~ +99 kWh/区间）。且 §15 的"时间权重 ρ"只加权了**裕量**、**没有加权基准**，
      所以"基准的偏高"从未被针对性处理。本脚本专门测**只改 base** 的几种做法。
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

from q2_model import (T, MULT, E_FEB1, WARMUP_DAY,  # noqa: E402
                      Policy, load_data, make_plan, execute)

POL = Policy("S1", forecast="cquant", W=14, beta=0.70, exec_mode="greedy",
             discharge_policy="must", charge_policy="max")


def _col_quantile(X, qs):
    """按列线性插值分位数（与 np.quantile method='linear' 等价）。"""
    n, m = X.shape
    vs = np.sort(X, axis=0)
    pos = np.asarray(qs, float) * (n - 1.0)
    lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, n - 1)
    fr = pos - lo; cols = np.arange(m)
    return vs[lo, cols] * (1.0 - fr) + vs[hi, cols] * fr


def _wq_cols(X, q, rho):
    """按列计算**加权**分位数，权重 ∝ rho^(age)（age 越大越旧，越旧权重越小）。"""
    n, m = X.shape
    w = rho ** np.arange(n - 1, -1, -1)
    idx = np.argsort(X, axis=0)
    Xs = np.take_along_axis(X, idx, axis=0)
    Ws = np.take_along_axis(np.broadcast_to(w[:, None], X.shape).copy(), idx, axis=0)
    cw = np.cumsum(Ws, axis=0); cw = cw / cw[-1]
    out = np.empty(m)
    for t in range(m):
        out[t] = np.interp(q, cw[:, t], Xs[:, t])
    return out


def base_of(kind, d, D, W):
    """各种 base（中位基准/其它估计量）。"""
    hist = D["net"][max(0, d - W):d]
    if hist.shape[0] < 7:
        return D["net_ref"].copy(), None
    med = np.median(hist, axis=0)
    if kind == "median":
        return med.copy(), hist
    if kind == "wmedian":
        return _wq_cols(hist, 0.5, 0.8), hist
    if kind == "trend":
        dm = hist.mean(axis=1)
        shift = float(dm[-3:].mean() - dm[:-3].mean()) if len(dm) >= 7 else 0.0
        return med + shift, hist
    if kind == "persist":
        return D["net"][d - 1].copy(), hist
    raise ValueError(kind)


def forecast(d, D, kind, W, beta, level_ref=False, shape_ref=False, oracle=False):
    if oracle:
        return D["net"][d].copy()
    base, hist = base_of(kind, d, D, W)
    resid = hist - np.median(hist, axis=0)
    margin = np.maximum(_col_quantile(resid, np.full(T, beta)), 0.0)
    out = base + margin
    if level_ref:
        out = out + (D["net"][d].mean() - out.mean())
    if shape_ref:
        out = out.mean() + (D["net"][d] - D["net"][d].mean())
    return out


def run(D, kind, W, beta, idx, level_ref=False, shape_ref=False, oracle=False):
    price, net, dates = D["price"], D["net"], D["dates"]
    E = float(E_FEB1)
    dp = np.zeros(len(dates)); de = np.zeros(len(dates)); dk = np.zeros(len(dates))
    dw = np.zeros(len(dates)); dev = np.zeros(len(dates))
    for d in range(WARMUP_DAY, len(dates)):
        N_hat = forecast(d, D, kind, W, beta, level_ref, shape_ref, oracle)
        # 偏差 = 预测 − 实测（逐区间均值）
        dev[d] = float((N_hat - net[d]).mean())
        b, c_plan = make_plan(POL, price, N_hat, E)
        c, dd, e, w, Etraj = execute(POL, b, c_plan, N_hat, net[d], price, E)
        dp[d] = float((b * price).sum()); de[d] = float(MULT * (e * price).sum())
        dk[d] = float(e.sum()); dw[d] = float(w.sum())
        E = Etraj[-1]
    return dp, de, dk, dw, dev


def main():
    D = load_data()
    dates = D["dates"]
    idx = np.where(dates >= np.datetime64("2025-02-01"))[0]

    print("=" * 104)
    print("§12-7 预测构造阶梯（只读；策略/执行/预热/评价边界全部不动）")
    print("=" * 104)
    print(f"  评价期 {len(idx)} 天；β=0.70, W=14（除注明）；must/max greedy")

    specs = [
        ("① W14 中位基准（现状）", dict(kind="median", W=14, beta=0.70)),
        ("② W7 中位基准", dict(kind="median", W=7, beta=0.70)),
        ("③ W21 中位基准", dict(kind="median", W=21, beta=0.70)),
        ("④ base 时间加权 ρ=0.8", dict(kind="wmedian", W=14, beta=0.70)),
        ("⑤ base 趋势平移", dict(kind="trend", W=14, beta=0.70)),
        ("⑥ base=昨日实测", dict(kind="persist", W=14, beta=0.70)),
        ("⑦ W14 + β=0.85", dict(kind="median", W=14, beta=0.85)),
        ("[参照] 完美日总量+历史形状", dict(kind="median", W=14, beta=0.70, level_ref=True)),
        ("[参照] 历史水平+完美形状", dict(kind="median", W=14, beta=0.70, shape_ref=True)),
        ("[参照] 完美日前 PI", dict(kind="median", W=14, beta=0.70, oracle=True)),
    ]
    out = {}
    print(f"\n  {'预测构造':<26}{'计划/万元':>11}{'紧急/万元':>11}{'总费用/万元':>13}"
          f"{'紧急电量/kWh':>14}{'弃置/kWh':>12}{'平均偏差':>10}")
    print("  " + "-" * 98)
    for nm, kw in specs:
        dp, de, dk, dw, dev = run(D, idx=idx, **kw)
        tot = float(dp[idx].sum() + de[idx].sum())
        out[nm] = dict(dp=dp, de=de, dk=dk, dw=dw, dev=dev, tot=tot)
        print(f"  {nm:<26}{dp[idx].sum()/1e4:>11.4f}{de[idx].sum()/1e4:>11.4f}"
              f"{tot/1e4:>13.4f}{dk[idx].sum():>14.1f}{dw[idx].sum():>12.1f}"
              f"{dev[idx].mean():>+10.3f}")

    b0 = out["① W14 中位基准（现状）"]["tot"]
    pi = out["[参照] 完美日前 PI"]["tot"]
    gap = b0 - pi
    print(f"\n  基线校验：① = {b0/1e4:.4f} 万元（应为 1719.7084；"
          f"偏差 {b0/1e4-1719.7084:+.6f}）")
    print(f"  PI 差距 = {gap/1e4:.4f} 万元")

    print("\n  相对现状（①）的差额，及其对 PI 差距的收回比例：")
    print(f"    {'预测构造':<26}{'相对①/万元':>13}{'收回PI差距':>13}")
    print("    " + "-" * 52)
    for nm, _ in specs:
        t = out[nm]["tot"]
        print(f"    {nm:<26}{t/1e4-b0/1e4:>+13.4f}"
              f"{(b0-t)/gap*100:>12.1f}%")

    best_causal = min([nm for nm, _ in specs if not nm.startswith("[参照]")],
                      key=lambda k: out[k]["tot"])
    bt = out[best_causal]["tot"]
    print(f"\n  ⇒ 最好的**可实施**构造：{best_causal} = {bt/1e4:.4f} 万元，"
          f"相对现状 {bt/1e4-b0/1e4:+.4f} 万元，收回 PI 差距 {(b0-bt)/gap*100:.1f}%")
    print(f"  ⇒ 诊断参照（不可实施）：完美日总量+历史形状 收回 "
          f"{(b0-out['[参照] 完美日总量+历史形状']['tot'])/gap*100:.1f}%；"
          f"完美日前 PI 收回 100%")
    print("\n  判读：若‘可实施构造’全都只收回个位数百分比，而‘完美日总量’能收回一大截，")
    print("        则剩余费用的性质是**当日总量/形状的不可预知**，不是估计量选得不好——")
    print("        这正是‘不该继续调参’的正面证据。")


if __name__ == "__main__":
    main()
