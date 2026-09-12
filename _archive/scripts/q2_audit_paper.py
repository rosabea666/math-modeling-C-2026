# -*- coding: utf-8 -*-
"""Q2 论文初稿数值独立复核（只读，不改动任何结果文件）。
逐项重算论文中未写入 q2_strategies.json 的数值，并打印比对结论。"""
import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23")
DT = 1.0 / 6.0
T = 144
EVAL0 = 31  # 2025-02-01 的日期索引（1月31天）


def load_price():
    a1 = pd.read_excel(BASE / r"data\附件1.xlsx", sheet_name=0, header=0)
    return a1


a1 = load_price()
print("附件1 形状:", a1.shape, "| 列名:", list(a1.columns))
price_col = a1.iloc[:, 1].to_numpy(float)
print(f"附件1 电价列长度 = {len(price_col)} (T={T})")
print(f"电价范围: min={price_col.min():.4f} max={price_col.max():.4f} 均值={price_col.mean():.4f}")
print(f"电价分位: 10%={np.quantile(price_col,0.1):.4f} 50%={np.quantile(price_col,0.5):.4f} 90%={np.quantile(price_col,0.9):.4f}")
p = price_col[:T]

ld = pd.read_excel(pd.ExcelFile(BASE / r"data\附件2.xlsx"), sheet_name=0)
dates = pd.to_datetime(ld.iloc[:, 0])
print(f"\n附件2 天数 = {len(dates)}, {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}")
print(f"附件2 数据列数 = {ld.shape[1]-1}，末列标签 = {ld.columns[-1]!r}，首列标签 = {ld.columns[1]!r}")

# ---------- 1. 2月1日 / 12月31日 储能状态 ----------
print("\n=== [1] 年末/2月1日 实际储能（论文 7.2 表）===")
print(f"{'策略':>5} {'2月1日0:00':>14} {'12月31日24:00':>15}  论文值对照")
paper_y = {"A": (10800.0, 10500.6480), "B": (10800.0, 10129.6112), "B48": (10800.0, 8275.5791),
           "C": (10800.0, 10452.8234), "D": (10800.0, 10696.9885), "PI": (6000.0, 6000.0)}
soc = {}
for name in ("A", "B", "B48", "C", "D", "PI"):
    z = np.load(BASE / rf"results\q2_strategy_{name}.npz")
    s0, s1 = z["SOC0"], z["SOC1"]
    soc[name] = (s0, s1, z)
    ref = paper_y[name]
    f1 = "OK" if abs(s1[EVAL0 - 1] - ref[0]) < 1e-3 else "**不符**"
    f2 = "OK" if abs(s1[-1] - ref[1]) < 1e-3 else "**不符**"
    print(f"{name:>5} {s1[EVAL0-1]:14.4f} {s1[-1]:15.4f}  [{f1} / {f2}]")
    # 跨日连续
    assert np.max(np.abs(s1[:-1] - s0[1:])) < 1e-9, name

# ---------- 2. 表1a 指定区间计划购电量（策略D） ----------
print("\n=== [2] 表1a 指定区间计划购电量（D，t=60/72/84/96/108/120）===")
paper_1a = {
    "2025-03-20": [0.0000, 553.1281, 0.0000, 522.3932, 686.6784, 0.0000],
    "2025-06-21": [0.0000, 700.0382, 0.0000, 610.2398, 691.5773, 50.4318],
    "2025-09-23": [0.0000, 412.6429, 0.0000, 473.1142, 777.5356, 0.0000],
    "2025-12-21": [0.0000, 932.2482, 0.0000, 830.3562, 715.4787, 0.0000],
}
BX = soc["D"][2]["B"]
ts = [60, 72, 84, 96, 108, 120]
for ds, ref in paper_1a.items():
    d = int(np.where(dates == pd.Timestamp(ds))[0][0])
    got = [BX[d, t] for t in ts]
    flag = all(abs(g - r) < 5e-4 for g, r in zip(got, ref))
    print(f"  {ds}: {['%.4f' % g for g in got]}  [{'OK' if flag else '**不符**'}]")

# ---------- 3. 表1b 全天电量与费用（策略D） ----------
print("\n=== [3] 表1b 全天电量与费用（D）===")
paper_1b = {
    "2025-03-20": (68145.2845, 699.6736, 68844.9580, 41172.4079, 2960.0603, 44132.4682),
    "2025-06-21": (76818.4370, 0.0000, 76818.4370, 47420.2720, 0.0000, 47420.2720),
    "2025-09-23": (63360.3155, 3543.2349, 66903.5504, 39217.6622, 18285.6063, 57503.2685),
    "2025-12-21": (93476.0765, 1416.8827, 94892.9592, 59523.5142, 7132.0291, 66655.5433),
}
for ds, ref in paper_1b.items():
    d = int(np.where(dates == pd.Timestamp(ds))[0][0])
    plan_q = BX[d].sum()
    em_q = soc["D"][2]["EM"][d].sum()
    plan_c = (BX[d] * p).sum()
    em_c = 5.0 * (soc["D"][2]["EM"][d] * p).sum()
    got = (plan_q, em_q, plan_q + em_q, plan_c, em_c, plan_c + em_c)
    dev = max(abs(g - r) for g, r in zip(got, ref))
    print(f"  {ds}: " + " | ".join(f"{g:.4f}" for g in got) + f"  [最大偏差 {dev:.2e}]")

# ---------- 4. 表2a 四小时充放电（D） ----------
print("\n=== [4] 表2a 四小时区间实际充放电（D）===")
paper_2a = {
    "2025-03-20": [(0,0),(0,5295.1138),(6051.6360,2716.2514),(4153.4109,254.7228),(0,5749.8292),(10628.4763,2877.2999)],
    "2025-06-21": [(0,0),(0,1382.9190),(1707.3075,0),(0,0),(0,3840.7538),(6905.8800,1753.0089)],
    "2025-09-23": [(0,0),(106.6149,6701.5230),(3763.5384,1714.7810),(5370.4495,80.8719),(0,5548.9992),(10470.7635,1979.1009)],
    "2025-12-21": [(0,0),(831.3860,1506.7560),(5144.7816,7740.8316),(7590.1584,2211.0497),(28.9743,5700.1499),(10581.5048,2424.4250)],
}
CW, DW = soc["D"][2]["C"], soc["D"][2]["D"]
for ds, ref in paper_2a.items():
    d = int(np.where(dates == pd.Timestamp(ds))[0][0])
    dev = 0.0
    for bi, (rc, rd) in enumerate(ref):
        gc = CW[d, bi*24:(bi+1)*24].sum(); gd = DW[d, bi*24:(bi+1)*24].sum()
        dev = max(dev, abs(gc - rc), abs(gd - rd))
    print(f"  {ds}: 最大偏差 {dev:.2e}  [{'OK' if dev < 5e-4 else '**不符**'}]")

# ---------- 5. 表2b 日初日末储能（D） ----------
print("\n=== [5] 表2b 日初/日末实际储能（D）===")
paper_2b = {"2025-03-20": (10800.0, 10779.9297), "2025-06-21": (10800.0, 10800.0),
            "2025-09-23": (10704.0466, 10638.4140), "2025-12-21": (10800.0, 10800.0)}
for ds, ref in paper_2b.items():
    d = int(np.where(dates == pd.Timestamp(ds))[0][0])
    got = (soc["D"][0][d], soc["D"][1][d])
    dev = max(abs(g - r) for g, r in zip(got, ref))
    print(f"  {ds}: {got[0]:.4f} / {got[1]:.4f}  偏差 {dev:.2e}")

# ---------- 6. 表3 紧急事件（D 还是 C？） ----------
print("\n=== [6] 表3 紧急事件：论文第7.5节标题称'由本次D策略数组生成'，但第6节表3说明属C ===")
TOL = 1e-6
def events(e):
    out, t = [], 0
    while t < T:
        if e[t] > TOL:
            s = t; tot = 0.0
            while t < T and e[t] > TOL:
                tot += e[t]; t += 1
            out.append((s, t, tot))
        else:
            t += 1
    return out
for name in ("C", "D"):
    EMx = soc[name][2]["EM"]
    print(f"  --- 策略 {name} ---")
    for ds in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
        d = int(np.where(dates == pd.Timestamp(ds))[0][0])
        ev = events(EMx[d])
        print(f"    {ds}: {len(ev)} 事件, 合计 {EMx[d].sum():.4f} kWh")

# ---------- 7. C/D 计划数组累计绝对差 ----------
print("\n=== [7] C 与 D 计划数组（b）差异 ===")
BC = soc["C"][2]["B"]
diff = np.abs(BC - BX)
print(f"  评价期累计绝对差 = {diff[EVAL0:].sum():.4f} kWh  (论文 66930.7070)")
print(f"  评价期净差（Σ符号差） = {(BX[EVAL0:] - BC[EVAL0:]).sum():.4f} kWh")
print(f"  评价期计划总量 C = {BC[EVAL0:].sum():.2f}, D = {BX[EVAL0:].sum():.2f}")

# ---------- 8. D 的日末<日初 天数 ----------
print("\n=== [8] D 实际日末 < 日初 的天数 ===")
s0d, s1d = soc["D"][0][EVAL0:], soc["D"][1][EVAL0:]
mask = s1d < s0d
print(f"  天数 = {mask.sum()} / {len(s0d)}  (论文 137/334)")
print(f"  最小差 = {(s1d - s0d).min():.4f} kWh  (论文 -1288.9649)")
print(f"  最大差 = {(s1d - s0d).max():.4f} kWh")

# ---------- 9. 紧急电量加权普通电价均值 ----------
print("\n=== [9] 以紧急电量为权重的普通电价均值（7.4节）===")
for name in ("C", "D"):
    EMx = soc[name][2]["EM"][EVAL0:]
    wavg = (EMx * p).sum() / EMx.sum()
    print(f"  {name}: {wavg:.4f} 元/kWh   论文 C=1.2486 / D=0.9674")

# ---------- 10. D 的紧急事件低价占比 ----------
print("\n=== [10] D 与 C 紧急购电的时段价格分布 ===")
for name in ("C", "D"):
    EMx = soc[name][2]["EM"][EVAL0:]
    tot = EMx.sum()
    wavg = (EMx * p).sum() / tot
    print(f"  {name}: 紧急电量 {tot:.1f} kWh, 加权均价 *5 = {5*wavg:.4f} 元/kWh")

print("\n复核结束（只读，未改动任何结果文件）")
