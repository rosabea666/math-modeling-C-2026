# -*- coding: utf-8 -*-
"""把Q1结果导出为CSV，供MATLAB绘图读取（电量口径，时间=区间末端时刻h）"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23")
OUT = BASE / "matlab" / "data"
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(BASE / r"results\q1_solution.npz")
DT = 1.0 / 6.0
t = (np.arange(144) + 1) * DT
df = pd.DataFrame({
    "t_h": t,
    "price_yuan_per_kwh": z["price"],
    "load_kw": z["load_kw"],
    "pv_kw": z["pv_kw"],
    "g_kwh": z["g"],
    "c_kwh": z["c"],
    "d_kwh": z["d"],
    "w_kwh": z["w"],
})
df.to_csv(OUT / "q1_interval.csv", index=False, float_format="%.6f")

soc = pd.DataFrame({"t_h": np.arange(145) * DT, "E_kwh": z["E"]})
soc.to_csv(OUT / "q1_soc.csv", index=False, float_format="%.6f")

with open(BASE / r"results\q1_analysis.json", encoding="utf-8") as f:
    A = json.load(f)
cap = pd.DataFrame(A["capacity_sensitivity"])[["window_kwh", "cost"]].rename(columns={"cost": "cost_cap"})
pow_ = pd.DataFrame(A["power_sensitivity"])[["p_kw", "cost"]].rename(columns={"cost": "cost_pow"})
sens = cap.join(pow_, how="outer")
sens.to_csv(OUT / "q1_sensitivity.csv", index=False, float_format="%.6f")

base = pd.DataFrame({
    "metric": ["cost_yuan", "purchase_kwh"],
    "no_storage": [A["baseline_no_storage"]["cost"], A["baseline_no_storage"]["purchase_kwh"]],
    "with_storage": [A["with_storage"]["cost"], A["with_storage"]["purchase_kwh"]],
})
base.to_csv(OUT / "q1_baseline.csv", index=False, float_format="%.6f")
print("已导出:", [p.name for p in OUT.glob("*.csv")])
