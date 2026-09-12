# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from pathlib import Path
BASE = Path(r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23")
z = np.load(BASE / r"results\q2_solution.npz")
B, C, D, EM, W = z["B"], z["C"], z["D"], z["EM"], z["W"]
load_act, pv_act = z["load_act"], z["pv_act"]
res = B + pv_act + D - load_act - C - W - EM
i = np.unravel_index(np.argmax(np.abs(res)), res.shape)
day, t = i
print("max residual:", res[i], "day", day, "t", t)
print("b,V,d,L,c,w,e =", B[day,t], pv_act[day,t], D[day,t], load_act[day,t], C[day,t], W[day,t], EM[day,t])
net = B[day,t] + pv_act[day,t] - load_act[day,t]
print("net =", net)
print("SOC0[day]=", z["SOC0"][day], "E at t:", z["SOC0"][day] + (0.9*C[day,:t]-D[day,:t]/0.9).sum())
# 检查多少区间残差显著
big = np.abs(res) > 1e-6
print("显著残差区间数:", big.sum(), " 其中 net>=0:", (big & (B+pv_act-load_act>=0)).sum())
