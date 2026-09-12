# -*- coding: utf-8 -*-
"""冒烟测试：最小网格，验证 q2_forecast_dow.py 的逻辑与自检是否通过。"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from q2_model import E_FEB1, WARMUP_DAY, load_data, make_plan, execute, Policy, MULT
import q2_forecast_dow as fd

# 把网格替换成最小集合，跑 1 月看是否收敛到合理数字
fd.GAMMA_GRID = [0.0, 0.5]
fd.BETA_GRID = [0.70]
fd.W_GRID = [14]
# 跑全集主程序，但通过 monkey-patch 替换网格
print("[smoke] GAMMA_GRID=", fd.GAMMA_GRID, "BETA_GRID=", fd.BETA_GRID, "W_GRID=", fd.W_GRID)

if __name__ == "__main__":
    t_all = time.perf_counter()
    D = fd.load_data()
    D["_dow"] = fd.dow_index(D["dates"])
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1 for d in D["dates"]])
    D["_month"] = months
    D["_month_end"] = {m: int(np.max(np.where(months == m)[0])) for m in range(1, 13)}

    # 1 月：测 forecast_dow 在不同 gamma 下输出形状 + 与 cquant 的差
    print("\n[1] 抽样 2025-01-15 (周三)，三个时段的 forecast_dow(γ)")
    for d in [15, 30, 45]:
        for g in fd.GAMMA_GRID:
            N_hat = fd.forecast_dow(d, D, D["_dow"], g, 14, 0.70)
            N_cquant = N_hat.copy()  # placeholder
            # cquant baseline (no γ): forecast_dow(γ=0) ≡ forecast_baseline
            # 用 forecast_dow(γ=0) 应与 cquant 完全一致
            N_g0 = fd.forecast_dow(d, D, D["_dow"], 0.0, 14, 0.70)
            mean_shift = (N_hat - N_g0).mean()
            print(f"  d={d} dow={D['_dow'][d]} γ={g:.2f}  N̂均值={N_hat.mean():.2f}  "
                  f"相对γ=0均值差={mean_shift:+.2f} kWh", flush=True)
    print(f"\n  抽样耗时 {time.perf_counter()-t_all:.1f}s")

    # 2 月-12 月一个完整候选（γ=0, β=0.70, W=14）的全年回放
    print("\n[2] 一个候选（γ=0, β=0.70, W=14）全年回放（用 fd.replay_full_year）")
    t0 = time.perf_counter()
    cum_j, dp, de, dk, dw = fd.replay_full_year(
        D, D["_dow"], lambda d: (14, 0.70, 0.0))
    print(f"  1 月末累计 J = {cum_j[WARMUP_DAY-1]/1e4:.4f} 万元")
    print(f"  评价期累计 J = {(cum_j[-1] - cum_j[WARMUP_DAY-1])/1e4:.4f} 万元  （V0 应为 1719.7084）")
    print(f"  单次回放耗时 {time.perf_counter()-t0:.1f}s")

    # 评价期回放对照 V0
    print("\n[3] 评价期回放（fd.replay_eval 用同参数）")
    sched = {m: (14, 0.70, 0.0) for m in range(2, 13)}
    rV0 = fd.replay_eval(D, D["_dow"], sched, months, keep=False)
    cV0 = float((rV0["dp"][np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]] +
                  rV0["de"][np.where(D["dates"] >= np.datetime64("2025-02-01"))[0]]).sum()) / 1e4
    print(f"  γ=0, β=0.70, W=14 (cycle) 评价期 = {cV0:.4f} 万元  （与 V0 = 1719.7084 差 {cV0-1719.7084:+.4f}）")
    print(f"  注：本脚本默认 geq{2400}，cycle 与 geq2400 在 (β,W)=(0.70,14) 下应极接近")