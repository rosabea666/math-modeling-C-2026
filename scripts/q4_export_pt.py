# -*- coding: utf-8 -*-
"""导出问题二预测器的中心预测序列 F_pt.csv，供问题四代码复现使用。

F_pt[d] = m_recent(d) + gamma * delta_dow(d)   （不含分位裕量，裕量由 Q4 代码自行施加）

只读，不改任何模型；输出 results/q4_v3/F_pt.csv。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from q2_model import T, load_data                                    # noqa: E402
from q2_finalize_E import dow_index, m_recent, delta_dow, SCHED      # noqa: E402

BASE = Path(__file__).resolve().parent.parent


def main():
    D = load_data()
    dow_idx = dow_index(D["dates"])
    months = np.array([d.astype("datetime64[M]").astype(int) % 12 + 1
                       for d in D["dates"]])
    nd = len(D["dates"])
    Fpt = np.zeros((nd, T))
    for d in range(nd):
        if d < 30:
            Fpt[d] = D["net_ref"]
            continue
        beta, W_level, gamma, n_dow = SCHED.get(months[d], (0.70, 7, 1.0, 4))
        m = m_recent(d, D["net"], W_level)
        dlt = delta_dow(d, D["net"], dow_idx, n_dow, W_level)
        Fpt[d] = m + gamma * dlt          # 中心预测，不含裕量

    out = BASE / "results" / "q4_v3"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Fpt).to_csv(out / "F_pt.csv", index=False, header=False)
    print(f"导出 {out / 'F_pt.csv'}  形状 {Fpt.shape}")
    print(f"  评价期(2/1起)中心预测均值 {Fpt[31:].mean():.2f} kWh/区间")


if __name__ == "__main__":
    main()
