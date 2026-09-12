# CUMCM-C

## 问题一

### 一、LP建模与求解

```python
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

DT = 1.0 / 6.0
T = 144
ETA_C = ETA_D = 0.9
E_MIN, E_MAX = 1200.0, 10800.0
Q_MAX = 5000.0 * DT
E0 = 6000.0


def solve_q1(price, L, V):
    """输入电价（元/kWh）、负载电量和光伏电量（kWh）。"""
    price = np.asarray(price, dtype=float)
    L = np.asarray(L, dtype=float)
    V = np.asarray(V, dtype=float)

    for name, values in (("price", price), ("L", L), ("V", V)):
        if values.shape != (T,) or not np.isfinite(values).all():
            raise ValueError(f"{name}须包含144个有限数值")

    if np.any(price < 0):
        raise ValueError("当前模型不支持负电价下的无限购电与余量处置")
    if np.any(L < 0) or np.any(V < 0):
        raise ValueError("负载电量和光伏电量不能为负")

    g = slice(0, T)
    c = slice(T, 2 * T)
    d = slice(2 * T, 3 * T)
    w = slice(3 * T, 4 * T)
    E = slice(4 * T, 5 * T + 1)

    n = 5 * T + 1
    objective = np.zeros(n)
    objective[g] = price

    A = lil_matrix((2 * T + 2, n))
    b = np.zeros(2 * T + 2)

    for t in range(T):
        A[t, E.start + t + 1] = 1.0
        A[t, E.start + t] = -1.0
        A[t, c.start + t] = -ETA_C
        A[t, d.start + t] = 1.0 / ETA_D

        row = T + t
        A[row, g.start + t] = 1.0
        A[row, d.start + t] = 1.0
        A[row, c.start + t] = -1.0
        A[row, w.start + t] = -1.0
        b[row] = L[t] - V[t]

    A[2 * T, E.start] = 1.0
    A[2 * T + 1, E.stop - 1] = 1.0
    b[2 * T:] = E0

    bounds = (
        [(0.0, None)] * T
        + [(0.0, Q_MAX)] * T
        + [(0.0, Q_MAX)] * T
        + [(0.0, None)] * T
        + [(E_MIN, E_MAX)] * (T + 1)
    )

    result = linprog(
        objective,
        A_eq=A.tocsr(),
        b_eq=b,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"求解失败：{result.message}")

    x = result.x
    return {
        "g": x[g],
        "c": x[c],
        "d": x[d],
        "w": x[w],
        "E": x[E],
        "cost": float(result.fun),
    }
```

### 二、互斥校验+独立复核

```python
def validate(sol, price, L, V):
    g = sol["g"]
    c = sol["c"]
    d = sol["d"]
    w = sol["w"]
    E = sol["E"]

    balance_error = np.max(np.abs(g + V + d - L - c - w))
    storage_error = np.max(
        np.abs(E[1:] - E[:-1] - 0.9 * c + d / 0.9)
    )

    return {
        "能量平衡最大残差": float(balance_error),
        "SOC递推最大残差": float(storage_error),
        "E最小/最大": (float(E.min()), float(E.max())),
        "同时充放电最大重叠": float(np.minimum(c, d).max()),
        "费用复算": float(price @ g),
    }
```

### 三、无储能基准与敏感性

```python
g0 = np.maximum(L - V, 0.0)
C0 = float(price @ g0)
saving = (C0 - C1) / C0

for h in (60, 120, 480, 1200):
    C_h = solve(
        price,
        L,
        V,
        e_min=1200,
        e_max=10800 + h,
        p_kw=5000,
    )["cost"]
    MV_plus = (C1 - C_h) / h
```

