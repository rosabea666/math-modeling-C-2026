# 2026 数学建模 C 题 · 微网与外部电网电力调控策略

> 本仓库只保留 **C 题** 的最终结果链路：问题一（Q1）+ 问题二（Q2）。
> 中间改进过程（40+ 个探索性实验脚本及其产物、A/B 题残留、png 副本）已归入 `_archive/`，确认无误后可整体删除。
hhhh
---

## 一、最终结果

### 问题一（代表日确定性优化）

| 指标 | 数值 |
|---|---|
| 全天购电费用 | **35126.9486 元** |
| 全天购电量 | 59482.6990 kWh |
| 无储能基准费用 | 48052.0466 元 |
| **节省率** | **26.90%**（−12925.0980 元） |
| 求解规模 | 721 变量 / 290 等式约束，HiGHS 367 次迭代 |
| 校验 | 能量平衡与 SOC 递推残差 0，SOC ∈ [1200, 10800]，E₀ = E₁₄₄ = 6000，充放电重叠 0 → **PASS** |

### 问题二（全年序贯决策，主方案 E = V_dow_N）

| 指标 | 数值 |
|---|---|
| 评价期 | 2025.2.1–12.31，334 天，2/1 起评储能 10800 kWh |
| **总费用** | **1395.6995 万元** |
| 计划费用 / 紧急费用 | 1318.2416 / 77.4578 万元 |
| 预测器 | 近期净负载水平（W=7）+ 星期偏差校正（γ=1.0，N=4）+ 分位数裕量（β=0.70 月2–3 / 0.75 月4–12） |
| 计划层 | 日前 LP（HiGHS），显式日循环边界，不设 2400 下限 |
| 执行层 | 尽限充放电（must/max），SOC 跨日连续，缺口按 5 倍电价紧急购电 |
| 风险 | P99 单日紧急费 3.20 万元，最大单日 7.58 万元，大紧急日 36/334，总弃置 252.63 万 kWh |

指定日期结果见 `results/q2_E_tables.json`（表1 计划购电 / 表2 充放电与日初日末 / 表3 紧急购电）。

---

## 二、目录结构

```
.
├── paper/
│   ├── Q1论文初稿.md        # 问题一论文
│   └── Q2论文初稿.md        # 问题二论文（主方案 E）
├── results/
│   ├── result1.xlsx         # 问题一提交表
│   ├── result2.xlsx         # 问题二提交表（按主方案 E 生成）
│   ├── q1_solution.npz      # Q1 逐区间解
│   ├── q1_summary.json      # Q1 汇总
│   ├── q1_analysis.json     # Q1 基准对比与敏感性
│   ├── q2_E_tables.json     # Q2 论文表1/2/3 数据
│   ├── q2_V_dow_N_profile.json  # Q2 月度成本与风险档案
│   ├── q2_dow_N_joint.json  # Q2 主方案选参（walk-forward）结果
│   └── Q1模型说明与证明.md
├── scripts/                 # 见下表
├── utils/
│   ├── plot_style.py        # 出版级样式（向后兼容层）
│   └── fig_export.py        # 图形导出兼容层（skill 不在本机时降级）
├── figures/                 # 论文配图（SVG）
├── data/                    # 附件 1–4 + 附件 5 结果模板
├── C题.pdf / problem_C.txt
└── _archive/                # 中间过程归档，可删除
```

---

## 三、脚本清单（最终链路）

| 脚本 | 作用 | 输出 |
|---|---|---|
| `q1_solve.py` | Q1 主求解（LP 装配 + 独立校验 + 写表） | `result1.xlsx`、`q1_solution.npz`、`q1_summary.json` |
| `q1_analysis.py` | Q1 无储能基准、容量/功率敏感性、边际价值 | `q1_analysis.json` |
| `q1_figures.py` | Q1 数据/过程/结果三类图 | `figures/*.svg` |
| `q1_analysis_figures.py` | Q1 基准对比与敏感性图 | `figures/*.svg` |
| `q2_model.py` | **模型内核**：数据装载、日前计划 LP、执行层仿真、结算 | 被以下脚本调用 |
| `q2_improve.py` | 对照策略 A/B/B48/C/D 与紧急事件合并工具 | 被 `q2_finalize_E.py` 调用 |
| `q2_dow_N_joint.py` | 主方案 E 选参：(β, W, γ, N) 36 候选联合 walk-forward | `q2_dow_N_joint.json` |
| `q2_V_dow_N_profile.py` | 主方案 E 月度费用与风险档案 | `q2_V_dow_N_profile.json` |
| `q2_finalize_E.py` | **主方案唯一入口**：复现总费 + 论文表1/2/3 + 重写 result2.xlsx | `result2.xlsx`、`q2_E_tables.json` |

---

## 四、复现

环境：Python 3.13，需 `numpy / scipy / pandas / openpyxl / matplotlib`；LP 求解器为 `scipy.optimize.linprog(method="highs")`。

```bash
python scripts/q1_solve.py        # 问题一：费用 35126.9486 元
python scripts/q1_analysis.py     # 问题一：基准与敏感性
python scripts/q2_finalize_E.py   # 问题二：总费 1395.6995 万元（内置 assert 校验）
python scripts/q2_V_dow_N_profile.py  # 问题二：月度与风险档案
python scripts/q2_dow_N_joint.py      # 问题二：选参过程（约 90 s）
```

所有脚本以自身位置推导项目根目录（不再依赖绝对路径），可在任意机器直接运行。

---

## 五、关键建模口径（易错点）

1. **因果性**：第 d 天决策只用第 1…d−1 天实测；超参数按**逐月 walk-forward** 选取，不得使用未来数据。
2. **冷启动**：1 月为公共预热轨迹，`d < 30` 时预测器回退代表日净负载曲线；评价期自 2/1 起算。
3. **计划即结算**：`b_t` 一经制定即付费，未消纳部分弃置，不售电；缺口只能靠储能重排或 5 倍电价紧急购电。
4. **两条 LP 链路并存**：`q2_model.make_plan` 为日循环链路（主方案 E），`q2_improve` 中的对照链路另有期末下限设定，二者数值**不可直接相减比较**。
5. **单位**：功率 kW × Δt(1/6 h) = 电量 kWh；费用除以 1e4 为万元。
