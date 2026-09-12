# 项目长期记忆（2026 数模 C 题）

## 运行环境
- Python：`C:/Users/LZK/.workbuddy/binaries/python/envs/default/Scripts/python.exe`（3.13.14，numpy 2.5 / scipy 1.18 / pandas 3.0 / openpyxl / matplotlib 均有）。
- **bash 残缺**：无 ls/dirname/head/tail/rm；文件操作走 PowerShell 或 python 脚本。**PowerShell 工具不回显 stdout** → 需要看输出时用 Bash 调 python 打印。
- LP 求解器：`scipy.optimize.linprog(method="highs")`。

## 最终结果
- **Q1**：全天购电费 **35126.9486 元**，购电 59482.6990 kWh；无储能基准 48052.0466 元 ⇒ 节省 **26.90%**。
- **Q2 主方案 E（V_dow_N）**：评价期 2/1–12/31（334 天，起评 10800 kWh）总费 **1395.6995 万元**（计划 1318.2416 + 紧急 77.4578）。参数 β=0.70(2–3月)/0.75(4–12月)、W=7、γ=1.0、N_DOW=4，逐月 walk-forward 选取。
- Q2 风险档案：P99 单日紧急费 3.20 万元，最大单日 7.58 万元，大紧急 36/334，触底 113/334，触顶 327/334，总弃置 252.63 万 kWh。
- **公平 PI（完美日前预测）= 1229.10 万元，是正文唯一保留的 PI 参照**。旧 1224.50 已作废（其 SOC 起点 6000 ≠ 10800，不可比）。

## 项目结构（2026-09-12 整理后）
- 只留最终链路：`scripts/` 9 个（q1_solve、q1_analysis、q1_figures、q1_analysis_figures、q2_model、q2_improve、q2_dow_N_joint、q2_V_dow_N_profile、q2_finalize_E）；`results/` 9 个；`figures/` 23 个 svg；`utils/`（plot_style、fig_export）；`paper/` 两篇论文。
- 中间过程 98 个文件 + 23 张 png + A/B 题残留 + Q2模型.md/选题分析报告.md/使用指南.md 全部归档至 `_archive/`（scripts/results/docs/figures_png/misc/figures_previews），确认后可整体删除。
- 新增 `README.md`：结构、复现命令、关键口径。`utils/fig_export.py` 为新增绘图导出兼容层（原依赖的 math-modeling skill 的 setup_style/export_figure 本机不存在，已做 try/except 降级）。
- 所有保留脚本的 BASE 已改为 `Path(__file__).resolve().parent.parent`，本机可直接运行；q1/q2 全部复现通过（Q2 内置 assert 1395.6995 校验）。

## 纪律（用户硬性要求）
- **只读优先**：分析/诊断脚本不得改模型逻辑、不得重出 Excel；改动须先经用户确认（重跑前先备份 results）。
- **因果纪律**：选参只用当期之前数据（walk-forward），评价期只用于报告；"事后最优"须标注不可实施。
- **不夸大**：不写"穷尽""任意估值下排序不变""真正内点最优""方向已关闭"等越界论断。
- 同一文件**禁止并行 Edit**；f-string 不内嵌 ASCII 双引号；差异值由未舍入总量相减。

## 代码坑
- 分时段分位数必须用 `q2_three_way._col_quantile`，不能 `np.quantile(resid, betas, axis=0)`（会返回 (T,T) 矩阵）。
- 多候选 × 顺序仿真要先堆叠成 (J,S,T) 一次性前推，比朴素快约 13×。
- 重定向到文件的日志要 `line_buffering=True`，否则长时间空白。

## 未决
- **O6**：结果模板列槽位映射冲突（默认按列序 1:1 提交）。**O8**：kW→kWh 积分规则未消除，与 O6 独立。
- 外部方案 1356.64 万元未核验、本文档不归因。
