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

## 论文（LaTeX，CUMCM2026-Template/）
- **模块化结构（2026-09-12 重构）**：`main.tex`（65 行）= 文档类 + 元数据 + `\input` 装配；正文在 `sections/00-abstract … 09-evaluation.tex`（序号 = `\section` 顺序）；导言区在 `preamble/{packages,fonts,layout}.tex`；参考文献与附录在 `backmatter/`。**加章节 = 新建文件 + 在 main.tex 登记**（用 `\input` 不用 `\include`，避免插页）。
- **编译**：`cd CUMCM2026-Template && latexmk main.tex`（`.latexmkrc` 已设 `$pdf_mode=5` + `$bibtex_use=0`，参考文献是手工 `thebibliography`，**不用 bibtex/biber**）。latexmk 需把 `C:/texlive/2026/bin/windows` 加进 PATH。`main_template_example.tex` = 原模板示例。**minted 在本机不可用**，附录用 fancyvrb 的 `\VerbatimInput`。
- **标签规范**：`sec:|fig:|tab:|eq:|app:` + `qN-` + 小写连字符；正文一律 `\ref`/`\eqref`/`\S\ref`，不写死编号。`\graphicspath{{figures/}}`，`\includegraphics` 只写文件名。
- **等价性验证手法**：PyMuPDF 逐页 `page.get_text()` 比对新旧 PDF（本次重构后 35/35 页文本逐字一致）。
- **配图约定**：`CUMCM2026-Template/figures/` 现放 20 个矢量 PDF，命名 `p1_fig1-3`（问题一）、`p2_fig1-7`（问题二）、`p3_fig1-6`（问题三）、`p4_fig1-4`（问题四，来源 `问题四/照片/fig1-4.pdf`）。**换图 = 复制进来 + 裁白边**（见下），不要再去跑 q_paper_figures.py（那是旧 png 链路）。
- **表格排版坑（Q4 新增，两次踩）**：中文长单元格**不要**用固定列宽的 `tabular`（会 Overfull 或 Underfull badness 10000）；一律用 `\begin{tabularx}{\textwidth}{lXcc}` 让文字列走 `X`。列数多（≥4 个中文列）时用 `\footnotesize` + `clXX` 紧凑版，否则触发 "Float too large for page"。
- **符号表已按问题拆分**：`tab:sym-common`（问题一至三）与 `tab:sym-q34`（问题三预报通道＋问题四实时电价）。加符号先判断归哪张，别塞回单张表（会 Float too large）。
- **裁白边**：这些 matplotlib PDF 是整幅 A4 页，内容只占中间一块。必须用 PyMuPDF 取 drawings+text+image 的并集外接框 `set_cropbox()` 再存，否则图和页面都会大片留白。
- **版式纪律**：高表（二十余行）用 `[H]` 反而更干净——`[htbp]` 会把它甩成**居中浮动页**（上下各留大片空白）并可能挤出一张**只剩一行正文**的页；`[H]` 牺牲的是表前那页的底部余白。矮表仍用 `[htbp]`。放宽 `\topfraction` 等参数（0.92/0.85/0.06/0.88）**试过无效且引入 Overfull**，已回退，别再加。编译后用 pdftotext / PyMuPDF 按页扫"稀疏页"（每页 <500 字符即人工复核，注意含矢量图的图页也会被判稀疏，需看 drawings 数）。
- **中文字体缺字**：带圈数字 ①②③、希腊 β 等会被 xeCJK 判为西文而显示空白，导言区加 `\xeCJKDeclareCharClass{CJK}{"2460 -> "2473, "0370 -> "03FF}`。
- 论文现状（2026-09-13 二次修订后）：**60 页，0 warning / 0 undefined / 0 overfull / 0 underfull / 0 missing char**；摘要正好 1 页（关键词 5 个），正文 p2–p48（**47 页**），AI 声明 p49，参考文献 p50，附录 p50–60。5 张答案表已换成题目原表样式（见下节）。
- **修订记录（2026-09-13，按《初版论文修改建议》纯文本改）**：详见 `.workbuddy/memory/2026-09-13.md` 第二节。要点：
  - **账本统一**：Q3/Q4 逐日 $J=\sum p_t b^{\rm f}_t+\frac12\sum p_t|b^{\rm f}_t-b^{0}_t|+5\sum p_t e_t$（净额；调增时即含 1.5p 边际成本，**不存在**"Q4 不引入 1.5p"）。
  - **符号层次**：点预测 $\mu$ / 规划需求 $\widetilde N$ / 实测 $N$；日前合同 $b^{0}$ / 最终合同 $b^{\rm f}$；计划库存 $\widehat E$ / 实际库存 $E$；供能余量 $w_t$、价格权重 $\omega_t$；PRM 幂指数 $\lambda_{\rm p}$、门控强度 $\lambda$；每日区间数固定 $T=144$、优化时域 $H$。
  - **Q2 裕量**＝窗口中心化净负载分位（同一窗口中位数、无 max 截断）；报童临界分位 **0.8**；执行器为显式 must/max（计划充放电意图**不是**执行上限）。
  - **冷启动**：零向量分支**确实被调用**（5 次，$d=30..34$，候选日 $d'=2..6$）——原文"不会使用零向量"已改。
  - **附件4 结构（复算）**：corr 0.999999996、最大绝对偏差 $5.15\times10^{-5}$ 元/kWh、$\ell_d$ std **0.129**（图中 0.077 是错的）、残差 std 0.081。
  - **口径**：1277.9221 称**放松下界**（非严格下界）；7.7030 仅作"24h+G_fixed 替换价格输入"的条件差额，不得分配 126.32 万差距；"次日全知上界"→次日完美信息参照；48h 收益分分支报（11.08 / 6.70 万，**不相加**）。
  - 新增 `backmatter/declaration.tex`（AI 声明，**5 处【待填】待用户提供事实**）；references 增至 5 条（+Hyndman fpp3 §5.10、Mayne 2000 Automatica 36(6):789–814）；附录重写为清单+入口+版本+映射表+审计分项。

## 答案表样式（2026-09-13，按 `D:\09_Temp\C题_表格汇总.tex` 对齐）
- 范围：**仅 5 张"对应题目表 N"的答案表** —— `tab:q1-t1`（表3）、`tab:q1-t2`（表4）、`tab:q2-t1`（表8）、`tab:q2-t2`（表9）、`tab:q2-t3`（表10），位于 `sections/04-problem1.tex` 与 `sections/06-problem2-results.tex`。**正文一字未改**（已用"旧 PDF 全部 ≥25 字符长行去空白后在新 PDF 中查找，0 缺失"验证）。
- 题目原表样式 = `|c|c|...|` 全框线 + 全 `\hline`（**不用 booktabs**）；行排布照抄：
  - 表1：`时间段|购电量`×3 并排 → 两行时段（10/12/14、16/18/20）→ 末行「全天购电量｜值｜全天购电费｜值」（标签 `\multicolumn{2}{|c|}{…}`，值放下一列）。
  - 表2：`时间段|充电量|放电量`×2 → 三行时段 → 末行「0:00 储电量｜值｜24:00 储电量｜值」。
  - 表3：4 个日期组「时间段|购电量」×4 并排 + `\multicolumn{2}` 日期表头 + 3 行数据。
- **Q2 四个指定日期**：按题面"按表 1/表 2/表 3 的格式给出"改为**每日期一块、纵向堆叠**（表8 21 行、表9 17 行），日期用 `\multicolumn{6}{|c|}{2025-03-20}` 表头行分隔；结论：表8/表9/表10 用 `[H]`，`\small`+`tabcolsep 4pt`（表10 用 `\footnotesize`+`3pt`+`\resizebox`，表头 `\shortstack{紧急购电量\\(kWh)}` 压列宽）。
- 备份：`_refactor_backup/{04-problem1.tex,06-problem2-results.tex,main.pdf}.0913-1122.bak`。`sections/` 中已无 `\multirow` 使用。

## 问题四（Q4）资产与口径
- 初稿引用的 `scripts/q4_model_v3.py`、`scripts/q4_check_v3.py` 本机不存在；`results/q4_v3/F_pt.csv` 已由 `scripts/q4_export_pt.py` 重建（365×144，评价期均值 374.31 kWh/区间）。
- **Q4 结果**（照抄初稿数字，已部分独立复核）：F_48（Q2 分支）**1438.2280 万**、G_48（Q3 分支）**1404.2431 万**；
  48h 时域收益 −11.08 / −6.70 万（合计 17.78）；价格预知价值 **7.7030 万**（同政策对照）；严格下界 1277.9221 万。
- **已独立复核通过**：附件 4 全部 8 项统计量；两 xlsx 的合同费（1395.1120 / 1351.6506）与附加费（18.3849）**逐位吻合**。
  未复核：20 个配置费用、选择器实验、终端归一化 → 正文按初稿原值呈现，未声称全量核验。
- **F_pt 口径复现差异（未解决）**：用重建的 F_pt 跑 `code/q4_core.py` 得 F_48=1456.4631、G_48=1418.7323（各高约 15~18 万）。
  分解定位：合同费 1392.56 vs 1395.11（差 2.5 万，吻合），**紧急费 63.90 vs 43.14（差 20.8 万 / 48%）** ⇒ 根因在执行层预测偏低，非 LP 层。
  试过 6 种口径（center / with_margin / A_noweight / B_nomargin / C_prm_delta / D_center_w1）全部对不上；**原口径无法反推**。用户决定照抄初稿数字。
- **Q4 章节**：8 张表 + 4 张图（图17–20）＋附录代码节 `app:q4-code`（`code/q4_core.py`，与 q1_core/q2_core 同体例）。
  正文 §8.5 校验小节含「（五）关于复现范围的说明」，**如实披露**哪些经独立复算、哪些来自建模过程记录。

## 未决（2026-09-13 更新）
- **正文 47 页 > 建议的 30 页**：纯文本修订已到极限（摘要省 1 页、重述/分析/假设压缩约 3 页，但为落实公式与口径修正又增约 4 页）。再压必须**把约 8–9 幅次要图移到附录**（建议保留 Q1 3 幅、Q2 3 幅、Q3 3 幅、Q4 2 幅），用户本次选了"不动图"，未做。
- **问题三仍无结果文件与可跑链路**：`results/` 无 `result3.xlsx`；`问题三/代码/代码.md` 是带 `...` 占位的草稿，且其中 `ETA_C=ETA_D=0.95` 与论文/其他问的 0.9 矛盾。论文 Q3 章节**缺题面指定的四个日期表 1/表 2/表 3**，正文数字只取自图注；附录已如实标注 `result3` 未纳入核验。
- **图内标注改不了（只在图注/正文声明以复核值为准）**：`p4_fig1.pdf` 内 $\sigma(\ell_d)=0.077$（实为 0.129）；`p4_fig2.pdf`（图19）虚线基准是 F_base=1453.25 / G_base=1420.95，不是 F_fixed；`p4_fig4.pdf`（图20）仍写"上界"字样。
- **问题四复现口径未闭合**：`F_pt` 原始口径未知，重建版使 F_48/G_48 高约 15~18 万；20 配置未全量复核（用户决定照抄初稿数字）。
- **AI 声明待填**：`backmatter/declaration.tex` 中 5 处 `【待填】`（工具与型号、提示过程、人工核验）需用户提供事实清单。
- **O6**：结果模板列槽位映射冲突（默认按列序 1:1 提交）。**O8**：kW→kWh 积分规则未消除，与 O6 独立。外部方案 1356.64 万元未核验、本文档不归因。
