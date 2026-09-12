# 全国大学生数学建模竞赛 LaTeX 模板（2026 年修订稿）

> 本文由 Jiayi ([@langongic](https://github.com/langonginc)) 依据竞赛论文格式规范，使用 OpenAI GPT 5.6 辅助制作，使用本品所产生的一切影响 Jiayi 不予负责。分发时务必注明原作者来源。

本模板依据全国大学生数学建模竞赛组委会发布的《全国大学生数学建模竞赛论文格式规范（2026 年修订稿）》制作，适用于**电子版论文**。它已设置：

- A4 纸张和上下左右 2.5 cm 页边距；
- 摘要页为 PDF 第 1 页，页脚中部以阿拉伯数字从 1 开始连续编号；
- 无目录的正文结构；
- 附录、参考文献和程序清单示例；
- `ctex` 中文支持，可用 XeLaTeX 编译。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `main.tex` | 主控文件：文档类、论文元数据与模块装配；正文内容不在本文件。 |
| `cumcm2026.sty` | 版式与常用命令。通常不需要修改。 |
| `preamble/` | 导言区：宏包、编码与字体、版式与全局参数。 |
| `sections/` | 正文，文件名序号与 `\section` 顺序一一对应。 |
| `backmatter/` | `references.tex`（参考文献）与 `appendix.tex`（附录）。 |
| `figures/` | 正文配图；路径由 `\graphicspath` 统一解析，正文只写文件名。 |
| `code/` | 附录程序清单，用 `fancyvrb` 的 `\VerbatimInput` 逐字输入。 |
| `.latexmkrc` | latexmk 配置：强制 XeLaTeX，参考文献为手工条目故不调用 bibtex。 |

## 目录结构

```
CUMCM2026-Template/
├── main.tex                     主控文件：文档类、元数据、模块装配
├── cumcm2026.sty                模板样式（版式、页码、摘要环境、超链接）
├── .latexmkrc                   latexmk 配置：XeLaTeX + 不调用 bibtex
├── preamble/
│   ├── packages.tex               宏包声明
│   ├── fonts.tex                  编码、中文字体与字符类
│   └── layout.tex                 版式、图形路径与全局排版参数
├── sections/
│   ├── 00-abstract.tex            摘要页
│   ├── 01-restatement.tex         一、问题重述
│   ├── 02-analysis.tex            二、问题分析
│   ├── 03-assumptions.tex         三、模型假设与符号说明
│   ├── 04-problem1.tex            四、问题一：确定性日调度模型
│   ├── 05-problem2-model.tex      五、问题二：序贯决策模型
│   ├── 06-problem2-results.tex    六、问题二的结果分析
│   ├── 07-problem3.tex            七、问题三：滚动合同调整模型
│   ├── 08-problem4.tex            八、问题四（留白待补）
│   └── 09-evaluation.tex          九、模型评价
├── backmatter/
│   ├── references.tex             参考文献（thebibliography 手工条目）
│   └── appendix.tex               附录：支撑材料清单与核心程序
├── figures/                     正文配图（矢量 PDF）
├── code/                        附录程序清单
├── test/                        模板自带的图形测试文件（未使用）
└── _minted/                     模板自带的旧 minted 缓存（未使用，minted 不可用）
```

### 标签命名规范

统一为 `sec:qN-name`、`fig:qN-name`、`tab:qN-name`、`eq:qN-name`、`app:name`（全小写、连字符分段）。
正文一律用 `\ref` / `\eqref` / `\S\ref` 交叉引用，不写死编号。

### 增删章节

在 `sections/` 下新建文件，并在 `main.tex` 的 `\input` 列表中按顺序登记即可；
`\input`（而非 `\include`）保证不会插入额外分页，章节增删不改变整体排版。

## 编译

### Overleaf 平台

本模版正在 [overleaf](https://www.overleaf.com/) 平台使用，我们建议您在这里使用。

**需要注意的是，请在 `Files -> Settings -> Compiler` 中设置编译器为 `XeLaTex` 。**

<img width="330" height="399" alt="image" src="https://github.com/user-attachments/assets/41d4af1a-e869-4bc2-a7ab-f0e5d93a858f" />

<img width="960" height="563" alt="image" src="https://github.com/user-attachments/assets/d61a544f-e821-4c13-a939-a848009347ee" />

### 手动编译

请使用 **XeLaTeX**，不要使用 pdfLaTeX（ctex + fontspec + xeCJK 依赖它）。

推荐用 `latexmk`（已随附 `.latexmkrc`，在项目根目录执行会自动跑满所需次数）：

```bash
latexmk main.tex
```

或手动跑两遍：

```bash
xelatex main.tex
xelatex main.tex
```

本论文的参考文献是 `backmatter/references.tex` 中的 `thebibliography` 手工条目，
**不需要**（也不应）调用 `bibtex` 或 `biber`。若将来改为 `.bib` + `biblatex`，
需同步修改 `.latexmkrc`，届时才会用到 biber。

在 TeXStudio 中，将“默认编译器”改为 `XeLaTeX` 后编译即可。Overleaf 中请选择 `XeLaTeX` 编译器。

## 使用前清单

- 电子版**不要**放入承诺书和编号专用页；电子版第一页必须为摘要页。
- 正文不要目录，且正文不超过 30 页；附录页数不限。
- 摘要（含标题和关键词）原则上不超过一页。
- 摘要、正文和附录中不要出现参赛者姓名、学校、赛区或其他身份信息。
- 在正文中标注所有引用，并在参考文献中完整列出。
- 在附录中列出支撑材料；将全部完整、可运行的源程序、数据资料和必要中间结果按竞赛要求另行提交。
- 最终提交前检查 PDF 小于 20 MB，且与纸质版（如需提交）完全一致。

> 说明：字号、字体、行距和颜色并非该规范统一要求；本模板只提供一套清晰、保守的默认样式。各赛区如有额外要求，应以赛区要求为准。

## 依据

全国大学生数学建模竞赛组委会，《全国大学生数学建模竞赛论文格式规范（2026 年修订稿）》，2026-03-03：
<https://www.mcm.edu.cn/html_cn/node/4cd596519c9eb9fbd866398f6df0caa3.html>
