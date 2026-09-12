# latexmk 配置：本模板必须用 XeLaTeX（ctex + fontspec + xeCJK）
$pdf_mode        = 5;    # 5 = xelatex
$xelatex         = 'xelatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 %O %S';
$bibtex_use      = 0;    # 参考文献为 thebibliography 手工条目，不调用 bibtex/biber
$max_repeat      = 5;
$pdf_previewer   = '';
$pdf_update_method = 0;
