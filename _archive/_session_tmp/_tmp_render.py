# -*- coding: utf-8 -*-
"""把各问题图片 PDF 首页渲染为 PNG，供人眼核对内容。只读，不改动任何原始文件。"""
import subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / ".tmp_preview"
OUT.mkdir(exist_ok=True)

targets = []
for q, n in [("问题一", 3), ("问题二", 7), ("问题三", 6)]:
    for i in range(1, n + 1):
        targets.append(BASE / q / "图片" / f"fig{i}.pdf")

for pdf in targets:
    if not pdf.exists():
        print("MISSING", pdf)
        continue
    tag = pdf.parent.parent.name + "_" + pdf.stem
    dst = OUT / tag
    subprocess.run(["pdftoppm", "-png", "-r", "110", "-f", "1", "-l", "1",
                    str(pdf), str(dst)], check=True)
    print("OK", tag)

print("--- files ---")
for p in sorted(OUT.iterdir()):
    print(p.name)
