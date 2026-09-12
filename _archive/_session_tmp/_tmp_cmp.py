# -*- coding: utf-8 -*-
"""对比 figures/ 下的 png 与各问题图片 pdf 的关系（尺寸/页数/内容哈希不适用，仅列大小与页数）。"""
from pathlib import Path
import subprocess, re

BASE = Path(__file__).resolve().parent
FIG = BASE / "CUMCM2026-Template" / "figures"

print("=== figures/ ===")
for p in sorted(FIG.iterdir()):
    print(f"{p.name:28s} {p.stat().st_size:>9d}")

print()
print("=== pdf 页数 & 尺寸 ===")
pdfs = []
for q, n in [("问题一", 3), ("问题二", 7), ("问题三", 6)]:
    for i in range(1, n + 1):
        pdfs.append(BASE / q / "图片" / f"fig{i}.pdf")
for p in pdfs:
    out = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True).stdout
    pages = re.search(r"Pages:\s+(\d+)", out)
    size = re.search(r"Page size:\s+(.+)", out)
    print(f"{p.parent.parent.name}/fig{p.stem[3:]}.pdf  pages={pages.group(1) if pages else '?'}  {size.group(1).strip() if size else '?'}")

from PIL import Image
print()
print("=== png 尺寸 ===")
for p in sorted(FIG.glob("*.png")):
    im = Image.open(p)
    print(f"{p.name:28s} {im.size}")
