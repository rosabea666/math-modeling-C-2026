# -*- coding: utf-8 -*-
"""数据驱动地把 figures/*.png 与 问题*/*/fig*.pdf 配对：渲染 PDF 到同尺寸后算像素差。"""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
FIG = BASE / "CUMCM2026-Template" / "figures"
TMP = BASE / ".tmp_pair"
TMP.mkdir(exist_ok=True)

pngs = sorted(FIG.glob("p*_fig*.png"))
pdfs = []
for q, n in [("问题一", 3), ("问题二", 7), ("问题三", 6)]:
    for i in range(1, n + 1):
        pdfs.append(BASE / q / "图片" / f"fig{i}.pdf")

def render(pdf, w, h, tag):
    dst = TMP / tag
    subprocess.run(["pdftoppm", "-png", "-scale-to-x", str(w), "-scale-to-y", str(h),
                    "-f", "1", "-l", "1", str(pdf), str(dst)], check=True,
                   capture_output=True)
    got = list(TMP.glob(tag + "*.png"))
    return got[0] if got else None

for png in pngs:
    a = Image.open(png).convert("RGB")
    w, h = a.size
    a_arr = np.asarray(a.resize((400, 400)), dtype=float)
    scores = []
    for pdf in pdfs:
        tag = "r_" + pdf.parent.parent.name + "_" + pdf.stem
        p = render(pdf, w, h, tag)
        if p is None:
            scores.append((999.0, str(pdf)))
            continue
        b = Image.open(p).convert("RGB").resize((400, 400))
        b_arr = np.asarray(b, dtype=float)
        scores.append((float(np.abs(a_arr - b_arr).mean()), f"{pdf.parent.parent.name}/图片/{pdf.name}"))
    scores.sort()
    print(f"{png.name:20s} -> {scores[0][1]:22s} diff={scores[0][0]:6.2f} | 次优 {scores[1][1]} diff={scores[1][0]:6.2f}")
