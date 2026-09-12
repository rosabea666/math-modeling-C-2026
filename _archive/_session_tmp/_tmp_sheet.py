# -*- coding: utf-8 -*-
"""拼对照表：一张图里并排显示所有 PNG 与所有问题 PDF 首页。"""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
FIG = BASE / "CUMCM2026-Template" / "figures"
TMP = BASE / ".tmp_pair"
TMP.mkdir(exist_ok=True)
OUT = BASE / ".tmp_preview"

CELL = 420
COLS = 4

def tile(img, label):
    img = img.convert("RGB")
    img.thumbnail((CELL - 10, CELL - 30))
    canvas = Image.new("RGB", (CELL, CELL), "white")
    canvas.paste(img, ((CELL - img.width) // 2, 24))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, CELL - 1, CELL - 1], outline="#cccccc")
    d.text((6, 6), label, fill="black")
    return canvas

def sheet(items, out_name):
    rows = (len(items) + COLS - 1) // COLS
    sh = Image.new("RGB", (CELL * COLS, CELL * rows), "#f2f2f2")
    for i, (img, lab) in enumerate(items):
        sh.paste(tile(img, lab), ((i % COLS) * CELL, (i // COLS) * CELL))
    sh.save(OUT / out_name)
    print("saved", out_name, sh.size)

# PNG 组
pngs = sorted(FIG.glob("p*_fig*.png"))
sheet([(Image.open(p), p.name) for p in pngs], "sheet_png.png")

# PDF 组
items = []
for q, n in [("问题一", 3), ("问题二", 7), ("问题三", 6)]:
    for i in range(1, n + 1):
        pdf = BASE / q / "图片" / f"fig{i}.pdf"
        tag = TMP / f"c_{q}_{i}"
        subprocess.run(["pdftoppm", "-png", "-r", "140", "-f", "1", "-l", "1",
                        str(pdf), str(tag)], check=True, capture_output=True)
        got = sorted(TMP.glob(f"c_{q}_{i}*.png"))[0]
        items.append((Image.open(got), f"{q[:2]}-fig{i}"))
sheet(items, "sheet_pdf.png")
