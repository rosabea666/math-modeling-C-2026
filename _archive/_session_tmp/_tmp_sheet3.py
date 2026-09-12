# -*- coding: utf-8 -*-
"""放大核对论文 29-40 页版式。"""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
T = BASE / "CUMCM2026-Template"
OUT = BASE / ".tmp_preview" / "pdf2"
OUT.mkdir(parents=True, exist_ok=True)

PAGES = list(range(29, 41))
for p in PAGES:
    subprocess.run(["pdftoppm", "-png", "-r", "110", "-f", str(p), "-l", str(p),
                    str(T / "main.pdf"), str(OUT / f"q{p:02d}")], check=True,
                   capture_output=True)

imgs = []
for p in PAGES:
    f = sorted(OUT.glob(f"q{p:02d}*.png"))[0]
    im = Image.open(f)
    im.thumbnail((520, 730))
    c = Image.new("RGB", (530, 760), "white")
    c.paste(im, (5, 25))
    ImageDraw.Draw(c).text((6, 6), f"page {p}", fill="red")
    imgs.append(c)

COLS = 3
rows = (len(imgs) + COLS - 1) // COLS
sheet = Image.new("RGB", (530 * COLS, 760 * rows), "#dddddd")
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % COLS) * 530, (i // COLS) * 760))
sheet.save(OUT / "sheet_q3.png")
print("saved", sheet.size)
