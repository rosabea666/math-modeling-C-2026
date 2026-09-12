# -*- coding: utf-8 -*-
"""把论文若干页拼成联络表，便于一次核对。"""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
T = BASE / "CUMCM2026-Template"
OUT = BASE / ".tmp_preview" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

PAGES = [1, 29, 30, 31, 32, 33, 34, 35, 36, 37, 42, 44]
for p in PAGES:
    subprocess.run(["pdftoppm", "-png", "-r", "95", "-f", str(p), "-l", str(p),
                    str(T / "main.pdf"), str(OUT / f"s{p:02d}")], check=True,
                   capture_output=True)

imgs = []
for p in PAGES:
    f = sorted(OUT.glob(f"s{p:02d}*.png"))[0]
    im = Image.open(f)
    im.thumbnail((430, 620))
    c = Image.new("RGB", (440, 650), "white")
    c.paste(im, (5, 25))
    ImageDraw.Draw(c).text((6, 6), f"page {p}", fill="red")
    imgs.append(c)

COLS = 4
rows = (len(imgs) + COLS - 1) // COLS
sheet = Image.new("RGB", (440 * COLS, 650 * rows), "#dddddd")
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % COLS) * 440, (i // COLS) * 650))
sheet.save(OUT / "sheet_paper.png")
print("saved", sheet.size)
