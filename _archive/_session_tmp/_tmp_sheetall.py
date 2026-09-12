# -*- coding: utf-8 -*-
"""全文联络表：每页缩略图，一次核对版式与配图。"""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
T = BASE / "CUMCM2026-Template"
OUT = BASE / ".tmp_preview" / "all"
OUT.mkdir(parents=True, exist_ok=True)

pdf = T / "main.pdf"
subprocess.run(["pdftoppm", "-png", "-r", "55", str(pdf), str(OUT / "a")],
               check=True, capture_output=True)
files = sorted(OUT.glob("a-*.png"))
print("pages:", len(files))

imgs = []
for i, f in enumerate(files, 1):
    im = Image.open(f)
    im.thumbnail((300, 424))
    c = Image.new("RGB", (306, 440), "white")
    c.paste(im, (3, 3))
    ImageDraw.Draw(c).text((6, 428), f"p{i}", fill="red")
    imgs.append(c)

COLS = 7
rows = (len(imgs) + COLS - 1) // COLS
sheet = Image.new("RGB", (306 * COLS, 440 * rows), "#cccccc")
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % COLS) * 306, (i // COLS) * 440))
sheet.save(OUT / "sheet_all.png")
print("saved", sheet.size)
