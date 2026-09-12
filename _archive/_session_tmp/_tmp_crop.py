# -*- coding: utf-8 -*-
"""高清裁剪：fig1（费用矩阵）、fig6(a)（门控分布）用于精确读数。"""
import subprocess
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
TMP = BASE / ".tmp_preview"

def crop(pdf, tag, box, dpi=400):
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1",
                    str(pdf), str(TMP / tag)], check=True, capture_output=True)
    p = sorted(TMP.glob(tag + "*.png"))[0]
    im = Image.open(p)
    W, H = im.size
    l, t, r, b = [int(v * s) for v, s in zip(box, (W, H, W, H))]
    im.crop((l, t, r, b)).save(TMP / (tag + "_crop.png"))
    print(tag, im.size, "->", (r - l, b - t))

crop(BASE / "问题三" / "图片" / "fig1.pdf", "z1", (0.15, 0.32, 0.95, 0.62))
crop(BASE / "问题三" / "图片" / "fig6.pdf", "z6", (0.05, 0.40, 0.55, 0.85))
