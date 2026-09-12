# -*- coding: utf-8 -*-
"""定位问题三章节所在页，并渲染为 PNG 供核对版式。"""
import subprocess
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
T = BASE / "CUMCM2026-Template"
OUT = BASE / ".tmp_preview" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

pdf = T / "main.pdf"
txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                     capture_output=True).stdout.decode("utf-8", "replace")
pages = txt.split("\f")
print("总页数:", len(pages))

for i, p in enumerate(pages):
    for key in ["问题三：多时刻预报", "两维对照", "是否需要引入其他时刻", "支撑材料文件列表",
                "模型评价"]:
        if key in p:
            print(f"page {i+1}: {key}")

hit = [i + 1 for i, p in enumerate(pages)
       if "问题三：多时刻预报" in p or "是否需要引入其他时刻" in p]
print("渲染页:", hit)
for pno in hit:
    subprocess.run(["pdftoppm", "-png", "-r", "80", "-f", str(pno), "-l", str(pno),
                    str(pdf), str(OUT / f"p{pno:02d}")], check=True, capture_output=True)

imgs = sorted(OUT.glob("p*.png"))
print([p.name for p in imgs])
