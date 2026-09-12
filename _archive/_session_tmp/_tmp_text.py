# -*- coding: utf-8 -*-
"""抽取问题三各图 PDF 的文本（用于精确读取图中数值）。"""
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
for i in range(1, 7):
    pdf = BASE / "问题三" / "图片" / f"fig{i}.pdf"
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True)
    txt = out.stdout.decode("utf-8", "replace")
    print("=" * 30, f"fig{i}", "=" * 30)
    print(txt.strip()[:3000])
