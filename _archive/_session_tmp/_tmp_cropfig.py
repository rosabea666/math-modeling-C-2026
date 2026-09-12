# -*- coding: utf-8 -*-
"""把 figures/ 下的配图 PDF 按内容外接框裁边（等价于 matplotlib bbox_inches='tight'）。
只改 CUMCM2026-Template/figures/ 下的副本；问题一/二/三 下的原始图不动。"""
from pathlib import Path
import pymupdf

BASE = Path(__file__).resolve().parent
FIG = BASE / "CUMCM2026-Template" / "figures"
MARGIN = 3.0

for src in sorted(FIG.glob("*.pdf")):
    doc = pymupdf.open(src)
    page = doc[0]
    rects = [d["rect"] for d in page.get_drawings()]
    rects += [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")]
    for im in page.get_image_info():
        rects.append(pymupdf.Rect(im["bbox"]))
    if not rects:
        print("SKIP(no content)", src.name)
        continue
    r = rects[0]
    for x in rects[1:]:
        r |= x
    r = pymupdf.Rect(r.x0 - MARGIN, r.y0 - MARGIN, r.x1 + MARGIN, r.y1 + MARGIN)
    r &= page.mediabox
    old = (page.rect.width, page.rect.height)
    page.set_cropbox(r)
    tmp = src.with_suffix(".cropped.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    tmp.replace(src)
    new = (r.width, r.height)
    print(f"{src.name:20s} {old[0]:7.1f}x{old[1]:7.1f} -> {new[0]:7.1f}x{new[1]:7.1f}"
          f"  ratio {new[0]/new[1]:.3f}")
