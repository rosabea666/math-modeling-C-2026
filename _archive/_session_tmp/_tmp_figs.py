# -*- coding: utf-8 -*-
"""把 问题一/二/三/图片/*.pdf 复制进 CUMCM2026-Template/figures/，
旧位图与孤立图移入 _archive/figures_replaced/（可逆，不硬删）。"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
FIG = BASE / "CUMCM2026-Template" / "figures"
ARC = BASE / "_archive" / "figures_replaced"
ARC.mkdir(parents=True, exist_ok=True)

PLAN = [("问题一", 3, "p1"), ("问题二", 7, "p2"), ("问题三", 6, "p3")]

keep = set()
for q, n, pre in PLAN:
    for i in range(1, n + 1):
        src = BASE / q / "图片" / f"fig{i}.pdf"
        dst = FIG / f"{pre}_fig{i}.pdf"
        shutil.copy2(src, dst)
        keep.add(dst.name)
        print(f"copy {q}/图片/fig{i}.pdf -> figures/{dst.name}")

print()
for p in sorted(FIG.iterdir()):
    if p.name in keep:
        continue
    shutil.move(str(p), str(ARC / p.name))
    print(f"archive figures/{p.name}")

print("\n=== figures/ 现在 ===")
for p in sorted(FIG.iterdir()):
    print(" ", p.name)
