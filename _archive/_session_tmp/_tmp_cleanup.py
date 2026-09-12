# -*- coding: utf-8 -*-
"""把本次会话产生的临时脚本/预览/构建日志归档到 _archive/_session_tmp/（不删除）。"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
DST = BASE / "_archive" / "_session_tmp"
DST.mkdir(parents=True, exist_ok=True)

moved = []
for p in sorted(BASE.glob("_tmp_*.py")):
    shutil.move(str(p), str(DST / p.name))
    moved.append(p.name)
for p in sorted(BASE.glob("_tmp_*")):
    if p.is_dir():
        shutil.move(str(p), str(DST / p.name))
        moved.append(p.name + "/")
for p in sorted((BASE / "CUMCM2026-Template").glob("_build_pass*.log")):
    shutil.move(str(p), str(DST / p.name))
    moved.append("CUMCM2026-Template/" + p.name)

print("归档:")
for m in moved:
    print("  ", m)

print("\n=== 项目根目录 ===")
for p in sorted(BASE.iterdir()):
    print("  ", p.name + ("/" if p.is_dir() else ""))
print("\n=== CUMCM2026-Template/ ===")
for p in sorted((BASE / "CUMCM2026-Template").iterdir()):
    print("  ", p.name + ("/" if p.is_dir() else ""))
