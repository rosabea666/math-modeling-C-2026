# -*- coding: utf-8 -*-
import glob
import os
import fitz

base = r"C:\Users\fzz17\AppData\Local\Temp"
out_dir = r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23"

pdfs = glob.glob(os.path.join(base, "*CUMCM2026Problems*", "*", "*.pdf"))
for p in pdfs:
    name = os.path.basename(p)
    try:
        doc = fitz.open(p)
        text = ""
        for i, page in enumerate(doc):
            text += f"===== PAGE {i+1} =====\n" + page.get_text() + "\n"
        safe = name.encode("ascii", "ignore").decode() or "unknown"
        # use parent folder name to disambiguate
        parent = os.path.basename(os.path.dirname(p))
        tag = parent[0] if parent else "X"
        out = os.path.join(out_dir, f"problem_{tag}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"OK {parent}/{name} -> {out} pages={doc.page_count}")
    except Exception as e:
        print(f"FAIL {p}: {e}")
