# -*- coding: utf-8 -*-
"""检查result2.xlsx模板结构"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import openpyxl

p = r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23\data\附件5\result2.xlsx"
wb = openpyxl.load_workbook(p, read_only=True)
for ws in wb.worksheets:
    rows = list(ws.iter_rows(values_only=True))
    print(f"-- {ws.title}: rows={len(rows)} cols={len(rows[0]) if rows else 0}")
    for r in rows[:6]:
        print("  ", r)
    if len(rows) > 9:
        print("   ...")
        for r in rows[-3:]:
            print("  ", r)
    print()
wb.close()
