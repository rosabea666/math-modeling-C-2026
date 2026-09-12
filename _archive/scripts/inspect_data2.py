# -*- coding: utf-8 -*-
"""检查附件2/3/4结构，确认时间口径"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import openpyxl

BASE = r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23\data"

def peek(path, name, n=6):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    print(f"##### {name} #####")
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        print(f"-- {ws.title}: rows={len(rows)} cols={len(rows[0]) if rows else 0}")
        for r in rows[:n]:
            print("  ", r)
        print("   ...")
        for r in rows[-3:]:
            print("  ", r)
    wb.close()
    print()

peek(BASE + r"\附件2.xlsx", "附件2")
peek(BASE + r"\附件3.xlsx", "附件3")
peek(BASE + r"\附件4.xlsx", "附件4")
