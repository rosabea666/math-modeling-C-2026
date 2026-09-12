# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import openpyxl
p = r"C:\Users\fzz17\WorkBuddy\2026-09-10-18-17-23\data\附件5\result2.xlsx"
wb = openpyxl.load_workbook(p, read_only=True)
for name in ("充放电量", "紧急购电量"):
    ws = wb[name]
    print(f"===== {name} =====")
    for i, r in enumerate(ws.iter_rows(values_only=True), 1):
        print(i, r)
wb.close()
