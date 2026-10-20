"""Splice the generated tables (R15/tables/*.md) verbatim into the report template R15/REPORT_R15_template.md at the
`<!-- TABLE: name.md -->` markers and write R15/REPORT_R15_2026-10-13.md. Nothing else is changed."""
import os, re, sys
REP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T = f"{REP}/tables"
src = open(f"{REP}/REPORT_R15_template.md").read()
def sub(m): return open(f"{T}/{m.group(1)}").read().rstrip("\n")
out = re.sub(r"<!-- TABLE: ([A-Za-z0-9_\.]+) -->", sub, src); assert "<!-- TABLE:" not in out
open(f"{REP}/REPORT_R15_2026-10-13.md", "w").write(out); print("wrote", f"{REP}/REPORT_R15_2026-10-13.md", len(out), "bytes")
