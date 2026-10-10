"""Splice the generated tables into the R10 report: {{table:NAME}} -> R08/tables/NAME.md verbatim. No number in a report
table is typed by hand. Writes REPORT_R10_2026-10-10.md from REPORT_R10_template.md."""
import os, re, sys
REP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TAB = f"{REP}/tables"
src = open(f"{REP}/REPORT_R10_template.md").read()
out = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: open(f"{TAB}/{m.group(1)}.md").read().rstrip("\n"), src)
assert "{{" not in out, re.findall(r"\{\{[^}]+\}\}", out)
open(f"{REP}/REPORT_R10_2026-10-10.md", "w").write(out); print(f"wrote REPORT_R10_2026-10-10.md ({len(out):,} chars)")
