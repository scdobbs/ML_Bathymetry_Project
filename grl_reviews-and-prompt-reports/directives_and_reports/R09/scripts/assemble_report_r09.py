"""Splice the generated tables into the R09 report: {{table:NAME}} -> R08/tables/NAME.md verbatim. No number in a report
table is typed by hand. Writes REPORT_R09_2026-10-09.md from REPORT_R09_template.md."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r09_common as R9; RB = R9.RB; R8 = R9.R8
src = open(f"{R9.REP}/REPORT_R09_template.md").read()
out = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: open(f"{R9.TABLES}/{m.group(1)}.md").read().rstrip("\n"), src)
assert "{{" not in out, re.findall(r"\{\{[^}]+\}\}", out)
open(f"{R9.REP}/REPORT_R09_2026-10-09.md", "w").write(out); print(f"wrote REPORT_R09_2026-10-09.md ({len(out):,} chars)")
