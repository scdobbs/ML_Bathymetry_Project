"""Splice the generated tables into the R08b report: {{table:NAME}} -> R08/tables/NAME.md verbatim. No number in a report
table is typed by hand. Writes REPORT_R08b_2026-10-09.md from REPORT_R08b_template.md."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r08b_common as RB; R8 = RB.R8
src = open(f"{RB.REP}/REPORT_R08b_template.md").read()
out = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: open(f"{RB.TABLES}/{m.group(1)}.md").read().rstrip("\n"), src)
assert "{{" not in out, re.findall(r"\{\{[^}]+\}\}", out)
open(f"{RB.REP}/REPORT_R08b_2026-10-09.md", "w").write(out); print(f"wrote REPORT_R08b_2026-10-09.md ({len(out):,} chars)")
