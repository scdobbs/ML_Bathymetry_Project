"""Splice the generated tables into the R08 report: {{table:NAME}} -> R08/tables/NAME.md verbatim. No number in a report
table is typed by hand. Writes REPORT_R08_2026-10-09.md from REPORT_R08_template.md."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r08_common as R8
src = open(f"{R8.REP}/REPORT_R08_template.md").read()
out = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: open(f"{R8.TABLES}/{m.group(1)}.md").read().rstrip("\n"), src)
assert "{{" not in out, re.findall(r"\{\{[^}]+\}\}", out)
open(f"{R8.REP}/REPORT_R08_2026-10-09.md", "w").write(out); print(f"wrote REPORT_R08_2026-10-09.md ({len(out):,} chars)")
