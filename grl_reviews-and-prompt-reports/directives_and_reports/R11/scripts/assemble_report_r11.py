"""Splice the generated tables into the R11 report: {{table:NAME}} -> R11/tables/NAME.md verbatim. No number in a report table is typed by hand."""
import os, re
REP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TAB = f"{REP}/tables"
src = open(f"{REP}/REPORT_R11_template.md").read()
out = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: open(f"{TAB}/{m.group(1)}.md").read().rstrip("\n"), src)
assert "{{" not in out, re.findall(r"\{\{[^}]+\}\}", out)
open(f"{REP}/REPORT_R11_2026-10-10.md", "w").write(out); print(f"wrote REPORT_R11_2026-10-10.md ({len(out):,} chars)")
