"""R08b §5 — freeze record: sha256 of every R08b script, table, the ruling file and the calls CSV; the fixed-input and prediction-source checks; git HEAD. No tag."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r08b_common as RB
HERE = os.path.dirname(os.path.abspath(__file__)); RB.start("freeze_r08b")
scripts = {f: RB.R8.sha256_file(f"{HERE}/{f}") for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sbatch"))}
tables = {os.path.basename(f): RB.R8.sha256_file(f) for f in sorted(glob.glob(f"{RB.TABLES}/*"))}
extra = {f: RB.R8.sha256_file(f"{RB.REP}/{f}") for f in ("ruling_unit_set_2026-10-09.md", "R08_figset_A_calls_steve_2026-10-09.csv")}
rec = dict(scripts_sha256=scripts, tables_sha256=tables, ruling_and_calls_sha256=extra, fixed_inputs=RB.R8.check_fixed_inputs(), prediction_sources=RB.check_pred_sources(), git_head_at_run=RB.R8.git_head(), tag=None)
json.dump(rec, open(f"{RB.REP}/freeze_record.json", "w"), indent=1, default=str)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `scripts/{f}` | `{h}` |" for f, h in scripts.items()] + [f"| `{f}` | `{h}` |" for f, h in extra.items()] + [f"| tables | {len(tables)} files, sha256 in freeze_record.json |", f"| git HEAD at run | {rec['git_head_at_run']} |", "| tag | none (directive §5) |"]
open(f"{RB.REP}/freeze_record.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
