"""R08 §7 — freeze record: sha256 of every R08 script, table, figure and the key file; the fixed-input checks; git HEAD. No tag."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r08_common as R8
HERE = os.path.dirname(os.path.abspath(__file__)); R8.start("freeze_r08")
scripts = {f: R8.sha256_file(f"{HERE}/{f}") for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sbatch"))}
tables = {os.path.basename(f): R8.sha256_file(f) for f in sorted(glob.glob(f"{R8.TABLES}/*"))}
figs = {os.path.relpath(f, R8.FIGURES): R8.sha256_file(f) for f in sorted(glob.glob(f"{R8.FIGURES}/**/*", recursive=True)) if os.path.isfile(f)}
rec = dict(scripts_sha256=scripts, tables_sha256=tables, figures_sha256=figs, fixed_inputs=R8.check_fixed_inputs(), git_head_at_run=R8.git_head(), tag=None, phase0_freeze_commit=R8.R6.PHASE0_FREEZE_COMMIT)
json.dump(rec, open(f"{R8.REP}/freeze_record.json", "w"), indent=1, default=str)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `scripts/{f}` | `{h}` |" for f, h in scripts.items()] + [f"| `figures/{f}` | `{h}` |" for f, h in figs.items() if f.endswith(".pdf") or f.endswith(".json")] + [f"| tables | {len(tables)} files, sha256 in freeze_record.json |", f"| context_v1.parquet | `{rec['fixed_inputs']['context_v1_parquet']['sha256']}` |", f"| git HEAD at run | {rec['git_head_at_run']} |", "| tag | none (directive §7) |"]
open(f"{R8.REP}/freeze_record.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
