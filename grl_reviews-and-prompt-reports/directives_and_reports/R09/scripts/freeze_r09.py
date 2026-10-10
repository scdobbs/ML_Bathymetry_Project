"""R09 §D — freeze record: sha256 of every R09 script (targets_v1_int.py first), table, PREREG_phase4_v1.md, the GBT artifact; the fixed-input checks; git HEAD; tag phase4-prereg."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r09_common as R9
HERE = os.path.dirname(os.path.abspath(__file__)); R9.start("freeze_r09")
scripts = {f: R9.R8.sha256_file(f"{HERE}/{f}") for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sbatch"))}
tables = {os.path.basename(f): R9.R8.sha256_file(f) for f in sorted(glob.glob(f"{R9.TABLES}/*"))}
extra = {f: R9.R8.sha256_file(f"{R9.REP}/{f}") for f in ("PREREG_phase4_v1.md", "PREREG_phase4_v1_template.md")}
art = {os.path.basename(f): R9.R8.sha256_file(f) for f in sorted(glob.glob(f"{R9.OUT_ART}/*"))}
rec = dict(scripts_sha256=scripts, targets_v1_int_sha256=scripts["targets_v1_int.py"], tables_sha256=tables, prereg_sha256=extra, artifacts_sha256=art, fixed_inputs_r09=R9.check_fixed_r09(), git_head_at_run=R9.R8.git_head(), tag="phase4-prereg")
json.dump(rec, open(f"{R9.REP}/freeze_record.json", "w"), indent=1, default=str)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `scripts/{f}` | `{h}` |" for f, h in scripts.items()] + [f"| `{f}` | `{h}` |" for f, h in extra.items()] + [f"| artifact `{f}` | `{h}` |" for f, h in art.items()] + [f"| tables | {len(tables)} files, sha256 in freeze_record.json |", f"| git HEAD at run | {rec['git_head_at_run']} |", "| tag | phase4-prereg (applied to the R09 commit) |"]
open(f"{R9.REP}/freeze_record.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
