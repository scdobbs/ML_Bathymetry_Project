"""R10 §E — freeze record: sha256 of the five Phase 4 scripts, every R10 script and table, the pre-registration v2 (or draft), and the R09 / R08 fixed inputs; git HEAD."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts")); import phase4_common as P4
REP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.dirname(os.path.abspath(__file__))
m0 = P4.C4.load_manifest(); P4.R9.install_guard_r09(m0); ver = P4.verify_frozen("rehearsal")
p4 = {sc: P4.sha256_file(os.path.join(P4.HERE, sc)) for sc in P4.SCRIPTS}; scripts = {f: P4.sha256_file(f"{HERE}/{f}") for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sbatch", ".sh"))}
tables = {os.path.basename(f): P4.sha256_file(f) for f in sorted(glob.glob(f"{REP}/tables/*"))}; prereg = {f: P4.sha256_file(f"{REP}/{f}") for f in ("PREREG_phase4_v2.md", "PREREG_phase4_v2_DRAFT.md") if os.path.exists(f"{REP}/{f}")}
rec = dict(phase4_scripts_sha256=p4, r10_scripts_sha256=scripts, tables_sha256=tables, prereg_v2=prereg, prereg_v1=dict(path=P4.PREREG_V1, sha256=P4.sha256_file(P4.PREREG_V1)), fixed_inputs={k: v for k, v in ver.items() if k != "frozen_r05"}, git_head_at_run=P4.R8.git_head())
json.dump(rec, open(f"{REP}/freeze_record.json", "w"), indent=1, default=str)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `phase4/scripts/{f}` | `{h}` |" for f, h in p4.items()] + [f"| `R10/scripts/{f}` | `{h}` |" for f, h in scripts.items()] + [f"| `{f}` | `{h}` |" for f, h in prereg.items()] + [f"| tables | {len(tables)} files, sha256 in freeze_record.json |", f"| git HEAD at run | {rec['git_head_at_run']} |"]
open(f"{REP}/freeze_record.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
