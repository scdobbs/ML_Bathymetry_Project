"""R06 §6.1/6.3 — round-1 freeze record: sha256 of every R06 script (training / model / inference / scoring / drivers),
the combined training-code hash recorded in each train_log.json, the frozen-input hashes, git HEAD and the tag to apply.
Also checks that every fold's train_log.json carries the same training-code hash as the committed sources."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r06_common as R, r04_common as C4
HERE = os.path.dirname(os.path.abspath(__file__)); REP = os.path.dirname(HERE); m = R.start("freeze_r06")
scripts = sorted(f for f in os.listdir(HERE) if f.endswith((".py", ".sbatch", ".txt")) and f != "__pycache__")
code = {f: R.sha256_file(f"{HERE}/{f}") for f in scripts}
logs = sorted(glob.glob(f"{R.MODELS}/*/seed*/fold_*/train_log.json")); hashes = {}
for f in logs: hashes.setdefault(json.load(open(f))["code_hash"], []).append(f.split("/models/")[1])
rec = dict(scripts_sha256=code, training_code_sha256=R.code_hash(), training_code_files=["r06_common.py", "build_inputs_r06.py", "r06_model.py", "train_r06.py"],
           train_logs=len(logs), training_code_hashes_in_logs={h: len(v) for h, v in hashes.items()}, frozen_inputs=R.check_frozen(verbose=False),
           phase0_freeze_commit=R.PHASE0_FREEZE_COMMIT, git_head_at_run=R.git_head(), tag="phase2-r1",
           inputs_leak_test=json.load(open(f"{R.INPUTS}/leak_test.json")) if os.path.exists(f"{R.INPUTS}/leak_test.json") else None)
rec["inputs_leak_test"] = {k: v for k, v in rec["inputs_leak_test"].items() if k != "rows"} if rec["inputs_leak_test"] else None
json.dump(rec, open(f"{REP}/freeze_record.json", "w"), indent=1)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `{f}` | `{h}` |" for f, h in code.items()] + [f"| training code (r06_common + build_inputs_r06 + r06_model + train_r06) | `{rec['training_code_sha256']}` |",
      f"| training-code hashes found in the {len(logs)} train_log.json files | {rec['training_code_hashes_in_logs']} |", f"| input leak test | {rec['inputs_leak_test']} |",
      f"| phase0-freeze commit | {R.PHASE0_FREEZE_COMMIT} |", f"| git HEAD at run | {rec['git_head_at_run']} |", f"| tag | phase2-r1 (applied to the R06 commit) |"]
open(f"{REP}/freeze_record.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
if len(hashes) != 1 or list(hashes)[0] != rec["training_code_sha256"]: print("WARNING: training-code hash in logs differs from the committed sources:", rec["training_code_hashes_in_logs"], "vs", rec["training_code_sha256"])
