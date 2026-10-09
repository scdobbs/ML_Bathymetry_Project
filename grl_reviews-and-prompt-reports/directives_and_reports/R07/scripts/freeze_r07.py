"""R07 §5.1/5.3 — freeze record: sha256 of every R07 script, the combined code hash recorded in the training / pretraining
logs, the frozen-input checks, git HEAD and the tag to apply (phase3-r1)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r07_common as R7
HERE = os.path.dirname(os.path.abspath(__file__)); REP = os.path.dirname(HERE); R7.start("freeze_r07", check_corpus_files=False)
scripts = sorted(f for f in os.listdir(HERE) if f.endswith((".py", ".sbatch", ".txt"))); code = {f: R7.sha256_file(f"{HERE}/{f}") for f in scripts}
logs = sorted(glob.glob(f"{R7.MODELS}/*/seed*/fold_*/train_log.json")); hashes = {}
for f in logs: hashes.setdefault(json.load(open(f))["code_hash"], []).append(f.split("/models/")[1])
pre = {os.path.basename(os.path.dirname(f)): json.load(open(f))["code_hash"] for f in glob.glob(f"{R7.PRETRAIN}/*/pretrain_log.json")}
bodies = {b: R7.sha256_file(f"{R7.PRETRAIN}/{b}/body_{b}.pt") for b in R7.BODIES if os.path.exists(f"{R7.PRETRAIN}/{b}/body_{b}.pt")}
rec = dict(scripts_sha256=code, code_sha256=R7.code_hash(), code_files=R7.CODE_FILES + ["R06/scripts/r06_model.py"], train_logs=len(logs), code_hashes_in_train_logs={h: len(v) for h, v in hashes.items()},
           code_hashes_in_pretrain_logs=pre, pretrained_bodies_sha256=bodies, frozen_r05=R7.R6.check_frozen(verbose=False), r06_freeze=R7.check_r06_freeze(), corpus=R7.check_corpus(require=False),
           cache_manifest=json.load(open(f"{R7.CACHE}/cache_manifest.json")) if os.path.exists(f"{R7.CACHE}/cache_manifest.json") else None, phase0_freeze_commit=R7.PHASE0_FREEZE_COMMIT, phase2_r1_commit=R7.PHASE2_R1_COMMIT,
           git_head_at_run=R7.git_head(), tag="phase3-r1", scratch_equiv=json.load(open(f"{R7.CHECKS}/scratch_equiv/result.json")) if os.path.exists(f"{R7.CHECKS}/scratch_equiv/result.json") else None,
           equiv_windows={k: v for k, v in json.load(open(f"{R7.CHECKS}/equiv_windows.json")).items() if k != "rows"} if os.path.exists(f"{R7.CHECKS}/equiv_windows.json") else None)
json.dump(rec, open(f"{REP}/freeze_record.json", "w"), indent=1, default=str)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `{f}` | `{h}` |" for f, h in code.items()] + [f"| code (R07 sources + R06 r06_model.py) | `{rec['code_sha256']}` |", f"| code hashes in the {len(logs)} train_log.json files | {rec['code_hashes_in_train_logs']} |",
      f"| code hashes in the pretraining logs | {pre} |", f"| pretrained bodies | {bodies} |", f"| phase0-freeze / phase2-r1 commits | {R7.PHASE0_FREEZE_COMMIT} / {R7.PHASE2_R1_COMMIT} |", f"| git HEAD at run | {rec['git_head_at_run']} |", "| tag | phase3-r1 (applied to the R07 commit) |"]
open(f"{REP}/freeze_record.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
