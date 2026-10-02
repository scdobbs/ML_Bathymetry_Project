"""R04 §6.3 — freeze record: sha256 of the frozen code (targets_v1, eval_v1, features_v1, bar_v1, r04_common), of
tiles_v1.parquet, the manifest and the contract; the frozen unit list; the E_1m threshold. Writes freeze_record.json
next to the report and prints the markdown block."""
import json, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r04_common as C
HERE = os.path.dirname(os.path.abspath(__file__)); REP = os.path.dirname(HERE)
m = C.start("freeze")
rec = dict(code={f: C.sha256_file(f"{HERE}/{f}") for f in ("targets_v1.py", "eval_v1.py", "features_v1.py", "bar_v1.py", "r04_common.py", "build_tiles_targets.py", "build_features.py", "run_bar.py", "contract_v2_1_validate.py")},
           tiles_v1_parquet=C.sha256_file(f"{C.ROOT}/tiles/tiles_v1.parquet"), blocks_v1_parquet=C.sha256_file(f"{C.ROOT}/tiles/blocks_v1.parquet"),
           manifest=dict(path=C.MANIFEST, sha256=m.attrs["sha256"], matches_directive=m.attrs["sha256"] == C.MANIFEST_SHA256),
           contract=dict(path=C.CONTRACT, version="2.1", sha256=C.sha256_file(C.CONTRACT), matches_directive=C.sha256_file(C.CONTRACT) == C.CONTRACT_SHA256),
           units_version=C.UNITS_VERSION, units=C.UNITS, ew0207_separate=C.EW0207, not_used_no_k=C.NOT_USED_NO_K, sealed=list(C.SEALED), e1m_threshold_m=C.E1M_THRESHOLD_M,
           git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, capture_output=True, text=True).stdout.strip())
json.dump(rec, open(f"{REP}/freeze_record.json", "w"), indent=1)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `{k}` | `{v}` |" for k, v in rec["code"].items()] + [f"| `tiles_v1.parquet` | `{rec['tiles_v1_parquet']}` |", f"| `blocks_v1.parquet` | `{rec['blocks_v1_parquet']}` |",
      f"| manifest v2.1 | `{rec['manifest']['sha256']}` (matches directive: {rec['manifest']['matches_directive']}) |", f"| contract | v2.1, `{rec['contract']['sha256']}` (matches directive: {rec['contract']['matches_directive']}) |",
      f"| unit set | {rec['units_version']}: {len(C.UNITS)} units / {len(C.DEV_PAIRS)} pairs; EW0207 separate; no-k not used: {len(C.NOT_USED_NO_K)} |", f"| E_1m threshold | {rec['e1m_threshold_m']} m |", f"| git HEAD at freeze run | {rec['git_head']} |"]
print("\n".join(md)); open(f"{REP}/freeze_record.md", "w").write("\n".join(md) + "\n")
