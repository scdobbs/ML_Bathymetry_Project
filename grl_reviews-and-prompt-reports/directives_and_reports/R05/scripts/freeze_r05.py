"""R05 §5.3 — freeze record: sha256 of targets_v1.py (R04, unchanged), eval_v1_1.py, features_v1_1.py, bar_v1_1.py,
run_bar_v1_1.py, tiles_v1_1.parquet, blocks_v1_1.parquet; manifest sha256; contract version; unit set; rules in force."""
import json, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common as C5, r04_common as C, eval_v1_1 as E
HERE = os.path.dirname(os.path.abspath(__file__)); REP = os.path.dirname(HERE); m = C5.start("freeze_r05")
r04 = json.load(open(f"{os.path.dirname(REP)}/R04/freeze_record.json"))
code = {"targets_v1.py (R04, unchanged)": C.sha256_file(f"{C.R04_SCRIPTS}/targets_v1.py"), **{f: C.sha256_file(f"{HERE}/{f}") for f in ("eval_v1_1.py", "features_v1_1.py", "bar_v1_1.py", "run_bar_v1_1.py", "build_tiles_v1_1.py", "r05_common.py")},
        "r04_common.py (R04, unchanged)": C.sha256_file(f"{C.R04_SCRIPTS}/r04_common.py")}
assert code["targets_v1.py (R04, unchanged)"] == r04["code"]["targets_v1.py"], "targets_v1.py changed since R04"
rec = dict(code=code, tiles_v1_1_parquet=C.sha256_file(f"{C5.ROOT}/tiles/tiles_v1_1.parquet"), blocks_v1_1_parquet=C.sha256_file(f"{C5.ROOT}/tiles/blocks_v1_1.parquet"),
           manifest=dict(path=C.MANIFEST, sha256=m.attrs["sha256"], matches_directive=m.attrs["sha256"] == C.MANIFEST_SHA256), contract=dict(version="2.1", sha256=C.sha256_file(C.CONTRACT), matches_directive=C.sha256_file(C.CONTRACT) == C.CONTRACT_SHA256),
           units_version=C.UNITS_VERSION, n_units=len(C.UNITS), n_pairs=len(C.DEV_PAIRS), ew0207_separate=True, rules=dict(pair_eligibility_min_tiles=E.MIN_TILES_PER_PAIR, detection_bounds_per_pair=list(E.DETECTION_BOUNDS), e1m_threshold_m=C.E1M_THRESHOLD_M, tile_floor=C5.TILE_FLOOR, primary_endpoint="A_fine / tile / within-pair Spearman rho, median over pairs then units"),
           r04_commit="8b1d52db2ed0de661c606a561fbafa61ce6cd6a6", git_head_at_run=subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, capture_output=True, text=True).stdout.strip())
json.dump(rec, open(f"{REP}/freeze_record.json", "w"), indent=1)
md = ["| item | sha256 / value |", "|---|---|"] + [f"| `{k}` | `{v}` |" for k, v in code.items()] + [f"| `tiles_v1_1.parquet` | `{rec['tiles_v1_1_parquet']}` |", f"| `blocks_v1_1.parquet` | `{rec['blocks_v1_1_parquet']}` |",
      f"| manifest v2.1 | `{rec['manifest']['sha256']}` (matches directive: {rec['manifest']['matches_directive']}) |", f"| contract | v2.1, `{rec['contract']['sha256']}` (matches directive: {rec['contract']['matches_directive']}) |",
      f"| unit set | {rec['units_version']}: {rec['n_units']} units / {rec['n_pairs']} pairs; EW0207 separate; Venere removed |", f"| rules in force | pair eligibility ≥ {E.MIN_TILES_PER_PAIR} tiles; detection per pair in [{E.DETECTION_BOUNDS[0]}, {E.DETECTION_BOUNDS[1]}]; E_1m threshold {C.E1M_THRESHOLD_M} m; tile floor {C5.TILE_FLOOR} |",
      f"| primary endpoint | {rec['rules']['primary_endpoint']} |", f"| R04 commit (tag r04-as-run) | {rec['r04_commit']} |"]
open(f"{REP}/freeze_record.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
