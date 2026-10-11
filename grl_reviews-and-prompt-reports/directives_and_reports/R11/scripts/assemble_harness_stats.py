"""R11 §A3 — {pair: pair_stats} of the harness builds (role added) for phase4_score.py --stats."""
import json, sys
HR = sys.argv[1]; out = {}
for pid in ("TN399__MGDS_33068", "FK151121__MGDS_22383", "EX1202L2__MGDS_20815"):
    s = json.load(open(f"{HR}/build/{pid}/pair_stats.json")); s["role"] = "development"; out[pid] = s
json.dump(out, open(f"{HR}/harness_stats_build.json", "w"), indent=1); print({p: s["status"] for p, s in out.items()})
