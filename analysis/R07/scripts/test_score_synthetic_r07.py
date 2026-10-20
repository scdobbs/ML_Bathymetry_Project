"""Scorer self-test on SYNTHETIC predictions (never used for results): builds fake pred.parquet files for P-V1 / P-V2
(3 seeds x 16 folds + ALL) and V2 seeds 20261003 / 20261004 from the R06 V1 predictions plus noise under
checks/selftest_models/, then runs score_r07.py with R07_MODELS / R07_SCORES pointed there and checks the output exists."""
import os, subprocess, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r07_common as R7, r06_common as R6, r04_common as C4
root = f"{R7.CHECKS}/selftest_models"; sc = f"{R7.CHECKS}/selftest_scores"; rng = np.random.default_rng(0)
folds = list(C4.UNITS) + ["ALL"]
for arm, seeds in (("P-V1", R7.SEEDS), ("P-V2", R7.SEEDS), ("V2", R7.SEEDS[1:])):
    for s in seeds:
        for f in folds:
            src = f"{R6.MODELS}/V1/seed{R7.SEED_MAIN}/fold_{f}/pred.parquet"; d = pd.read_parquet(src)
            for j in range(16): d[f"blk{j}"] = d[f"blk{j}"] + rng.normal(0, 0.1, len(d)) + (0.05 if arm.startswith("P") else 0.0)
            d["tile_pred_m"] = np.sqrt(np.mean(10 ** (2 * d[[f"blk{j}" for j in range(16)]].values), axis=1))
            out = f"{root}/{arm}/seed{s}/fold_{f}"; os.makedirs(out, exist_ok=True); d.to_parquet(f"{out}/pred.parquet")
env = dict(os.environ, R07_MODELS=root, R07_SCORES=sc)
r = subprocess.run([sys.executable, f"{os.path.dirname(os.path.abspath(__file__))}/score_r07.py"], env=env, capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-3000:]); assert r.returncode == 0 and os.path.exists(f"{sc}/R07_scores.md"), "scorer self-test failed"; print("SCORER SELF-TEST OK")
