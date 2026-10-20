"""Self-test of score_r06.py on SYNTHETIC predictions (random noise + a fraction of the true log10 A_fine, plus fake
train logs); exercises every table and comparison. Never used for results."""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r06_common as R, r04_common as C4
out = f"{R.ROOT}/selftest_models"; rng = np.random.default_rng(0)
tiles = pd.read_parquet(R.TILES); blocks = pd.read_parquet(R.BLOCKS); B = blocks.pivot(index="tile_id", columns="block_id", values="A_fine")
def fake(df, skill):
    d = df[["tile_id", "unit", "pair"]].copy(); A = np.log10(np.maximum(B.loc[d.tile_id].values, 1e-4)); A = np.where(np.isfinite(A), A, np.nanmean(A))
    P = skill * A + rng.normal(0, 0.3, A.shape)
    for j in range(16): d[f"blk{j}"] = P[:, j]
    d["tile_pred_m"] = np.sqrt(np.mean(10 ** (2 * P), axis=1)); return d
for v, seeds, skill in (("V0", [R.SEED_MAIN], 0.3), ("V1", R.SEEDS_V1, 0.5), ("V2", [R.SEED_MAIN], 0.5), ("V3", [R.SEED_MAIN], 0.4)):
    for s in seeds:
        for f in list(C4.UNITS) + (["ALL"] if s == R.SEED_MAIN else []):
            d = f"{out}/{v}/seed{s}/fold_{f}"; os.makedirs(d, exist_ok=True)
            sub = tiles[tiles.unit == ("lu16" if f == "ALL" else f)]; fake(sub, skill).to_parquet(f"{d}/pred.parquet")
            json.dump(dict(variant=v, fold=f, seed=s, wall_total_s=1500.0, wall_train_s=1400.0, gpu="fake", s_per_step=0.2, params=1914081, steps=8000, batch=32, groups_per_batch=4, windows_per_group=8, lr=2e-4, weight_decay=1e-4, eta_min=1e-6, grad_clip=1.0, precision="bf16", optimizer="AdamW", schedule="cosine"), open(f"{d}/train_log.json", "w"))
print("synthetic predictions written to", out)
