"""R05 §2.1 — tiles_v1.1: selection on R04's tiles_v1 / blocks_v1 (fields and targets_v1 untouched): keep a tile iff
>= 50 % of its central 128 x 128 px are joint-valid; blocks inside a dropped tile are dropped; the block rule is unchanged."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common as C
m = C.start("build_tiles_v1_1"); OUT = f"{C.ROOT}/tiles"; os.makedirs(OUT, exist_ok=True)
t = pd.read_parquet(f"{C.R04}/tiles/tiles_v1.parquet"); b = pd.read_parquet(f"{C.R04}/tiles/blocks_v1.parquet")
keep = t.frac_joint >= C.TILE_FLOOR; t1 = t[keep].reset_index(drop=True); b1 = b[b.tile_id.isin(t1.tile_id)].reset_index(drop=True)
t1.to_parquet(f"{OUT}/tiles_v1_1.parquet"); b1.to_parquet(f"{OUT}/blocks_v1_1.parquet")
rem_t = t[~keep]; rem_b = b[~b.tile_id.isin(t1.tile_id)]
per_pair = pd.DataFrame(dict(unit=t.groupby("pair").unit.first(), tiles_v1=t.groupby("pair").size(), tiles_removed=rem_t.groupby("pair").size(), tiles_v1_1=t1.groupby("pair").size(),
                             blocks_v1=b.groupby("pair").size(), blocks_removed=rem_b.groupby("pair").size(), kept_blocks_removed=rem_b[rem_b.kept].groupby("pair").size(),
                             blocks_v1_1=b1.groupby("pair").size(), kept_blocks_v1_1=b1[b1.kept].groupby("pair").size())).fillna(0).astype({c: int for c in ["tiles_v1", "tiles_removed", "tiles_v1_1", "blocks_v1", "blocks_removed", "kept_blocks_removed", "blocks_v1_1", "kept_blocks_v1_1"]})
per_unit = per_pair.groupby("unit").sum(numeric_only=True)
per_pair.to_csv(f"{OUT}/tiles_v1_1_removed_per_pair.csv"); per_unit.to_csv(f"{OUT}/tiles_v1_1_removed_per_unit.csv")
summ = dict(tile_floor=C.TILE_FLOOR, tiles_v1=int(len(t)), tiles_removed=int(len(rem_t)), tiles_v1_1=int(len(t1)), min_n_joint_removed=int(rem_t.n_joint.min()), blocks_v1=int(len(b)), blocks_removed=int(len(rem_b)),
            kept_blocks_removed=int(rem_b.kept.sum()), blocks_v1_1=int(len(b1)), kept_blocks_v1_1=int(b1.kept.sum()), pairs_below_10_tiles=per_pair[per_pair.tiles_v1_1 < C.MIN_TILES_PER_PAIR].index.tolist())
json.dump(summ, open(f"{OUT}/tiles_v1_1_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)); pd.set_option("display.width", 250); print(per_unit.to_string())
