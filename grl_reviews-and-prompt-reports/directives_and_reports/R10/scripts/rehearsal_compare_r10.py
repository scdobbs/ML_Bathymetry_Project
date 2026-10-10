"""R10 §A2 — compare the rehearsal build of each listed pair with the frozen development artifacts: tile keys / r0 / c0 / s_lr
and A_fine (tiles_v1.1), F_hr / joint_v1 (R04 fields), feature columns (R04 feature table), the interior flag at D1 (R08b
lists), A_fine_int at D1 (R09 tiles_int). Writes compare_build.json and prints per pair / item."""
import json, os, sys
import numpy as np, pandas as pd, rasterio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, bar_v1_1 as B
R8, R9, RB = P4.R8, P4.R9, P4.R9.RB
if __name__ == "__main__":
    pairs = P4.split_pairs(sys.argv[1]); m, root, rec = P4.start("rehearsal_compare_r10.py", "rehearsal", pairs); tiles = R8.load_tiles(); feats = R8.load_features(); TI = pd.read_parquet(f"{R9.OUT_INT}/tiles_int.parquet").set_index("tile_id"); out = {}
    for pid in pairs:
        d = f"{root}/build/{pid}"; t = pd.read_parquet(f"{d}/tiles_v1_1.parquet").set_index("tile_id"); ref = tiles[tiles.pair == pid].set_index("tile_id"); r = {}
        r["tile_keys"] = dict(identical=bool(set(t.index) == set(ref.index)), n_build=len(t), n_frozen=len(ref)); ref = ref.reindex(t.index)
        for c in ("r0", "c0", "s_lr"): r[c] = dict(identical=bool(np.array_equal(t[c].values, ref[c].values)))
        r["A_fine"] = dict(max_abs_diff=float(np.max(np.abs(t.A_fine.values - ref.A_fine.values))))
        for name in ("F_hr", "joint_v1"):
            with rasterio.open(f"{root}/fields/{pid}/{name}.tif") as a, rasterio.open(f"{R8.R04_FIELDS}/{pid}/{name}.tif") as b:
                A, Bm = a.read(1), b.read(1); assert A.shape == Bm.shape
                if name == "F_hr": va, vb = np.isfinite(A), np.isfinite(Bm); r[name] = dict(valid_mask_identical=bool(np.array_equal(va, vb)), max_abs_diff=float(np.max(np.abs(A[va & vb] - Bm[va & vb]))) if (va & vb).any() else 0.0)
                else: r[name] = dict(identical=bool(np.array_equal(A, Bm)))
        f = pd.read_parquet(f"{d}/features.parquet"); fr = feats[feats.pair == pid]; key = ["tile_id", "block_id"]; f = f.set_index(key); fr = fr.set_index(key).reindex(f.index); cols = [c for c in B.ALL if c in f.columns]
        a = f[cols].values.astype(float); b = fr[cols].values.astype(float); nan_same = bool(np.array_equal(np.isnan(a), np.isnan(b))); ok = ~np.isnan(a) & ~np.isnan(b)
        r["features"] = dict(n_rows=len(f), n_rows_unmatched_in_frozen=int(np.isnan(b).all(axis=1).sum()), n_cols=len(cols), nan_pattern_identical=nan_same, max_abs_diff=float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0)
        ti = pd.read_parquet(f"{d}/tiles_int.parquet").set_index("tile_id"); l1 = set(pd.read_csv(f"{RB.OUT_INT}/lists/{pid}_D1.csv").tile_id)
        r["interior_D1"] = dict(identical=bool(set(ti[ti.interior_D1].index) == l1), n_build=int(ti.interior_D1.sum()), n_frozen=len(l1))
        ref_int = TI.A_fine_int_D1.reindex(ti.index).values; a = ti.A_fine_int_D1.values; both = np.isfinite(a) & np.isfinite(ref_int)
        r["A_fine_int_D1"] = dict(nan_pattern_identical=bool(np.array_equal(np.isfinite(a), np.isfinite(ref_int))), max_abs_diff=float(np.max(np.abs(a[both] - ref_int[both]))) if both.any() else 0.0)
        out[pid] = r; print(pid, json.dumps(r))
    json.dump(out, open(f"{root}/build/compare_build.json", "w"), indent=1)
