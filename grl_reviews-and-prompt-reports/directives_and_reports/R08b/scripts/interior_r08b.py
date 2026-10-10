"""R08b §2 — interior tiles. Per pair: the joint-valid mask of the frozen R04 joint_v1.tif (OAK-verified), padded with one
invalid px, Euclidean distance transform to the nearest invalid px; D1 = ceil(lr_native / hr_res) px from manifest v2.1
(hr_res from the mosaic), D2 = 2 D1; a tiles_v1.1 tile is interior at D if >= 95 % of its joint-valid px have distance >= D.
Writes interior/tiles_interior.parquet (every tile: n_joint, frac_ge_D1, frac_ge_D2, interior_D1, interior_D2), per-pair
interior tile lists interior/lists/<pair>_D1.csv / _D2.csv, and interior/pairs.json (D1 px and m, counts, A_fine medians)."""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r08b_common as RB, r04_common as C4, targets_v1 as T
D = {}

def one_pair(pid):
    t0 = time.time(); row = C4.pair_row(D["m"], pid); tl = D["tiles"][D["tiles"].pair == pid]; ver = RB.verify_field(pid)
    with rasterio.open(f"{RB.R8.R04_FIELDS}/{pid}/joint_v1.tif") as s: J = s.read(1) == 1; hr_res = float(abs(s.transform.a)); H, W = s.shape
    lr_native = float(row["lr_native_res_m"]); assert abs(lr_native - float(tl.lr_native.iloc[0])) < 1e-9 and abs(hr_res - float(tl.hr_res.iloc[0])) < 1e-9
    D1 = int(math.ceil(lr_native / hr_res)); Dpx = {k: v * D1 for k, v in RB.D_LEVELS.items()}
    Jp = np.pad(J, 1, constant_values=False); d = distance_transform_edt(Jp)[1:-1, 1:-1]; del Jp
    rows = []
    for r in tl.itertuples():
        sl = (slice(r.tr0, r.tr0 + T.TILE), slice(r.tc0, r.tc0 + T.TILE)); j = J[sl]; n = int(j.sum()); dd = d[sl][j]
        rec = dict(tile_id=r.tile_id, pair=pid, unit=C4.UNIT_OF[pid], n_joint=n, A_fine=r.A_fine)
        for k, px in Dpx.items(): fr = float((dd >= px).mean()) if n else 0.0; rec[f"frac_ge_{k}"] = fr; rec[f"interior_{k}"] = bool(n > 0 and fr >= RB.INTERIOR_FRAC)
        rows.append(rec)
    df = pd.DataFrame(rows); os.makedirs(f"{RB.OUT_INT}/lists", exist_ok=True)
    for k in Dpx: df[df[f"interior_{k}"]][["tile_id"]].to_csv(f"{RB.OUT_INT}/lists/{pid}_{k}.csv", index=False)
    info = dict(pair=pid, unit=C4.UNIT_OF[pid], lr_native_m=lr_native, hr_res_m=hr_res, D1_px=D1, D1_m=D1 * hr_res, D2_px=2 * D1, D2_m=2 * D1 * hr_res, n_tiles=len(df), mosaic_hw=[H, W], joint_px=int(J.sum()),
                A_fine_median_all=float(df.A_fine.median()), verification=ver, seconds=time.time() - t0)
    for k in Dpx: s = df[df[f"interior_{k}"]]; info[f"n_interior_{k}"] = int(len(s)); info[f"frac_interior_{k}"] = float(len(s) / len(df)); info[f"A_fine_median_{k}"] = float(s.A_fine.median()) if len(s) else np.nan
    return df, info

if __name__ == "__main__":
    workers = int(os.environ.get("R08B_WORKERS", 4)); t0 = time.time(); m = RB.start("interior_r08b"); tiles = RB.R8.load_tiles(); D.update(m=m, tiles=tiles)
    pairs = sorted(tiles.pair.unique()); C4.assert_no_sealed(pairs)
    with Pool(workers) as pool: outs = pool.map(one_pair, pairs, chunksize=1)
    df = pd.concat([o[0] for o in outs], ignore_index=True); df.to_parquet(f"{RB.OUT_INT}/tiles_interior.parquet")
    info = [o[1] for o in outs]; RB.R8.dump(dict(rule=dict(interior_frac=RB.INTERIOR_FRAC, D_levels=RB.D_LEVELS, D1="ceil(lr_native / hr_res) px", padding="1 invalid px around the mosaic"), pairs=info), f"{RB.OUT_INT}/pairs.json")
    for i in info: print(f"  {i['unit']} {i['pair']}: D1 {i['D1_px']} px ({i['D1_m']:.0f} m); tiles {i['n_tiles']} -> D1 {i['n_interior_D1']} ({i['frac_interior_D1']:.2f}), D2 {i['n_interior_D2']} ({i['frac_interior_D2']:.2f}); median A_fine all {i['A_fine_median_all']:.4f} D1 {i['A_fine_median_D1']:.4f} ({i['seconds']:.0f} s)", flush=True)
    print(f"done ({time.time()-t0:.0f} s)")
