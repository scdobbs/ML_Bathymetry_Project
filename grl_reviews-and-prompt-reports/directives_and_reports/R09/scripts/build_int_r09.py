"""R09 Part A build: per development pair (and EW0207), the R08b distance map of the frozen joint_v1.tif (OAK-verified) and,
per tiles_v1.1 tile, A_fine_int and the interior-px count at D1 and at D125 (the 125 m stress cell), plus the interior-TILE
flag at D125 (R08b rule). The D1 interior-tile flag is recomputed and asserted equal to R08b's table. Writes
targets_int/tiles_int.parquet and targets_int/pairs.json."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r09_common as R9, r04_common as C4, targets_v1_int as TI
D = {}

def one_pair(pid):
    t0 = time.time(); row = C4.pair_row(D["m"], pid); tl = D["tiles"][D["tiles"].pair == pid]; ver = R9.RB.verify_field(pid); verF = R9.RB.verify_field(pid, "F_hr.tif")
    with rasterio.open(f"{R9.R8.R04_FIELDS}/{pid}/F_hr.tif") as s: F = s.read(1); hr_res = float(abs(s.transform.a))
    with rasterio.open(f"{R9.R8.R04_FIELDS}/{pid}/joint_v1.tif") as s: J = (s.read(1) == 1) & np.isfinite(F)
    lr_native = float(row["lr_native_res_m"]); assert abs(lr_native - float(tl.lr_native.iloc[0])) < 1e-9 and abs(hr_res - float(tl.hr_res.iloc[0])) < 1e-9
    D1, D125 = TI.d_px(lr_native, hr_res), TI.d_px(R9.CELL_STRESS_M, hr_res); dist = TI.distance_map(J)
    prev = D["r08b"].reindex(tl.tile_id); rows = []
    for r in tl.itertuples():
        sl = (slice(r.tr0, r.tr0 + TI.TILE), slice(r.tc0, r.tc0 + TI.TILE)); j, d, f = J[sl], dist[sl], F[sl]
        i1, f1 = TI.interior_tile(j, d, D1); assert bool(i1) == bool(prev.loc[r.tile_id, "interior_D1"]) and abs(f1 - prev.loc[r.tile_id, "frac_ge_D1"]) < 1e-9, (pid, r.tile_id)
        a1, n1 = TI.afine_int(f, j, d, D1); a2, n2 = TI.afine_int(f, j, d, D125); i2, f2 = TI.interior_tile(j, d, D125)
        rows.append(dict(tile_id=r.tile_id, pair=pid, unit=C4.UNIT_OF[pid], A_fine=r.A_fine, n_joint=int(j.sum()), interior_D1=i1, n_int_px_D1=n1, A_fine_int_D1=a1, interior_D125=i2, frac_ge_D125=f2, n_int_px_D125=n2, A_fine_int_D125=a2))
    df = pd.DataFrame(rows)
    info = dict(pair=pid, unit=C4.UNIT_OF[pid], lr_native_m=lr_native, hr_res_m=hr_res, D1_px=D1, D125_px=D125, n_tiles=len(df), n_interior_D1=int(df.interior_D1.sum()), n_defined_int_D1=int(df.A_fine_int_D1.notna().sum()),
                n_interior_D125=int(df.interior_D125.sum()), n_defined_int_D125=int(df.A_fine_int_D125.notna().sum()), A_fine_median_all=float(df.A_fine.median()), A_fine_int_D1_median=float(df.A_fine_int_D1.median()), A_fine_int_D125_median=float(df.A_fine_int_D125.median()),
                verification=dict(joint=ver["sha256"], F_hr=verF["sha256"]), seconds=time.time() - t0)
    return df, info

if __name__ == "__main__":
    workers = int(os.environ.get("R09_WORKERS", 4)); t0 = time.time(); m = R9.start("build_int_r09"); tiles = R9.R8.load_tiles(); D.update(m=m, tiles=tiles, r08b=pd.read_parquet(R9.R08B_INTERIOR).set_index("tile_id"))
    pairs = sorted(tiles.pair.unique()); C4.assert_no_sealed(pairs); assert not set(pairs) & set(C4.SEALED)
    with Pool(workers) as pool: outs = pool.map(one_pair, pairs, chunksize=1)
    df = pd.concat([o[0] for o in outs], ignore_index=True); df.to_parquet(f"{R9.OUT_INT}/tiles_int.parquet"); info = [o[1] for o in outs]
    R9.R8.dump(dict(definition=dict(min_int_px=TI.MIN_INT_PX, D1="ceil(lr_native / hr_res) px", D125=f"ceil({R9.CELL_STRESS_M:.0f} / hr_res) px", interior_tile_frac=TI.INTERIOR_TILE_FRAC), targets_v1_int_sha256=R9.R8.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets_v1_int.py")), pairs=info), f"{R9.OUT_INT}/pairs.json")
    for i in info: print(f"  {i['unit']} {i['pair']}: D1 {i['D1_px']} px: interior tiles {i['n_interior_D1']}, A_fine_int defined {i['n_defined_int_D1']} of {i['n_tiles']}; D125 {i['D125_px']} px: interior {i['n_interior_D125']}, defined {i['n_defined_int_D125']} ({i['seconds']:.0f} s)", flush=True)
    print(f"done ({time.time()-t0:.0f} s)")
