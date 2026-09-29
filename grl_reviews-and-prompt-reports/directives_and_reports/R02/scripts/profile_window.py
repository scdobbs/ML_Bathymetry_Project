"""Profile the per-window cost of mosaic_infer.run_windows on NA080 (the slow unit) and NA090 (fast)."""
import sys, time, cProfile, pstats, io, numpy as np
sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project")
from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import load_manifest
DATA = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy"; pairs = load_manifest(DATA)
for pid in ["NA090__MGDS_31212", "NA080__MGDS_31290"]:
    row = pairs[pairs.pair_id == pid].iloc[0]
    t = time.time(); g = ecp.pair_setup(DATA, row, 256); print(f"{pid}: pair_setup {time.time()-t:.1f}s  HR {g['H']}x{g['W']} LR {g['lr_arr'].shape} lr_px={g['lr_px']} lr_crs==dst_crs: {g['lr_crs']==g['dst_crs']} lrv_crs==dst: {g['lrv_crs']==g['dst_crs']}", flush=True)
    H, W = g["H"], g["W"]; stride = 160
    wins = [(r, c) for r in range(0, H - 256 + 1, stride) for c in range(0, W - 256 + 1, stride)]
    rng = np.random.default_rng(0); pick = [wins[i] for i in rng.choice(len(wins), 20, replace=False)]
    for name, fn in [("tile_ship_side", lambda r0, c0: ecp.tile_ship_side(g, r0, c0, 256)),
                     ("build_inputs", lambda r0, c0: ecp.build_inputs(*ecp.tile_ship_side(g, r0, c0, 256), 256, 0.5)),
                     ("label_median", lambda r0, c0: (lambda hr, hv: float(np.nanmedian(np.where(hv, hr, np.nan))))(g["hr_arr"][r0:r0+256, c0:c0+256], g["hrv_arr"][r0:r0+256, c0:c0+256]))]:
        t = time.time()
        for r0, c0 in pick: fn(r0, c0)
        print(f"   {name:16s} {(time.time()-t)/len(pick)*1000:8.1f} ms/window", flush=True)
    pr = cProfile.Profile(); pr.enable()
    for r0, c0 in pick[:5]: ecp.build_inputs(*ecp.tile_ship_side(g, r0, c0, 256), 256, 0.5)
    pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12); print(s.getvalue()[:3000], flush=True)
