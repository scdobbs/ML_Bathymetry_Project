"""R02 §3.1 helper — precompute the ship-side LR tiles (ecp.tile_ship_side, unchanged) for every mosaic window and
seam-check window of each pair, in parallel on CPU, so the GPU mosaic job does not spend 50-300 ms per window in the
full-source GDAL reprojection (job 45797098 timed out after 2 units). Output: lr_tile_cache/<pair_id>.npz with
r0, c0, lr (N, lr_px, lr_px) float32 (NaN where invalid), lv (N, lr_px, lr_px) bool. Same function, same values."""
import os, sys, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R02"); sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project")
import mosaic_infer as mi
from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import load_manifest
OUT = "/scratch/users/scdobbs/grl_review/R02/lr_tile_cache"; os.makedirs(OUT, exist_ok=True)
G = None
def work(rc):
    lr, lv = ecp.tile_ship_side(G, rc[0], rc[1], mi.WIN); return rc, lr, lv

if __name__ == "__main__":
    rim = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    pids = sys.argv[2].split(",") if len(sys.argv) > 2 else [p for u in list(mi.hc.FOLDS) + ["EW0207"] for p in mi.UNIT_PAIRS[u]]
    pairs = load_manifest(mi.DATA); nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    for pid in pids:
        if os.path.exists(f"{OUT}/{pid}.npz"): print(f"{pid}: cached, skip", flush=True); continue
        t0 = time.time(); row = pairs[pairs.pair_id == pid].iloc[0]; G = ecp.pair_setup(mi.DATA, row, mi.WIN)
        rows, cols = mi.window_grid(G["H"], G["W"], rim); stride = mi.WIN - 2 * rim
        wins = [(r, c) for r in rows for c in cols]
        for r0, c0 in mi.seam_picks(rows, cols, G["W"], rim):
            wins += [(r0, c0), (r0, c0 + stride // 2)]
        wins = sorted(set(wins))
        with Pool(nw) as pool: res = pool.map(work, wins, chunksize=32)
        np.savez_compressed(f"{OUT}/{pid}.npz", r0=np.array([r[0][0] for r in res], np.int32), c0=np.array([r[0][1] for r in res], np.int32),
                            lr=np.stack([r[1] for r in res]).astype(np.float32), lv=np.stack([r[2] for r in res]))
        print(f"{pid}: {len(wins)} windows (lr_px {G['lr_px']}) in {time.time()-t0:.0f}s with {nw} workers", flush=True)
