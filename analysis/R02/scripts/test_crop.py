"""Test a cropped-source variant of tile_ship_side: (1) bit-identity of x / lr_valid_hr / lr_up_m against the stored clean
patches for 120 random tiles across all 9 pairs; (2) per-window timing, full-source vs cropped, on NA080 and TN399."""
import sys, time, math, glob, os, hashlib, numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project")
from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import load_manifest
DATA = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy"; P = "/scratch/users/scdobbs/grl_review/R02/patches_clean"
pairs = load_manifest(DATA)

def _crop_reproject(arr, tr, crs, bounds, out_px, dst_crs, resampling, margin=4):
    """Crop `arr` to the source pixels covering `bounds` (+margin) before GDAL reproject. Requires crs == dst_crs."""
    assert crs == dst_crs
    w = rasterio.windows.from_bounds(*bounds, transform=tr)
    r0 = max(0, int(math.floor(w.row_off)) - margin); c0 = max(0, int(math.floor(w.col_off)) - margin)
    r1 = min(arr.shape[0], int(math.ceil(w.row_off + w.height)) + margin); c1 = min(arr.shape[1], int(math.ceil(w.col_off + w.width)) + margin)
    dst = np.full((out_px, out_px), np.nan, dtype=np.float32)
    if r1 <= r0 or c1 <= c0: return dst
    sub = np.ascontiguousarray(arr[r0:r1, c0:c1]); sub_tr = tr * Affine.translation(c0, r0)
    dst_tr = rasterio.transform.from_bounds(*bounds, out_px, out_px)
    reproject(source=sub, destination=dst, src_transform=sub_tr, src_crs=crs, dst_transform=dst_tr, dst_crs=dst_crs,
              resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return dst

def tile_ship_side_crop(g, r0, c0, win):
    b = rasterio.windows.bounds(rasterio.windows.Window(c0, r0, win, win), g["transform"])
    lr = _crop_reproject(g["lr_arr"], g["lr_tr"], g["lr_crs"], b, g["lr_px"], g["dst_crs"], Resampling.average)
    lv = _crop_reproject(g["lrv_raw"], g["lrv_tr"], g["lrv_crs"], b, g["lr_px"], g["dst_crs"], Resampling.nearest) == 1
    lv = lv & np.isfinite(lr)
    return np.where(lv, lr, np.nan).astype(np.float32), lv

hb = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
idx = pd.read_parquet(f"{P}/patch_index.parquet"); rng = np.random.default_rng(1)
sample = idx.groupby("pair_id", group_keys=False).apply(lambda d: d.sample(n=min(14, len(d)), random_state=1))
n_ok = n_lr_ok = n = 0; bad = []
for pid, sub in sample.groupby("pair_id"):
    row = pairs[pairs.pair_id == pid].iloc[0]; g = ecp.pair_setup(DATA, row, 256)
    print(f"{pid}: lr_crs==dst {g['lr_crs']==g['dst_crs']} lrv_crs==dst {g['lrv_crs']==g['dst_crs']} LR {g['lr_arr'].shape}", flush=True)
    for _, t in sub.iterrows():
        lr_a, lv_a = ecp.tile_ship_side(g, int(t.r0), int(t.c0), 256); lr_b, lv_b = tile_ship_side_crop(g, int(t.r0), int(t.c0), 256)
        ia = ecp.build_inputs(lr_a, lv_a, 256, 0.5); ib = ecp.build_inputs(lr_b, lv_b, 256, 0.5)
        z = np.load(f"{P}/{t.split}/{t.file}", allow_pickle=True)
        lr_same = hb(np.nan_to_num(lr_a, nan=-1e30)) == hb(np.nan_to_num(lr_b, nan=-1e30)) and hb(lv_a) == hb(lv_b)
        same = lr_same and hb(ia["x"]) == hb(ib["x"]) == hb(z["x"][0]) and hb(ia["lr_valid_hr"]) == hb(ib["lr_valid_hr"]) == hb(z["lr_valid_hr"][0]) and hb(ia["lr_up_m"]) == hb(ib["lr_up_m"])
        n += 1; n_ok += same; n_lr_ok += lr_same
        if not same: bad.append((pid, int(t.r0), int(t.c0), float(np.nanmax(np.abs(lr_a - lr_b))) if lr_a.shape == lr_b.shape else -1))
print(f"\nIDENTITY: {n_ok}/{n} tiles bit-identical (full-source vs cropped vs stored); LR-level identical {n_lr_ok}/{n}; bad={bad[:10]}", flush=True)
for pid in ["NA080__MGDS_31290", "TN399__MGDS_30373"]:
    row = pairs[pairs.pair_id == pid].iloc[0]; g = ecp.pair_setup(DATA, row, 256)
    wins = [(r, c) for r in range(0, g["H"] - 256 + 1, 160) for c in range(0, g["W"] - 256 + 1, 160)]
    pick = [wins[i] for i in rng.choice(len(wins), 200, replace=False)]
    for name, fn in [("full", ecp.tile_ship_side), ("crop", tile_ship_side_crop)]:
        t = time.time()
        for r0, c0 in pick: ecp.build_inputs(*fn(g, r0, c0, 256), 256, 0.5)
        print(f"  {pid} {name}: {(time.time()-t)/len(pick)*1000:.1f} ms/window (tile_ship_side + build_inputs)", flush=True)
