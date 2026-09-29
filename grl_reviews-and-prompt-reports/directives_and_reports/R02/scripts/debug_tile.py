"""Why does AT37-13 tile (768,640) differ between full-source and cropped reprojection? Compare masks/NaNs across margins."""
import sys, numpy as np, rasterio
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R02/profile"); sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project")
import test_crop as tc
from preprocessing import extract_clean_patches as ecp
from rasterio.enums import Resampling
row = tc.pairs[tc.pairs.pair_id == "AT37-13__MGDS_31199"].iloc[0]; g = ecp.pair_setup(tc.DATA, row, 256)
r0, c0 = 768, 640; b = rasterio.windows.bounds(rasterio.windows.Window(c0, r0, 256, 256), g["transform"])
lr_a, lv_a = ecp.tile_ship_side(g, r0, c0, 256)
print("full: n_valid", lv_a.sum(), "n_nan", np.isnan(lr_a).sum(), "lr_px", g["lr_px"])
print("src window:", rasterio.windows.from_bounds(*b, transform=g["lr_tr"]))
for m in (4, 8, 16, 64):
    lr_b, lv_b = None, None
    lr = tc._crop_reproject(g["lr_arr"], g["lr_tr"], g["lr_crs"], b, g["lr_px"], g["dst_crs"], Resampling.average, margin=m)
    lv = tc._crop_reproject(g["lrv_raw"], g["lrv_tr"], g["lrv_crs"], b, g["lr_px"], g["dst_crs"], Resampling.nearest, margin=m) == 1
    lvb = lv & np.isfinite(lr)
    d_mask = np.argwhere(lvb != lv_a); d_nan = np.argwhere(np.isnan(lr) != np.isnan(lr_a)); d_lv = np.argwhere(lv != (tc._crop_reproject(g["lrv_raw"], g["lrv_tr"], g["lrv_crs"], b, g["lr_px"], g["dst_crs"], Resampling.nearest, margin=10**6) == 1))
    fin = lvb & lv_a
    print(f"margin {m}: mask diffs {len(d_mask)} at {d_mask.tolist()[:6]}; lr NaN-pattern diffs {len(d_nan)}; lrv(nearest) diffs vs full {len(d_lv)}; max|lr diff| on joint {np.abs(lr[fin]-lr_a[fin]).max() if fin.any() else None}")
    for (i, j) in d_mask.tolist()[:4]:
        print(f"    cell ({i},{j}): full lr={lr_a[i,j]} lv={lv_a[i,j]} | crop lr={lr[i,j]} lv_nearest={lv[i,j]}")
# Also: same tile through full path but with the source array made contiguous copy (is the full path itself deterministic?)
lr_c, lv_c = ecp.tile_ship_side(g, r0, c0, 256); print("full path repeatable:", np.array_equal(np.nan_to_num(lr_a, nan=-1e30), np.nan_to_num(lr_c, nan=-1e30)) and np.array_equal(lv_a, lv_c))
