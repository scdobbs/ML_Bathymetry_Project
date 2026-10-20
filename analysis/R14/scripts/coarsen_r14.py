"""R14 §3 — the coarsened ship grids. For every C1-set pair and every level c in LEVELS with lr_target = 256 x hr_res / c > lr_native:
area-weighted block averaging of the pair's native harmonized lr.tif onto a grid of cell lr_target aligned to the native grid's
origin, in the pair's CRS (exact overlap weights from separable 1-D overlap matrices; non-integer ratios allowed). A native cell
is valid when lr.tif is finite there and lr_valid.tif (HR grid; nearest onto the native LR grid, as features_v1.pair_context does)
is 1. A coarse cell is valid when >= 50 % of its area is covered by valid native cells (§3.2); its value is the valid-area-weighted
mean. The validity mask is written on the coarse grid (uint8 1 / 0, nodata 255). Where the pair's ship products are usable, the five
contract-v2.1 products are block averaged the same way (validity = finite) onto the same grid under <cdir>/ship_products_v2_1/, so
that the frozen features_v1.pair_context assertion (products grid == lr.tif grid) holds. A harness manifest row per pair and level
(ONLY lr_native_res_m, harmonized_path_lr, lr_valid changed) is written as <COARSE>/harness_<tag>.parquet. Every raster's sha256 is
recorded in <COARSE>/SHA256SUMS_coarse.txt and coarse_record.json. Caveat (§3.4, stated, not corrected): block averaging removes
resolution but not sonar noise, footprint shape or sounding density."""
import json, os, sys, time
import numpy as np, pandas as pd, rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_common as R
C4, R8 = R.C4, R.R8
if R.REPO not in sys.path: sys.path.insert(0, R.REPO)
from preprocessing.colocated_utils import resolve_pair_paths, load_band
import features_v1 as F

def overlap_matrix(n_native, s):
    """Sparse (n_coarse x n_native) matrix of overlap lengths (native-pixel units) between coarse cell J = [J s, (J+1) s) and native cell i = [i, i+1)."""
    n_coarse = int(np.ceil(n_native / s)); rows, cols, vals = [], [], []
    for J in range(n_coarse):
        lo, hi = J * s, (J + 1) * s; i0, i1 = int(np.floor(lo)), min(n_native, int(np.ceil(hi)))
        i = np.arange(i0, i1); w = np.minimum(hi, i + 1) - np.maximum(lo, i); keep = w > 1e-12
        rows.append(np.full(keep.sum(), J)); cols.append(i[keep]); vals.append(w[keep])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_coarse, n_native))

def block_average(arr, valid, s):
    """Area-weighted mean of arr over valid native cells per coarse cell; (mean, coverage fraction of the coarse cell's area, valid-area)."""
    H, W = arr.shape; Wy = overlap_matrix(H, s); Wx = overlap_matrix(W, s); M = valid.astype(np.float64); A = np.where(valid, arr, 0.0).astype(np.float64)
    num = Wy @ A @ Wx.T; den = Wy @ M @ Wx.T; cov = den / (s * s)
    with np.errstate(invalid="ignore", divide="ignore"): mean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return mean, cov

def write_tif(path, arr, transform, crs, dtype, nodata):
    prof = dict(driver="GTiff", dtype=dtype, count=1, height=arr.shape[0], width=arr.shape[1], crs=crs, transform=transform, nodata=nodata, compress="deflate", tiled=False)
    with rasterio.open(path, "w", **prof) as d: d.write(arr.astype(dtype), 1)
    return R.sha256_file(path)

def native_lr(row):
    paths = resolve_pair_paths(C4.DATA, row)
    with rasterio.open(paths["lr"]) as s: lr, lr_tr, lr_crs = load_band(s); lr_nodata = s.nodata
    with rasterio.open(paths["lr_valid"]) as s: lrv_raw = s.read(1).astype(np.float32); lrv_tr, lrv_crs = s.transform, s.crs
    lv = np.zeros(lr.shape, np.float32)
    reproject(source=lrv_raw, destination=lv, src_transform=lrv_tr, src_crs=lrv_crs, dst_transform=lr_tr, dst_crs=lr_crs, resampling=Resampling.nearest, src_nodata=None, dst_nodata=0)   # as features_v1.pair_context
    valid = (lv == 1) & np.isfinite(lr); assert abs(abs(lr_tr.a) - abs(lr_tr.e)) < 1e-9, lr_tr
    return dict(lr=lr, tr=lr_tr, crs=lr_crs, nodata=lr_nodata, valid=valid, posting=float(abs(lr_tr.a)), paths=paths)

def coarsen_pair(m, pid, tiles, out):
    row = C4.pair_row(m, pid); g = R.pair_geometry(tiles, pid); plan, skipped = R.level_plan(g["hr_res"], g["lr_native"]); n = native_lr(row); usable = C4.ship_usable(row)
    rec = dict(pair=pid, unit=C4.UNIT_OF[pid], **g, posting_native=n["posting"], lr_nodata=n["nodata"], native_shape=list(n["lr"].shape), n_native_valid=int(n["valid"].sum()), n_native_finite=int(np.isfinite(n["lr"]).sum()),
               ship_products_usable=usable, products_dir=(C4.products_dir(row) if usable else None), levels={}, skipped={str(c): v for c, v in skipped.items()})
    prods = {}
    if usable:
        for p in F.PRODUCTS:
            with rasterio.open(os.path.join(C4.products_dir(row), f"{p}.tif")) as s:
                a = s.read(1).astype(np.float32); nod = s.nodata; assert s.shape == n["lr"].shape and s.transform == n["tr"], (pid, p)
                prods[p] = dict(arr=a, nodata=nod, n_nonfinite=int((~np.isfinite(a)).sum()), n_eq_nodata=(int((a == nod).sum()) if nod is not None and np.isfinite(nod) else 0))
    for c, lt in plan.items():
        t0 = time.time(); s = lt / n["posting"]; cdir = R.coarse_dir(pid, c); os.makedirs(cdir, exist_ok=True); tr_c = n["tr"] * Affine.scale(s)
        mean, cov = block_average(n["lr"], n["valid"], s); ok = cov >= R.COVER_MIN; lr_c = np.where(ok, mean, np.nan).astype(np.float32)
        hashes = {}
        hashes["lr.tif"] = write_tif(f"{cdir}/lr.tif", np.where(ok, lr_c, -9999.0), tr_c, n["crs"], "float32", -9999.0)
        hashes["lr_valid.tif"] = write_tif(f"{cdir}/lr_valid.tif", ok.astype(np.uint8), tr_c, n["crs"], "uint8", 255)
        if usable:
            pdir = f"{cdir}/ship_products_v2_1"; os.makedirs(pdir, exist_ok=True)
            for p, d in prods.items():
                v = np.isfinite(d["arr"]) & ((d["arr"] != d["nodata"]) if (d["nodata"] is not None and np.isfinite(d["nodata"])) else True)
                pm, pc = block_average(d["arr"], v, s); pa = np.where(pc >= R.COVER_MIN, pm, np.nan).astype(np.float32)
                hashes[f"ship_products_v2_1/{p}.tif"] = write_tif(f"{pdir}/{p}.tif", pa, tr_c, n["crs"], "float32", np.nan)
        assert abs(abs(tr_c.a) - lt) < 1e-9 and tr_c.c == n["tr"].c and tr_c.f == n["tr"].f
        lrec = dict(level=c, lr_target=lt, ratio_s=s, lr_px_frozen_path=max(2, int(round(R.WIN * g["hr_res"] / lt))), coarse_shape=list(lr_c.shape), n_coarse_valid=int(ok.sum()), n_coarse_any=int((cov > 0).sum()), coverage_median_valid=float(np.median(cov[ok])) if ok.any() else np.nan,
                    cells_across=R.WIN * g["hr_res"] / lt, transform=list(tr_c)[:6], crs=str(n["crs"]), files=hashes, products=("coarsened" if usable else "masked (not usable at native resolution)"), seconds=time.time() - t0)
        R.dump(lrec, f"{cdir}/record.json"); rec["levels"][str(c)] = lrec
        print(f"  {pid:40s} c={c:4} lr_target {lt:7.2f} m (s={s:6.3f}) -> {lr_c.shape[0]}x{lr_c.shape[1]} cells, {int(ok.sum())} valid ({time.time()-t0:.1f} s)", flush=True)
    return rec

if __name__ == "__main__":
    t0 = time.time(); m, rec0 = R.start("coarsen_r14.py"); os.makedirs(R.COARSE, exist_ok=True); tiles = R8.load_tiles(); SI, PP, cmap, stats = R.load_r10b(); pairs = R.c1_pairs(PP); assert len(pairs) == 14, pairs
    recs = {pid: coarsen_pair(m, pid, tiles, R.COARSE) for pid in pairs}
    # harness rows per level (v2.1 columns; only the three columns changed)
    harness = {}
    for c in R.LEVELS:
        rows = []
        for pid in pairs:
            if str(c) not in recs[pid]["levels"]: continue
            row = R.harness_row(m, pid, recs[pid]["levels"][str(c)]["lr_target"], R.coarse_dir(pid, c)); ch = R.harness_changes(m, pid, row); assert sorted(ch) == ["harmonized_path_lr", "lr_native_res_m", "lr_valid"], (pid, c, ch); rows.append(row)
        if rows:
            H = pd.DataFrame(rows).reset_index(drop=True); H = H[list(m.columns)]; p = f"{R.COARSE}/harness_{R.tag(c)}.parquet"; H.to_parquet(p, index=False); harness[str(c)] = dict(path=p, sha256=R.sha256_file(p), pairs=list(H.pair_id))
    lines = []
    for pid, r in recs.items():
        for c, l in r["levels"].items():
            for f, h in l["files"].items(): lines.append(f"{h}  {pid}/{R.tag(float(c))}/{f}")
    open(f"{R.COARSE}/SHA256SUMS_coarse.txt", "w").write("\n".join(sorted(lines, key=lambda s: s.split('  ', 1)[1])) + "\n")
    R.dump(dict(pairs=recs, harness=harness, levels=list(R.LEVELS), cover_min=R.COVER_MIN, method="area-weighted block average onto a grid of cell lr_target aligned to the native origin; coarse cell valid iff >= 50 % of its area is covered by valid native cells (finite lr.tif & lr_valid.tif nearest on the LR grid); products block averaged the same way (validity = finite)",
                caveat="block averaging a modern grid removes resolution but not sonar noise, footprint shape or sounding density (directive §3.4)", n_rasters=len(lines), node=os.uname().nodename, seconds=time.time() - t0), f"{R.COARSE}/coarse_record.json")
    print(f"done: {len(lines)} rasters, harness rows for {len(harness)} levels ({time.time()-t0:.0f} s)", flush=True)
