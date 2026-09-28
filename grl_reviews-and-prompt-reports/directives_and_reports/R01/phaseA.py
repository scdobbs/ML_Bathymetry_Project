"""R01 Phase A — data integrity over every restored patch and the native source grids. No model."""
from __future__ import annotations
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

R = "/scratch/users/scdobbs/grl_review/R01/patches_restored/patches_native_phase2"
OUT = "/scratch/users/scdobbs/grl_review/R01/phaseA"; os.makedirs(OUT, exist_ok=True)
OAK = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy"
NATIVE = ['NA090__MGDS_31212','NA080__MGDS_31290','TN399__MGDS_30373','RR1506__MGDS_29779','AT37-13__MGDS_31199',
          'EW0207__MGDS_32556','FK181031__MGDS_24618','TN299__MGDS_27339','TN299__MGDS_31253']
FILLS = [-9999.0, -32768.0, -32767.0, 3.4028235e38]
RUN = 16

def max_run(a):
    """longest run of identical consecutive values along rows (any row), for 2-D array."""
    best = 0
    for row in a:
        if len(row) < 2: continue
        ch = np.r_[True, row[1:] != row[:-1]]
        idx = np.flatnonzero(ch); runs = np.diff(np.r_[idx, len(row)])
        best = max(best, int(runs.max()))
    return best

def zero_run_count(a, eps=0.0):
    """number of pixels lying in row-wise runs of exact zeros of length >= RUN."""
    n = 0
    for row in a:
        z = (row == eps)
        if not z.any(): continue
        ch = np.r_[True, z[1:] != z[:-1]]; idx = np.flatnonzero(ch); runs = np.diff(np.r_[idx, len(row)])
        for s, L in zip(idx, runs):
            if z[s] and L >= RUN: n += int(L)
    return n

def counts(arr, valid, raw):
    """A1 counts split by valid/invalid. arr = normalized array, raw = reconstructed meters."""
    out = {}
    for tag, m in (("valid", valid), ("invalid", ~valid)):
        a = arr[m]; r = raw[m]
        out[f"nan_{tag}"] = int(np.isnan(a).sum()); out[f"inf_{tag}"] = int(np.isinf(a).sum())
        out[f"le_m9000_{tag}"] = int((r <= -9000).sum())
        out[f"fill_{tag}"] = int(sum(np.isclose(r, f, rtol=0, atol=1e-3).sum() for f in FILLS))
    out["zero_run_px"] = zero_run_count(np.where(valid, arr, np.nan))      # zeros inside VALID pixels only
    out["zero_run_px_invalid"] = zero_run_count(np.where(valid, np.nan, arr))
    out["max_const_run_valid"] = max_run(np.where(valid, arr, np.nan)) if valid.any() else 0
    return out

def one(f):
    try:
        d = np.load(f, allow_pickle=True)
        hr = d['hr'][0].astype(np.float64); lu = d['lr_up'][0].astype(np.float64); lr = d['lr_raw'][0].astype(np.float64)
        hv = d['hr_valid'][0].astype(bool); lv = d['lr_valid'][0].astype(bool)
        st = np.asarray(d['stats']).ravel().astype(float); mean, std = st[0], st[1]
        H = hr.shape[0]; L = lr.shape[0]; fac = H // L
        lv_hr = np.kron(lv, np.ones((fac, fac), bool))[:H, :H] if fac >= 1 else np.ones_like(hv)
        if lv_hr.shape != hv.shape:  # non-integer factor: nearest resample
            yi = (np.arange(H) * L / H).astype(int); lv_hr = lv[yi][:, yi]
        jv = hv & lv_hr
        rec = dict(file=os.path.basename(f), pid=str(d['pair_id']), unit=str(d['leakage_unit']), split=f.split('/')[-2],
                   mean_m=mean, std_m=std, k=st[2], lr_native=st[4], hr_res=st[5], std_clamped=bool(std == 1.0 and np.nanstd(hr[hv]) < 0.5) if hv.any() else False)
        # A1
        for name, arr, valid, raw in (("hr", hr, hv, hr*std+mean), ("lr_raw", lr, lv, lr*std+mean), ("lr_up", lu, lv_hr, lu*std+mean)):
            for k, v in counts(arr, valid, raw).items(): rec[f"{name}_{k}"] = v
        for name, arr in (("hr_valid", hv), ("lr_valid", lv)):
            rec[f"{name}_nonbinary"] = int(((d[name][0] != 0) & (d[name][0] != 1)).sum())
        # A2
        rec.update(frac_hr_valid=float(hv.mean()), frac_lr_valid=float(lv.mean()), frac_joint_valid=float(jv.mean()),
                   hr_valid_nonfinite=int((~np.isfinite(hr[hv])).sum()), lr_valid_nonfinite=int((~np.isfinite(lr[lv])).sum()),
                   hr_invalid_not_fill=int((np.abs(hr[~hv]) > 1e-6).sum()),          # invalid HR should hold mean-fill (=0 normalized)
                   lr_invalid_not_fill=int((np.abs(lr[~lv]) > 1e-6).sum()),
                   invalid_touches_border=bool((~jv[0, :]).any() or (~jv[-1, :]).any() or (~jv[:, 0]).any() or (~jv[:, -1]).any()),
                   has_invalid=bool((~jv).any()))
        # A3 consistency of normalization
        if hv.sum() > 1:
            rec.update(hr_valid_mean_norm=float(hr[hv].mean()), hr_valid_std_norm=float(hr[hv].std()))
        # A4
        if lv.any():
            rec.update(lr_std_m=float(lr[lv].std() * std))
        rec.update(max_abs_lr_up=float(np.abs(lu[jv]).max()) if jv.any() else np.nan, max_abs_hr=float(np.abs(hr[hv]).max()) if hv.any() else np.nan,
                   max_abs_lr_up_all=float(np.abs(lu).max()))
        # A5 vertical offset
        if jv.sum() > 16:
            diff = (hr - lu)[jv]
            rec.update(offset_norm=float(np.median(diff)), offset_m=float(np.median(diff) * std), offset_mean_norm=float(diff.mean()))
        # A6 sign + coarse agreement (block-average HR to LR cells vs lr_raw on lr_valid cells)
        rec.update(hr_raw_mean=float(np.nanmean(hr[hv]) * std + mean) if hv.any() else np.nan,
                   lr_raw_mean=float(lr[lv].mean() * std + mean) if lv.any() else np.nan)
        # block-average HR onto the LR cell grid by nearest index (handles non-integer H/L)
        ci = (np.arange(H) * L // H)
        cell = (ci[:, None] * L + ci[None, :])
        num = np.bincount(cell[hv], weights=hr[hv], minlength=L * L); den = np.bincount(cell[hv], minlength=L * L)
        exp_px = (H / L) ** 2
        ok = (den > 0.5 * exp_px) & lv.ravel()
        if ok.sum() >= 8:
            hb_mean = num[ok] / den[ok]; lb = lr.ravel()[ok]
            rec["coarse_corr"] = float(np.corrcoef(hb_mean, lb)[0, 1]) if hb_mean.std() > 0 and lb.std() > 0 else np.nan
            rec["coarse_n"] = int(ok.sum())
        rec["error"] = ""
        return rec
    except Exception as e:
        return dict(file=os.path.basename(f), error=repr(e))

def source_grids():
    import rasterio
    rows = []
    for pid in NATIVE:
        base = f"{OAK}/harmonized/{pid}"
        for name in ("hr", "lr"):
            p = f"{base}/{name}.tif"
            with rasterio.open(p) as s:
                nod = s.nodata; a_un = s.read(1, masked=False).astype(np.float64); a_ma = s.read(1, masked=True)
                mask_valid = ~np.ma.getmaskarray(a_ma)
            vfile = f"{base}/{name}_valid.tif"
            with rasterio.open(vfile) as s: vv = s.read(1); vnod = s.nodata
            with rasterio.open(f"{base}/joint_valid.tif") as s: jvv = s.read(1)
            # shipped validity mask may be at HR grid for lr_valid; compare where shapes match
            same = vv.shape == a_un.shape
            v = (vv == 1) if same else None
            rec = dict(pid=pid, grid=name, path=p, nodata_meta=nod, dtype=str(a_un.dtype), shape=a_un.shape,
                       n_nan_unmasked=int(np.isnan(a_un).sum()), n_inf=int(np.isinf(a_un).sum()),
                       n_le_m9000=int((a_un <= -9000).sum()), n_fill_exact=int(sum(np.isclose(a_un, f, rtol=0, atol=1e-3).sum() for f in FILLS)),
                       n_masked_by_rasterio=int((~mask_valid).sum()), n_pixels=int(a_un.size),
                       valid_mask_shape=vv.shape, valid_mask_values=sorted(np.unique(vv).tolist())[:6], valid_mask_nodata=vnod,
                       joint_valid_frac=float((jvv == 1).mean()))
            if v is not None:
                fin = np.isfinite(a_un) & (a_un > -9000) & ~np.isclose(a_un, -9999, atol=1e-3)
                rec.update(n_valid_mask=int(v.sum()), valid_but_nonfinite_or_sentinel=int((v & ~fin).sum()),
                           invalid_but_finite=int((~v & fin).sum()),
                           depth_min_valid=float(a_un[v & fin].min()) if (v & fin).any() else np.nan,
                           depth_max_valid=float(a_un[v & fin].max()) if (v & fin).any() else np.nan,
                           depth_median_valid=float(np.median(a_un[v & fin])) if (v & fin).any() else np.nan,
                           frac_positive_valid=float((a_un[v & fin] > 0).mean()) if (v & fin).any() else np.nan)
            rows.append(rec)
            print(f"  source {pid} {name}: nodata={nod} nan={rec['n_nan_unmasked']} sentinel={rec['n_fill_exact']} valid_but_bad={rec.get('valid_but_nonfinite_or_sentinel')}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/source_grids.csv", index=False)
    json.dump(rows, open(f"{OUT}/source_grids.json", "w"), indent=1, default=str)

if __name__ == "__main__":
    t0 = time.time()
    files = sorted(glob.glob(f"{R}/*/*.npz")); print(f"{len(files)} patch files", flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        recs = pool.map(one, files, chunksize=64)
    df = pd.DataFrame(recs); df.to_parquet(f"{OUT}/perpatch.parquet")
    print(f"patches done in {time.time()-t0:.0f}s; errors={int((df.error != '').sum())}", flush=True)
    source_grids()
    print("source grids done", flush=True)
