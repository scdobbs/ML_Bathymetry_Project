"""R03 §4.1–4.3 — ship-only feature bank per tile at four context sizes, plus the spectral-extrapolation predictor ŝ_fine.

Context windows are cut from the harmonized LR raster (ship grid at native GSD) centred on the tile:
  a: tile extent (256 HR px);  b: tile + 48 HR px rim (352 px);  c: 32x32 ship cells;  d: 64x64 ship cells.
R02 roughness/metadata features are defined in cycles per SHIP CELL (R02 computed them on the bicubic HR-grid field,
i.e. at upsample factor = lr_native/hr_res); here every context uses the bicubic interpolant of the ship window at a
fixed factor 8 with the same bands in cyc/cell, so contexts are comparable and (d) stays tractable. Context (a) is also
computed the R02 way (from the clean patch's own bicubic on the HR grid) as `*_a_hrgrid` for a like-for-like check.
Spectral extrapolation (§4.2) works on the native-resolution ship window: plane removed, Hann taper (R02 had no taper;
its band-RMS features remain taper-free), radial PSD, log-log fit over [f_min, 0.8 f_Nyq], beta clamped to [1, 5],
A f^-beta integrated over the fine annulus [f_lr, f_hi] = [1, k] cyc/cell -> predicted fine variance -> ŝ_fine (m).
No HR quantity enters any feature (leak test in leak_test_features)."""
from __future__ import annotations
import sys, os, glob, json, math, hashlib, numpy as np, pandas as pd, rasterio
from scipy.ndimage import zoom, gaussian_filter, distance_transform_edt
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C
sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project"); from preprocessing import extract_clean_patches as ecp
sys.path.insert(0, "/scratch/users/scdobbs/step0_bakeoff"); import step0_bakeoff as b0
UP = 8; CONTEXTS = ("a", "b", "c", "d"); EPS = 1e-9

def hann2(h, w):
    return np.outer(np.hanning(h), np.hanning(w))

def radial_psd_cells(win, m):
    """Plane-removed, mean-filled, Hann-tapered native ship window -> (f cyc/cell, mean power per coefficient per bin,
    n coefficients per bin, 2-D power grid, fx, fy). Parseval: sum(p) = var of tapered field."""
    H, W = win.shape; yy, xx = np.mgrid[:H, :W]; A = np.c_[xx[m], yy[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, win[m], rcond=None); r = np.where(m, win - (coef[0] * xx + coef[1] * yy + coef[2]), 0.0)
    t = hann2(H, W); r = r * t / np.sqrt(np.mean(t ** 2))            # taper, power-normalized
    F = np.fft.fft2(r) / (H * W); P = np.abs(F) ** 2
    fy = np.fft.fftfreq(H)[:, None] * np.ones((1, W)); fx = np.fft.fftfreq(W)[None, :] * np.ones((H, 1)); fr = np.sqrt(fx ** 2 + fy ** 2)
    nb = max(4, min(H, W) // 2); edges = np.linspace(0, 0.5, nb + 1); idx = np.digitize(fr.ravel(), edges) - 1; ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb); pw = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=nb) / np.maximum(cnt, 1)
    return 0.5 * (edges[1:] + edges[:-1]), pw, cnt, P, fx, fy, fr

def spectral_extrap(win, m, k):
    """§4.2: fit log p = log a - beta log f over [f_min, 0.4] cyc/cell (0.8 x Nyquist 0.5), extrapolate to [1, k] cyc/cell."""
    f, pw, cnt, P, fx, fy, fr = radial_psd_cells(win, m); Wn = win.shape[0] * win.shape[1]
    sel = (cnt >= 8) & (f > 0) & (f <= 0.4) & (pw > 0)
    if sel.sum() < 3: return dict(beta=np.nan, logA=np.nan, fit_r2=np.nan, beta_clamped=np.nan, shat_fine=np.nan, noise_excess=np.nan, aniso_spec=np.nan, n_fit_bins=int(sel.sum()))
    lf, lp = np.log10(f[sel]), np.log10(pw[sel]); b, a = np.polyfit(lf, lp, 1); beta = -b; pred = a + b * lf
    r2 = 1 - np.sum((lp - pred) ** 2) / max(np.sum((lp - lp.mean()) ** 2), EPS); bc = float(np.clip(beta, 1.0, 5.0))
    # density per unit frequency area: p(f) is power per coefficient; coefficient spacing is 1/W x 1/H -> density = p * H*W
    A = 10 ** a; f1, f2 = 1.0, float(k)
    var = 2 * math.pi * A * Wn * ((f2 ** (2 - bc) - f1 ** (2 - bc)) / (2 - bc) if abs(bc - 2) > 1e-6 else math.log(f2 / f1))
    # noise indicator: measured power in the top 20% of ship frequencies [0.4, 0.5] relative to the fitted law there
    top = (cnt >= 4) & (f >= 0.4) & (f <= 0.5)
    noise = float(np.log10(np.mean(pw[top]) / np.mean(10 ** (a + b * np.log10(f[top]))))) if top.sum() else np.nan
    ann = (fr >= 0.05) & (fr <= 0.4)
    if ann.sum() >= 8:
        w = P[ann]; sw = w.sum() + 1e-30; cxx = np.sum(w * fx[ann] ** 2) / sw; cyy = np.sum(w * fy[ann] ** 2) / sw; cxy = np.sum(w * fx[ann] * fy[ann]) / sw
        ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]]); aniso = float(np.sqrt(ev.max() / ev.min())) if ev.min() > 0 else np.nan
    else: aniso = np.nan
    return dict(beta=float(beta), logA=float(a), fit_r2=float(r2), beta_clamped=bc, shat_fine=float(np.sqrt(max(var, 0.0))), noise_excess=noise, aniso_spec=aniso, n_fit_bins=int(sel.sum()))

def r02_features_on_field(field, m, f_lr):
    """R02 roughness-3 + slope on a bicubic field with ship sampling frequency f_lr (cyc/px of that field)."""
    Cc, curv, aniso = b0.coarse_features(field, m, f_lr)
    sig = max(1.0 / f_lr / 2.0, 1.0); sm = gaussian_filter(np.where(np.isfinite(field), field, np.nanmean(field)).astype(float), sigma=sig); gy, gx = np.gradient(sm)
    return dict(C=float(Cc), curv=float(curv), aniso=float(aniso), slope=float(np.mean(np.sqrt(gx ** 2 + gy ** 2)[m])))

def ship_window(g, r0, c0, win_px, half_cells=None, extra_px=0):
    """Ship-grid window centred on the HR tile (r0, c0, win_px). half_cells overrides the size (contexts c, d)."""
    cy, cx = r0 + win_px / 2.0, c0 + win_px / 2.0; x, y = g["transform"] * (cx, cy); lr_r, lr_c = rasterio.transform.rowcol(g["lr_tr"], x, y)
    cell_m = abs(g["lr_tr"].a); ratio = cell_m / g["hr_res"]
    hc = half_cells if half_cells is not None else (win_px + 2 * extra_px) / ratio / 2.0
    n = max(4, int(math.ceil(2 * hc))); r1 = int(round(lr_r - n / 2)); c1 = int(round(lr_c - n / 2)); H, W = g["lr_arr"].shape
    rs, re_, cs, ce = max(0, r1), min(H, r1 + n), max(0, c1), min(W, c1 + n)
    if re_ - rs < 4 or ce - cs < 4: return None, None, 0.0, n
    lr = g["lr_arr"][rs:re_, cs:ce]; lv = g["lrv_native"][rs:re_, cs:ce] & np.isfinite(lr)
    return lr, lv, float(lv.mean() * (re_ - rs) * (ce - cs) / (n * n)), n

def features_for_window(lr, lv, k, cell_m):
    """All ship-only features of one native ship window (meters). Returns dict."""
    if lr is None or lv.sum() < 0.25 * lv.size or lv.sum() < 16: return None
    fill = float(lr[lv].mean()); w = np.where(lv, lr, fill).astype(float)
    out = spectral_extrap(w, lv, k)
    yy, xx = np.mgrid[:w.shape[0], :w.shape[1]]; A = np.c_[xx[lv], yy[lv], np.ones(lv.sum())]; coef, *_ = np.linalg.lstsq(A, w[lv], rcond=None)
    det = w - (coef[0] * xx + coef[1] * yy + coef[2]); out["std_ship"] = float(det[lv].std())
    gy, gx = np.gradient(w, cell_m); sl = np.sqrt(gx ** 2 + gy ** 2)[lv]; out["slope_ship_mean"] = float(sl.mean()); out["slope_ship_p90"] = float(np.percentile(sl, 90))
    lap = (np.roll(w, 1, 0) + np.roll(w, -1, 0) + np.roll(w, 1, 1) + np.roll(w, -1, 1) - 4 * w) / cell_m ** 2; out["curv_ship_std"] = float(lap[lv].std())
    up = zoom(w, UP, order=3); m_up = zoom(lv.astype(float), UP, order=0) > 0.5; f_lr = 1.0 / UP     # ship sampling frequency on the x8 grid
    if m_up.sum() >= 64: out.update(r02_features_on_field(up, m_up, f_lr))
    else: out.update(C=np.nan, curv=np.nan, aniso=np.nan, slope=np.nan)
    return out

def slim_global(g):
    from rasterio.warp import reproject; from rasterio.enums import Resampling as RS
    lv = np.zeros(g["lr_arr"].shape, np.float32)
    reproject(source=g["lrv_raw"], destination=lv, src_transform=g["lrv_tr"], src_crs=g["lrv_crs"], dst_transform=g["lr_tr"], dst_crs=g["lr_crs"], resampling=RS.nearest, src_nodata=None, dst_nodata=0)
    return dict(lrv_native=(lv == 1) & np.isfinite(g["lr_arr"]))

def leak_test_features(pairs_df, tiles, n=100, seed=0):
    rng = np.random.default_rng(seed); hb = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True, default=float).encode()).hexdigest(); res = []
    for pid, sub in tiles.sample(n=min(n, len(tiles)), random_state=seed).groupby("pair_id"):
        row = pairs_df[pairs_df.pair_id == pid].iloc[0]; g = ecp.pair_setup(C.DATA, row, 256); g.update(slim_global(g))
        g2 = dict(g); g2["hr_arr"] = np.where(np.isfinite(g["hr_arr"]), rng.standard_normal(g["hr_arr"].shape).astype(np.float32) * 100 - 2000, np.nan)
        for _, t in sub.iterrows():
            fa = tile_features(g, int(t.r0), int(t.c0), int(t.k)); fb = tile_features(g2, int(t.r0), int(t.c0), int(t.k))
            res.append(dict(pair_id=pid, r0=int(t.r0), c0=int(t.c0), identical=hb(fa) == hb(fb)))
    return res

def tile_features(g, r0, c0, k):
    cell_m = abs(g["lr_tr"].a); out = {}
    for ctx, kw in (("a", dict(extra_px=0)), ("b", dict(extra_px=48)), ("c", dict(half_cells=16)), ("d", dict(half_cells=32))):
        lr, lv, vf, n = ship_window(g, r0, c0, 256, **kw); f = features_for_window(lr, lv, k, cell_m)
        out[f"ctx_{ctx}_cells"] = n; out[f"ctx_{ctx}_valid_frac"] = vf
        for kk, v in (f or {}).items(): out[f"{kk}_{ctx}"] = v
    return out

if __name__ == "__main__":
    import argparse; from multiprocessing import Pool
    from preprocessing.colocated_utils import load_manifest
    ap = argparse.ArgumentParser(); ap.add_argument("--patches", nargs="+", required=True); ap.add_argument("--units", default="all"); ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--out", required=True); ap.add_argument("--leak", type=int, default=100); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args(); pairs_df = load_manifest(C.DATA)
    units = {**C.NATIVE7, **C.EW0207, **C.CALDIG, **C.EXPANSION7} if args.units == "all" else {u: ({**C.NATIVE7, **C.EW0207, **C.CALDIG, **C.EXPANSION7})[u] for u in args.units.split(",")}
    files = []
    for u, pids in units.items():
        for pid in C.assert_no_lockbox(pids):
            fs = sorted(f for P in args.patches for f in glob.glob(f"{P}/*/{pid}_r*.npz"))
            if args.cap and len(fs) > args.cap: fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, args.cap).round().astype(int))]   # R02 files_for subsample rule
            files += [(u, pid, f) for f in fs]
    print(f"{len(files)} tiles over {len(units)} units", flush=True)
    G = {}
    def slim(g):
        """Ship-side arrays + geometry only; HR arrays dropped (workers never need them; keeps memory per pair ~ LR size).
        lr_valid.tif does not share lr.tif's grid, so ship validity is reprojected (nearest) onto the LR grid once per pair."""
        from rasterio.warp import reproject; from rasterio.enums import Resampling as RS
        lv = np.zeros(g["lr_arr"].shape, np.float32)
        reproject(source=g["lrv_raw"], destination=lv, src_transform=g["lrv_tr"], src_crs=g["lrv_crs"], dst_transform=g["lr_tr"], dst_crs=g["lr_crs"], resampling=RS.nearest, src_nodata=None, dst_nodata=0)
        g["lrv_native"] = (lv == 1) & np.isfinite(g["lr_arr"])
        return {k: g[k] for k in ("transform", "lr_tr", "lr_crs", "dst_crs", "lr_arr", "lrv_native", "hr_res", "lr_px", "scale_vec", "H", "W")}
    def work(job):
        u, pid, f = job; g = G["g"]
        d = np.load(f, allow_pickle=True); st = np.asarray(d["stats"]).ravel(); base = os.path.basename(f); r0 = int(base.split("_r")[-1][:6]); c0 = int(base.split("_c")[-1][:6])
        k = int(st[5]); out = dict(unit=u, pair_id=pid, key=base, r0=r0, c0=c0, k=k, mu_lr=float(st[0]), s_lr=float(st[1]), delta=float(st[2]), lr_native=float(st[6]), hr_res=float(st[7]), s_floor_hit=bool(st[8] > 0.5))
        out.update(tile_features(g, r0, c0, k))
        # context (a) the R02 way: on the clean patch's own bicubic (HR grid), + targets from the patch (label side, NOT features)
        x = d["x"][0]; y = d["y"][0]; hv = d["hr_valid"][0].astype(bool); lvh = d["lr_valid_hr"][0].astype(bool); j = hv & lvh
        lu = x * out["s_lr"] + out["mu_lr"]; hr_off = y * out["s_lr"] + out["mu_lr"]; f_lr = out["hr_res"] / out["lr_native"]; f_hi = min(k * f_lr, 0.45)
        fa = r02_features_on_field(lu, j, f_lr); out.update({f"{kk}_a_hrgrid": v for kk, v in fa.items()})
        out["dist_edge"] = float(np.mean(distance_transform_edt(lvh)[j])) if lvh.any() else 0.0; out["frac_valid"] = float(lvh.mean()); out["depth"] = out["mu_lr"]
        resid = np.where(j, hr_off - lu, 0.0); T2 = float(np.mean(resid[j] ** 2)); tf = b0.band_rms(resid, j, f_lr, f_hi); th = b0.band_rms(resid, j, f_hi, 1.0)
        out.update(T_ship=float(np.sqrt(T2)), T_fine=tf, T_hi=th, T_res=float(np.sqrt(max(T2 - tf ** 2 - th ** 2, 0.0))), A_fine=b0.band_rms(hr_off, j, f_lr, f_hi), B_fine=b0.band_rms(lu, j, f_lr, f_hi), frac_joint=float(j.mean()), f_lr=f_lr, f_hi=f_hi)
        return out
    # one pair at a time in the parent (slim ship-side arrays), workers forked per pair: no per-worker raster accumulation (job 45936095 OOM)
    rows = []; import time
    for pid in sorted({t[1] for t in files}):
        t0 = time.time(); row = pairs_df[pairs_df.pair_id == pid].iloc[0]; g = ecp.pair_setup(C.DATA, row, 256); g["lrv_native"] = g["lrv_raw"] == 1; G["g"] = slim(g); del g
        jobs_p = [t for t in files if t[1] == pid]
        with Pool(args.workers) as pool: rows += pool.map(work, jobs_p, chunksize=4)
        print(f"  {pid}: {len(jobs_p)} tiles in {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); os.makedirs(os.path.dirname(args.out), exist_ok=True); df.to_parquet(args.out); print("wrote", args.out, df.shape, flush=True)
    if args.leak:
        tl = df[["pair_id", "r0", "c0", "k"]]; res = leak_test_features(pairs_df, tl, n=args.leak); n_ok = sum(r["identical"] for r in res)
        json.dump(dict(n=len(res), n_identical=n_ok, tiles=res), open(args.out.replace(".parquet", "_leak.json"), "w"), indent=1); print(f"FEATURE LEAK TEST: {n_ok}/{len(res)} identical with HR replaced by noise", flush=True)
