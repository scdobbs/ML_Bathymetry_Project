"""features_v1_1 — FROZEN ship-only feature bank (R04 §3.1 code + the R05 §2.3 consumer-side availability override; tag phase0-freeze).
Override (R05 ruling H2): OVERRIDES maps (pair_id, product) -> reason; the product is treated as unavailable for that pair and its
features are NaN. Everything else is byte-for-byte the R04 feature code.

Every feature is computed from ship-side data only: the harmonized LR raster (native ship grid), its validity mask,
the contract-v2.1 ship products (where the manifest says ship_products_usable) and manifest metadata. No HR-derived
quantity is read: the pair context exposed to the feature code is a ShipOnlyView that raises on any HR key, and the
leak test (leak_test) recomputes features with the HR grid replaced by noise and compares hashes.

Supports: a window of WINDOWS_PX = (128, 224) HR px centred on the block (32 px) or on the tile (128 px); columns are
suffixed _128 / _224. Per window:
  native ship window (cells = ceil(win_px / ratio), ratio = ship cell / HR px; min 2):
    n_cells, valid_frac, mu_lr, s_lr (= max(std of valid cells, 0.5 m)), std_ship (plane-detrended), slope_ship_mean,
    slope_ship_p90, curv_ship_std (R03 bank, verbatim);
    spectral extrapolation beta, logA, fit_r2, beta_clamped, shat_fine, noise_excess, aniso_spec, n_fit_bins (R03 §4.2, verbatim);
    C, curv, aniso, slope on the x8 bicubic of the native window (R03 context features, verbatim);
    contract v2.1: rsd_med, rsd_p90, sd_med, logcount_med, xtrack_mean, beam_mean, rsd_over_sd_med (NaN where not usable);
  HR-grid window (R02 extract_clean_patches path: ship grid area-averaged onto lr_px cells, mu-filled, bicubic to win px):
    C_hr, curv_hr, aniso_hr, slope_hr (R02 roughness-3 + slope), dist_edge (mean distance to invalid LR px, ship mask only),
    frac_valid (LR validity fraction on the HR grid).
Metadata (driver): depth = mu_lr at 128, lr_native, k, hr_res, log_depth_over_lr, s_lr_256 (parent window, R02 definition).
Where a spectral quantity is undefined (too few cells) it is NaN; the GBT handles NaN natively."""
from __future__ import annotations
import hashlib, json, math, os
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import zoom, gaussian_filter, distance_transform_edt

WINDOWS_PX = (128, 224)
UP = 8; EPS = 1e-9; S_FLOOR = 0.5
HR_KEYS = ("hr_arr", "hrv_arr", "joint", "paths")
PRODUCTS = ("ship_count", "ship_sd", "ship_rsd", "ship_xtrack_frac", "ship_beam_angle")
OVERRIDES = {("EX1402L2__MGDS_31073", "ship_xtrack_frac"): "all-NaN where count >= 1 (R04 H2)"}   # R05 §2.3 consumer-side availability override
PRODUCT_OF_KEY = {"rsd_med": "ship_rsd", "rsd_p90": "ship_rsd", "sd_med": "ship_sd", "logcount_med": "ship_count", "xtrack_mean": "ship_xtrack_frac", "beam_mean": "ship_beam_angle", "rsd_over_sd_med": "ship_rsd"}

class ShipOnlyView(dict):
    """Pair context that raises if feature code touches any HR-derived key."""
    def __getitem__(self, k):
        if k in HR_KEYS: raise PermissionError(f"features_v1: HR-derived key '{k}' requested")
        return dict.__getitem__(self, k)
    def get(self, k, d=None):
        if k in HR_KEYS: raise PermissionError(f"features_v1: HR-derived key '{k}' requested")
        return dict.get(self, k, d)

# ---------------- R03 bank (features_context.py / step0_bakeoff.py, verbatim) ----------------
def _detrend(a, m):
    H, W = a.shape; yy, xx = np.mgrid[0:H, 0:W]
    A = np.c_[xx[m], yy[m], np.ones(m.sum())]; coef, *_ = np.linalg.lstsq(A, a[m], rcond=None)
    return np.where(m, a - (coef[0] * xx + coef[1] * yy + coef[2]), 0.0)

def _radial_freq(H, W):
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]; return np.sqrt(fy ** 2 + fx ** 2)

def bandpass(a, m, f_lo, f_hi):
    res = _detrend(np.asarray(a, float), m); F = np.fft.fft2(res); fr = _radial_freq(*a.shape)
    return np.real(np.fft.ifft2(F * ((fr >= f_lo) & (fr <= f_hi))))

def coarse_features(lr_up, m, f_lr):
    """R02 roughness-3: coarse-band RMS C, coarse curvature, coarse anisotropy on [f_lr/2, f_lr] of the bicubic field."""
    f_lo, f_hi = f_lr / 2.0, f_lr; cb = bandpass(lr_up, m, f_lo, f_hi)
    C = float(np.sqrt(np.mean(cb[m] ** 2))) if m.sum() else np.nan
    lap = (np.roll(cb, 1, 0) + np.roll(cb, -1, 0) + np.roll(cb, 1, 1) + np.roll(cb, -1, 1) - 4 * cb)
    curv = float(np.sqrt(np.mean(lap[m] ** 2))) if m.sum() else np.nan
    res = _detrend(np.asarray(lr_up, float), m); F = np.fft.fft2(res); P = np.abs(F) ** 2; H, W = lr_up.shape
    fy = np.fft.fftfreq(H)[:, None] * np.ones((1, W)); fx = np.fft.fftfreq(W)[None, :] * np.ones((H, 1)); fr = np.sqrt(fy ** 2 + fx ** 2)
    ann = (fr >= f_lo) & (fr <= f_hi) & (fr > 0); aniso = np.nan
    if ann.sum() >= 8:
        w = P[ann]; sw = w.sum() + 1e-30; cxx = np.sum(w * fx[ann] ** 2) / sw; cyy = np.sum(w * fy[ann] ** 2) / sw; cxy = np.sum(w * fx[ann] * fy[ann]) / sw
        ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
        if ev.min() > 0: aniso = float(np.sqrt(ev.max() / ev.min()))
    return C, curv, aniso

def r02_features_on_field(field, m, f_lr):
    Cc, curv, aniso = coarse_features(field, m, f_lr)
    sig = max(1.0 / f_lr / 2.0, 1.0); sm = gaussian_filter(np.where(np.isfinite(field), field, np.nanmean(field)).astype(float), sigma=sig); gy, gx = np.gradient(sm)
    return dict(C=float(Cc), curv=float(curv), aniso=float(aniso), slope=float(np.mean(np.sqrt(gx ** 2 + gy ** 2)[m])))

def hann2(h, w): return np.outer(np.hanning(h), np.hanning(w))

def radial_psd_cells(win, m):
    H, W = win.shape; yy, xx = np.mgrid[:H, :W]; A = np.c_[xx[m], yy[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, win[m], rcond=None); r = np.where(m, win - (coef[0] * xx + coef[1] * yy + coef[2]), 0.0)
    t = hann2(H, W); r = r * t / np.sqrt(np.mean(t ** 2))
    F = np.fft.fft2(r) / (H * W); P = np.abs(F) ** 2
    fy = np.fft.fftfreq(H)[:, None] * np.ones((1, W)); fx = np.fft.fftfreq(W)[None, :] * np.ones((H, 1)); fr = np.sqrt(fx ** 2 + fy ** 2)
    nb = max(4, min(H, W) // 2); edges = np.linspace(0, 0.5, nb + 1); idx = np.digitize(fr.ravel(), edges) - 1; ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb); pw = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=nb) / np.maximum(cnt, 1)
    return 0.5 * (edges[1:] + edges[:-1]), pw, cnt, P, fx, fy, fr

def spectral_extrap(win, m, k):
    f, pw, cnt, P, fx, fy, fr = radial_psd_cells(win, m); Wn = win.shape[0] * win.shape[1]
    sel = (cnt >= 8) & (f > 0) & (f <= 0.4) & (pw > 0)
    if sel.sum() < 3: return dict(beta=np.nan, logA=np.nan, fit_r2=np.nan, beta_clamped=np.nan, shat_fine=np.nan, noise_excess=np.nan, aniso_spec=np.nan, n_fit_bins=int(sel.sum()))
    lf, lp = np.log10(f[sel]), np.log10(pw[sel]); b, a = np.polyfit(lf, lp, 1); beta = -b; pred = a + b * lf
    r2 = 1 - np.sum((lp - pred) ** 2) / max(np.sum((lp - lp.mean()) ** 2), EPS); bc = float(np.clip(beta, 1.0, 5.0))
    A = 10 ** a; f1, f2 = 1.0, float(k)
    var = 2 * math.pi * A * Wn * ((f2 ** (2 - bc) - f1 ** (2 - bc)) / (2 - bc) if abs(bc - 2) > 1e-6 else math.log(f2 / f1))
    top = (cnt >= 4) & (f >= 0.4) & (f <= 0.5)
    noise = float(np.log10(np.mean(pw[top]) / np.mean(10 ** (a + b * np.log10(f[top]))))) if top.sum() else np.nan
    ann = (fr >= 0.05) & (fr <= 0.4)
    if ann.sum() >= 8:
        w = P[ann]; sw = w.sum() + 1e-30; cxx = np.sum(w * fx[ann] ** 2) / sw; cyy = np.sum(w * fy[ann] ** 2) / sw; cxy = np.sum(w * fx[ann] * fy[ann]) / sw
        ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]]); aniso = float(np.sqrt(ev.max() / ev.min())) if ev.min() > 0 else np.nan
    else: aniso = np.nan
    return dict(beta=float(beta), logA=float(a), fit_r2=float(r2), beta_clamped=bc, shat_fine=float(np.sqrt(max(var, 0.0))), noise_excess=noise, aniso_spec=aniso, n_fit_bins=int(sel.sum()))

# ---------------- R04 feature computation ----------------
NATIVE_KEYS = ["n_cells", "valid_frac", "mu_lr", "s_lr", "std_ship", "slope_ship_mean", "slope_ship_p90", "curv_ship_std",
               "beta", "logA", "fit_r2", "beta_clamped", "shat_fine", "noise_excess", "aniso_spec", "n_fit_bins", "C", "curv", "aniso", "slope"]
CONTRACT_KEYS = ["rsd_med", "rsd_p90", "sd_med", "logcount_med", "xtrack_mean", "beam_mean", "rsd_over_sd_med"]
HRGRID_KEYS = ["C_hr", "curv_hr", "aniso_hr", "slope_hr", "dist_edge", "frac_valid"]
ALL_KEYS = NATIVE_KEYS + CONTRACT_KEYS + HRGRID_KEYS

def nan_dict(keys): return {k: np.nan for k in keys}

def pair_context(ecp_g, row, products_dir, usable):
    """pair_id is read from row for the override."""
    """Ship-only context from the R02 pair_setup dict (HR keys are kept but shielded by ShipOnlyView) plus the
    LR validity on the LR grid (lr_valid.tif is on the HR grid: reprojected nearest) and the contract products."""
    g = ecp_g; lv = np.zeros(g["lr_arr"].shape, np.float32)
    reproject(source=g["lrv_raw"], destination=lv, src_transform=g["lrv_tr"], src_crs=g["lrv_crs"], dst_transform=g["lr_tr"], dst_crs=g["lr_crs"], resampling=Resampling.nearest, src_nodata=None, dst_nodata=0)
    ctx = ShipOnlyView(transform=g["transform"], dst_crs=g["dst_crs"], hr_res=g["hr_res"], lr_native=g["lr_native"], H=g["H"], W=g["W"],
                       lr_arr=g["lr_arr"], lr_tr=g["lr_tr"], lr_crs=g["lr_crs"], lrv_lr=(lv == 1) & np.isfinite(g["lr_arr"]),
                       lrv_raw=g["lrv_raw"], lrv_tr=g["lrv_tr"], lrv_crs=g["lrv_crs"], k=int(row["max_recoverable_k"]), cell_m=float(abs(g["lr_tr"].a)), usable=bool(usable), products=None, pair_id=str(row["pair_id"]),
                       hr_arr=g["hr_arr"], hrv_arr=g["hrv_arr"], joint=g["joint"], paths=g["paths"])
    # lr validity on the HR grid (for dist_edge / frac_valid): lr_valid.tif shares the HR grid on every pair (asserted; else reprojected)
    if g["lrv_raw"].shape == (g["H"], g["W"]) and g["lrv_tr"] == g["transform"]: lvh = g["lrv_raw"] == 1
    else:
        tmp = np.zeros((g["H"], g["W"]), np.float32)
        reproject(source=g["lrv_raw"], destination=tmp, src_transform=g["lrv_tr"], src_crs=g["lrv_crs"], dst_transform=g["transform"], dst_crs=g["dst_crs"], resampling=Resampling.nearest, src_nodata=None, dst_nodata=0)
        lvh = tmp == 1
    dict.__setitem__(ctx, "lrv_hr", lvh)
    if usable:
        prods = {}
        with rasterio.open(os.path.join(products_dir, "ship_count.tif")) as s0:
            assert s0.shape == g["lr_arr"].shape and s0.transform == g["lr_tr"], "products grid != lr.tif grid"
        for p in PRODUCTS:
            with rasterio.open(os.path.join(products_dir, f"{p}.tif")) as s: prods[p] = s.read(1).astype(np.float32)
        dict.__setitem__(ctx, "products", prods)
    return ctx

def ship_window(ctx, cy, cx, win_px):
    """Native ship-grid window centred on HR px (cy, cx): n = max(2, ceil(win_px / ratio)) cells. Returns (lr, valid, n, valid_frac, slices)."""
    x, y = ctx["transform"] * (cx, cy); lr_r, lr_c = rasterio.transform.rowcol(ctx["lr_tr"], x, y)
    ratio = ctx["cell_m"] / ctx["hr_res"]; n = max(2, int(math.ceil(win_px / ratio)))
    r1 = int(round(lr_r - n / 2.0)); c1 = int(round(lr_c - n / 2.0)); H, W = ctx["lr_arr"].shape
    rs, re_, cs, ce = max(0, r1), min(H, r1 + n), max(0, c1), min(W, c1 + n)
    if re_ - rs < 2 or ce - cs < 2: return None, None, n, 0.0, None
    sl = (slice(rs, re_), slice(cs, ce)); lr = ctx["lr_arr"][sl]; lv = ctx["lrv_lr"][sl] & np.isfinite(lr)
    return lr, lv, n, float(lv.sum() / (n * n)), sl

def native_features(lr, lv, n, vf, k, cell_m):
    out = nan_dict(NATIVE_KEYS); out["n_cells"] = n; out["valid_frac"] = vf
    if lr is None or lv.sum() < 2: return out
    vals = lr[lv].astype(float); out["mu_lr"] = float(vals.mean()); out["s_lr"] = float(max(vals.std(), S_FLOOR))
    fill = float(vals.mean()); w = np.where(lv, lr, fill).astype(float)
    if lv.sum() >= 16 and lv.sum() >= 0.25 * lv.size: out.update(spectral_extrap(w, lv, k))
    yy, xx = np.mgrid[:w.shape[0], :w.shape[1]]; A = np.c_[xx[lv], yy[lv], np.ones(lv.sum())]; coef, *_ = np.linalg.lstsq(A, w[lv], rcond=None)
    det = w - (coef[0] * xx + coef[1] * yy + coef[2]); out["std_ship"] = float(det[lv].std())
    gy, gx = np.gradient(w, cell_m); sl_ = np.sqrt(gx ** 2 + gy ** 2)[lv]; out["slope_ship_mean"] = float(sl_.mean()); out["slope_ship_p90"] = float(np.percentile(sl_, 90))
    lap = (np.roll(w, 1, 0) + np.roll(w, -1, 0) + np.roll(w, 1, 1) + np.roll(w, -1, 1) - 4 * w) / cell_m ** 2; out["curv_ship_std"] = float(lap[lv].std())
    if min(w.shape) >= 3:
        up = zoom(w, UP, order=3); m_up = zoom(lv.astype(float), UP, order=0) > 0.5
        if m_up.sum() >= 64: out.update(r02_features_on_field(up, m_up, 1.0 / UP))
    return out

def contract_features(ctx, sl):
    out = nan_dict(CONTRACT_KEYS)
    P = dict.get(ctx, "products")
    if not ctx["usable"] or P is None or sl is None: return out
    unavailable = {prod for (pid, prod) in OVERRIDES if pid == dict.get(ctx, "pair_id")}
    if unavailable:
        P = {k: (np.full_like(v, np.nan) if k in unavailable else v) for k, v in P.items()}
    rsd = P["ship_rsd"][sl]; sd = P["ship_sd"][sl]; cnt = P["ship_count"][sl]; xt = P["ship_xtrack_frac"][sl]; ba = P["ship_beam_angle"][sl]
    f = np.isfinite(rsd)
    if f.any(): out["rsd_med"] = float(np.median(rsd[f])); out["rsd_p90"] = float(np.percentile(rsd[f], 90))
    f = np.isfinite(sd)
    if f.any(): out["sd_med"] = float(np.median(sd[f]))
    f = np.isfinite(cnt) & (cnt > 0)
    if f.any(): out["logcount_med"] = float(np.median(np.log10(cnt[f])))
    f = np.isfinite(xt)
    if f.any(): out["xtrack_mean"] = float(xt[f].mean())
    f = np.isfinite(ba)
    if f.any(): out["beam_mean"] = float(ba[f].mean())
    f = np.isfinite(rsd) & np.isfinite(sd) & (sd > 0)
    if f.any(): out["rsd_over_sd_med"] = float(np.median(rsd[f] / sd[f]))
    return out

def hrgrid_features(ctx, r0, c0, win_px):
    """R02 path at the window footprint: LR area-averaged onto lr_px cells (from a cropped LR source with a 4-cell margin),
    mu fill, bicubic to win_px; roughness-3 + slope; dist_edge / frac_valid from the ship validity on the HR grid."""
    out = nan_dict(HRGRID_KEYS); H, W = ctx["H"], ctx["W"]
    r0c, c0c = max(0, r0), max(0, c0); r1c, c1c = min(H, r0 + win_px), min(W, c0 + win_px)
    if r1c - r0c < win_px // 2 or c1c - c0c < win_px // 2: return out
    lvh = ctx["lrv_hr"][r0c:r1c, c0c:c1c]; out["frac_valid"] = float(lvh.mean())
    if lvh.any(): out["dist_edge"] = float(np.mean(distance_transform_edt(lvh)[lvh]))
    b = rasterio.windows.bounds(rasterio.windows.Window(c0, r0, win_px, win_px), ctx["transform"])
    lr_px = max(2, int(round(win_px * ctx["hr_res"] / ctx["lr_native"])))
    # cropped source (margin 4 cells) for speed; not bit-identical to a full-raster reprojection (GDAL tie), documented
    lr_r0, lr_c0 = rasterio.transform.rowcol(ctx["lr_tr"], b[0], b[3]); lr_r1, lr_c1 = rasterio.transform.rowcol(ctx["lr_tr"], b[2], b[1])
    rs, re_ = max(0, min(lr_r0, lr_r1) - 4), min(ctx["lr_arr"].shape[0], max(lr_r0, lr_r1) + 5); cs, ce = max(0, min(lr_c0, lr_c1) - 4), min(ctx["lr_arr"].shape[1], max(lr_c0, lr_c1) + 5)
    if re_ - rs < 1 or ce - cs < 1: return out
    src = ctx["lr_arr"][rs:re_, cs:ce]; srcv = ctx["lrv_lr"][rs:re_, cs:ce].astype(np.float32); tr = ctx["lr_tr"] * rasterio.Affine.translation(cs, rs)
    dst_tr = rasterio.transform.from_bounds(b[0], b[1], b[2], b[3], lr_px, lr_px)
    lr = np.full((lr_px, lr_px), np.nan, np.float32); reproject(source=np.where(srcv > 0.5, src, np.nan).astype(np.float32), destination=lr, src_transform=tr, src_crs=ctx["lr_crs"], dst_transform=dst_tr, dst_crs=ctx["dst_crs"], resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    lv = np.zeros((lr_px, lr_px), np.float32); reproject(source=srcv, destination=lv, src_transform=tr, src_crs=ctx["lr_crs"], dst_transform=dst_tr, dst_crs=ctx["dst_crs"], resampling=Resampling.nearest, src_nodata=None, dst_nodata=0)
    lv = (lv == 1) & np.isfinite(lr)
    if lv.sum() < 2: return out
    mu = float(lr[lv].mean()); up = zoom(np.where(lv, lr, mu).astype(np.float32), win_px / lr_px, order=3)
    ci = (np.arange(win_px) * lr_px) // win_px; m = lv[ci][:, ci]
    if m.sum() < 64: return out
    f = r02_features_on_field(up.astype(float), m, ctx["hr_res"] / ctx["lr_native"])
    out.update(C_hr=f["C"], curv_hr=f["curv"], aniso_hr=f["aniso"], slope_hr=f["slope"])
    return out

def support_features(ctx, cy, cx):
    """All features for one support centre (HR px, float), both window sizes. Returns a flat dict with _128/_224 suffixes."""
    out = {}
    for w in WINDOWS_PX:
        lr, lv, n, vf, sl = ship_window(ctx, cy, cx, w)
        f = native_features(lr, lv, n, vf, ctx["k"], ctx["cell_m"]); f.update(contract_features(ctx, sl))
        f.update(hrgrid_features(ctx, int(round(cy - w / 2.0)), int(round(cx - w / 2.0)), w))
        out.update({f"{k}_{w}": v for k, v in f.items()})
    return out

def feature_columns():
    return [f"{k}_{w}" for w in WINDOWS_PX for k in ALL_KEYS]

def feature_hash(d): return hashlib.sha256(json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in sorted(d.items())}, default=float).encode()).hexdigest()
