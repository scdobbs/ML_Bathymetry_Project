"""R07 §1.4 — pretraining windows in the development representation, through the UNCHANGED R06 / R02 builder path
(extract_clean_patches.tile_ship_side + build_inputs; products as in build_inputs_r06.window_inputs), split in two so
that the expensive part can be cached:
  extract_lr_tile(g, P, r0, c0)  cache time: the window's LR tile (area-average onto lr_px cells), its validity, and the
                                  4 contract products reprojected (average) onto the same lr_px grid, raw units.
  assemble(tile, ...)             load time: build_inputs (mu_lr / s_lr over valid cells, fill, bicubic, normalise), the
                                  LR-validity channel, the FiLM vector [log2 cell, log2 hr_res, log2 s_lr, log2 |mu_lr|],
                                  and the V2 product channels + masks (norm, fill with the mean of the available cells,
                                  bicubic; mask nearest-upsampled); with an optional set of LR cells marked missing
                                  (§2.2 masked-cell reconstruction: they enter exactly as missing LR does).
assemble(extract_lr_tile(...)) must be byte-identical to build_inputs_r06.window_inputs on development pairs (the §1.4
equivalence test, equiv_test_r07.py). A corpus block plays the role of a development pair through block_geom(): its
grid.tif is lr.tif, validity = finite cells, and the HR grid is synthetic (pixel = the drawn development hr_res, anchored
at the grid's top-left corner, same CRS). The sub-cell sounding-spread target (§2.2 item 2) is rsd_block_target()."""
from __future__ import annotations
import math, os
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import r07_common as R7
from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import reproject_from_array
from preprocessing.data_utils import bicubic_upsample

WIN, TILE, MARGIN, BLOCK, NB = R7.WIN, R7.TILE, R7.MARGIN, R7.BLOCK, R7.NB
PRODUCTS = R7.PRODUCTS; S_FLOOR = R7.S_FLOOR

def norm_product(name, a):
    """build_inputs_r06.norm_product, verbatim."""
    if name == "ship_rsd": return np.log10(np.maximum(a, 1e-3))
    if name == "ship_count": return np.log10(a)
    if name == "ship_xtrack_frac": return a
    if name == "ship_beam_angle": return a / 90.0
    raise KeyError(name)

# ---------------------------------------------------------------- geometry
def lr_px_of(hr_res, lr_native): return max(2, int(round(WIN * hr_res / lr_native)))     # extract_clean_patches.pair_setup rule

def block_geom(grid_path, cell_m, hr_res):
    """A corpus block as a development-style geometry dict (the fields tile_ship_side / window_inputs read)."""
    with rasterio.open(grid_path) as s:
        lr_arr, lr_tr, lr_crs = ecp.load_band(s); nrow, ncol = s.height, s.width
    lrv_raw = np.isfinite(lr_arr).astype(np.float32)                     # validity = finite grid cells (no lr_valid.tif for blocks)
    H = int(math.floor(nrow * abs(lr_tr.e) / hr_res)); W = int(math.floor(ncol * abs(lr_tr.a) / hr_res))
    transform = Affine(hr_res, 0.0, lr_tr.c, 0.0, -hr_res, lr_tr.f)      # synthetic HR grid, anchored at the grid's top-left, same CRS
    return dict(lr_arr=lr_arr, lr_tr=lr_tr, lr_crs=lr_crs, lrv_raw=lrv_raw, lrv_tr=lr_tr, lrv_crs=lr_crs, lr_native=float(cell_m), hr_res=float(hr_res),
                dst_crs=lr_crs, transform=transform, H=H, W=W, lr_px=lr_px_of(hr_res, cell_m), finite_frac=float(lrv_raw.mean()))

def load_block_products(block_dir, g, usable):
    """The 4 contract products of a usable block, on the grid.tif grid (asserted), raw units; None when not usable."""
    if not usable: return None
    P = {}
    for p in PRODUCTS:
        with rasterio.open(os.path.join(block_dir, "ship_products_v2_1", f"{p}.tif")) as s:
            assert s.shape == g["lr_arr"].shape and s.transform == g["lr_tr"], f"{p}: products grid != grid.tif grid"
            P[p] = s.read(1).astype(np.float32)
    return P

# ---------------------------------------------------------------- cache-time step
def extract_lr_tile(g, P, r0, c0):
    """The window's LR tile over the exact HR window footprint (tile_ship_side) and the products reprojected onto the
    same lr_px grid (the reproject_from_array call of build_inputs_r06.window_inputs, verbatim). Raw units."""
    lr, lr_valid = ecp.tile_ship_side(g, r0, c0, WIN)
    prod_a = None
    if P is not None:
        b = rasterio.windows.bounds(rasterio.windows.Window(c0, r0, WIN, WIN), g["transform"])
        prod_a = np.stack([reproject_from_array(P[p], g["lr_tr"], g["lr_crs"], b, g["lr_px"], g["dst_crs"], Resampling.average) for p in PRODUCTS], 0)
    return dict(lr=lr, lr_valid=lr_valid, prod_a=prod_a, lr_px=int(g["lr_px"]), hr_res=float(g["hr_res"]), lr_native=float(g["lr_native"]), r0=int(r0), c0=int(c0))

# ---------------------------------------------------------------- load-time step
def assemble(lr, lr_valid, prod_a, lr_native, hr_res, mask=None, products=True, drop_rsd=False):
    """build_inputs_r06.window_inputs from a cached tile. mask (lr_px x lr_px bool, True = marked missing) removes cells
    from the LR before anything else, exactly as missing LR cells are treated (invalid -> filled with mu_lr, validity 0).
    Returns None if fewer than 2 valid cells remain (build_inputs rule)."""
    lv = lr_valid.astype(bool)
    if mask is not None: lv = lv & ~mask
    lr_m = np.where(lv, lr, np.nan).astype(np.float32)
    inp = ecp.build_inputs(lr_m, lv, WIN, S_FLOOR)
    if inp is None: return None
    cond = np.array([np.log2(lr_native), np.log2(hr_res), np.log2(inp["s_lr"]), np.log2(abs(inp["mu_lr"]))], np.float32)
    out = dict(x=inp["x"], lrv=inp["lr_valid_hr"].astype(np.uint8), mu_lr=inp["mu_lr"], s_lr=inp["s_lr"], n_lr_valid=inp["n_lr_valid"], cond=cond,
               lr_up_m=inp["lr_up_m"], prod=None, prodm=None)
    if products:
        prod = np.zeros((len(PRODUCTS), WIN, WIN), np.float32); prodm = np.zeros((len(PRODUCTS), WIN, WIN), np.uint8)
        if prod_a is not None:
            for j, p in enumerate(PRODUCTS):
                if drop_rsd and p == "ship_rsd": continue           # P-V2 pretraining: ship_rsd channel and mask zero on every window
                a = prod_a[j]
                avail = np.isfinite(a) & ((a > 0) if p == "ship_count" else True)
                if avail.any():
                    v = norm_product(p, np.where(avail, a, 1.0)); fill = float(v[avail].mean())
                    prod[j] = bicubic_upsample(np.where(avail, v, fill).astype(np.float32), WIN); prodm[j] = ecp.nearest_up(avail, WIN).astype(np.uint8)
        out["prod"], out["prodm"] = prod, prodm
    return out

def window_inputs(g, P, r0, c0, products=True):
    """The whole path in one call (what the §1.4 equivalence test compares with build_inputs_r06.window_inputs)."""
    t = extract_lr_tile(g, P, r0, c0)
    return assemble(t["lr"], t["lr_valid"], t["prod_a"], t["lr_native"], t["hr_res"], products=products)

def draw_mask(lr_valid, frac, rng):
    """§2.2 item 1: mark round(frac x n_valid) of the valid LR cells as missing, chosen uniformly."""
    lv = lr_valid.astype(bool); idx = np.flatnonzero(lv); n_mask = int(round(frac * len(idx)))
    m = np.zeros(lv.shape, bool)
    if n_mask: m.flat[rng.choice(idx, n_mask, replace=False)] = True
    return m

def n_mask_of(n_valid, frac=R7.MASK_FRAC): return int(round(frac * n_valid))

# ---------------------------------------------------------------- sub-cell sounding-spread target (§2.2 item 2)
def weighted_median(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])

def rsd_block_target(rsd, lr_tr, hr_transform, r0, c0, cover_min=R7.BLOCK_COVER_MIN, floor=R7.RSD_FLOOR_M):
    """log10 ship_rsd on the 4 x 4 block grid of the central 128-px tile: area-weighted median over the native product
    cells covering each 32-px block (overlap area as weight; cells with finite rsd only). A block is covered when the
    finite-rsd overlap is >= cover_min of its area; others are NaN / False. Native grid and HR grid share the CRS."""
    tgt = np.full((NB, NB), np.nan, np.float32); cov = np.zeros((NB, NB), bool)
    a, e, cx, fy = lr_tr.a, lr_tr.e, lr_tr.c, lr_tr.f; nrow, ncol = rsd.shape
    for i in range(NB):
        for j in range(NB):
            left, bottom, right, top = rasterio.windows.bounds(rasterio.windows.Window(c0 + MARGIN + j * BLOCK, r0 + MARGIN + i * BLOCK, BLOCK, BLOCK), hr_transform)
            c_lo = max(0, int(math.floor((left - cx) / a))); c_hi = min(ncol - 1, int(math.ceil((right - cx) / a)) - 1)
            r_lo = max(0, int(math.floor((top - fy) / e))); r_hi = min(nrow - 1, int(math.ceil((bottom - fy) / e)) - 1)
            if c_hi < c_lo or r_hi < r_lo: continue
            cols = np.arange(c_lo, c_hi + 1); rows = np.arange(r_lo, r_hi + 1)
            x0 = cx + cols * a; x1 = x0 + a; y1 = fy + rows * e; y0 = y1 + e            # e < 0: y1 = top edge, y0 = bottom edge
            wx = np.clip(np.minimum(right, x1) - np.maximum(left, x0), 0, None); wy = np.clip(np.minimum(top, y1) - np.maximum(bottom, y0), 0, None)
            w = wy[:, None] * wx[None, :]; v = rsd[r_lo:r_hi + 1, c_lo:c_hi + 1]
            ok = np.isfinite(v) & (w > 0)
            if not ok.any(): continue
            area = (right - left) * (top - bottom)
            if w[ok].sum() / area < cover_min: continue
            tgt[i, j] = weighted_median(np.log10(np.maximum(v[ok], floor)).astype(np.float64), w[ok]); cov[i, j] = True
    return tgt, cov
