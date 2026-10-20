"""R08 §3 — fine-field inspection of all 38 development pairs plus EW0207, from the frozen R04 field mosaics (F_hr, joint_v1).
§3.2 objective companions per pair: (1) directional concentration = fraction of fine-band [f_lr, f_hi] power of the
tile-averaged 2-D power spectrum of F_hr (128-px tiles of tiles_v1.1, NaN -> 0, no extra taper) in the most energetic 10 deg
orientation sector (isotropy: 1/18 = 0.056); (2) edge ratio = RMS of F_hr over joint-valid px within 3 px of an invalid px /
RMS over px farther than 10 px from any invalid px (Euclidean distance transform of the joint-valid mask); (3) median
tile A_fine with its 10th-90th percentiles.
§3.3 figures: set A (blind; hillshade of HR, F_hr at +/- 3 RMS, 500 m scale bar, no labels, order shuffled with
default_rng(20261010), codes P01..; key written to figures/key/figset_A_key.json) and set B (labelled; the same panels plus
the tile A_fine map and the GBT_ship prediction map, metrics in the caption). Pairs wider than 2.5 km show a 2 km x 2 km
subwindow: the largest square (<= 2 km) centred on the centroid of the joint-valid area whose joint-valid fraction is >= 0.9
(centre moved to the nearest valid px if the centroid is invalid). Hillshade: azimuth 315 deg, altitude 45 deg, vertical
exaggeration raised so that the window's 95th-percentile slope reads 20 deg where the true p95 slope is below 5 deg (recorded)."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r08_common as R8, r04_common as C4, targets_v1 as T
from preprocessing.colocated_utils import resolve_pair_paths
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TILE = T.TILE; DPI = 150
D = {}

def spectrum_sectors(Fhr, tl, f_lo, f_hi):
    P = np.zeros((TILE, TILE)); n = 0
    for tr0, tc0 in tl:
        a = Fhr[tr0:tr0 + TILE, tc0:tc0 + TILE]
        if a.shape != (TILE, TILE): continue
        a = np.where(np.isfinite(a), a, 0.0).astype(float); P += np.abs(np.fft.fft2(a)) ** 2; n += 1
    fy = np.fft.fftfreq(TILE)[:, None] * np.ones((1, TILE)); fx = np.fft.fftfreq(TILE)[None, :] * np.ones((TILE, 1)); fr = np.sqrt(fx ** 2 + fy ** 2)
    ann = (fr >= f_lo) & (fr <= f_hi) & (fr > 0)
    theta = np.degrees(np.arctan2(fy, fx)) % 180.0; sec = (np.floor(theta / R8.SECTOR_DEG).astype(int)) % (180 // R8.SECTOR_DEG)
    tot = P[ann].sum(); sums = np.bincount(sec[ann], weights=P[ann], minlength=180 // R8.SECTOR_DEG)
    i = int(np.argmax(sums)); return dict(n_tiles_fft=n, sector_frac_max=float(sums[i] / tot) if tot > 0 else np.nan, sector_deg_max=[i * R8.SECTOR_DEG, (i + 1) * R8.SECTOR_DEG], sector_fracs=(sums / tot).tolist() if tot > 0 else None, n_annulus_bins=int(ann.sum()))

def edge_ratio(Fhr, J):
    d = distance_transform_edt(J); near = J & (d <= R8.EDGE_NEAR_PX); far = J & (d > R8.EDGE_FAR_PX); del d
    rn = float(np.sqrt(np.mean(Fhr[near].astype(float) ** 2))) if near.any() else np.nan; rf = float(np.sqrt(np.mean(Fhr[far].astype(float) ** 2))) if far.any() else np.nan
    return dict(rms_near_m=rn, rms_far_m=rf, n_near=int(near.sum()), n_far=int(far.sum()), edge_ratio=(rn / rf if (far.any() and rf > 0) else np.nan))

def choose_window(J, hr_res):
    H, W = J.shape; full = max(H, W) * hr_res <= R8.SUBWIN_TRIGGER_M
    if full: return dict(subwindow=False, r0=0, c0=0, h=H, w=W, valid_frac=float(J.mean()), side_m=max(H, W) * hr_res)
    s = 8; Js = J[::s, ::s]; rr, cc = np.nonzero(Js); cy, cx = rr.mean(), cc.mean()
    if not Js[int(round(cy)), int(round(cx))]:
        i = int(np.argmin((rr - cy) ** 2 + (cc - cx) ** 2)); cy, cx = rr[i], cc[i]; moved = True
    else: moved = False
    cy, cx = int(round(cy * s)), int(round(cx * s)); side = int(round(R8.SUBWIN_M / hr_res)); best = None; min_side = side // 2
    while side >= min_side:
        h = side // 2; r0, c0 = max(0, cy - h), max(0, cx - h); r1, c1 = min(H, r0 + side), min(W, c0 + side); r0, c0 = max(0, r1 - side), max(0, c1 - side)
        vf = float(J[r0:r1, c0:c1].mean()); cand = dict(subwindow=True, r0=int(r0), c0=int(c0), h=int(r1 - r0), w=int(c1 - c0), valid_frac=vf, side_m=side * hr_res, centroid_moved=moved)
        if vf >= R8.SUBWIN_VALID_FRAC: return cand
        if best is None or vf > best["valid_frac"]: best = cand
        side = int(round(side * 0.95))
    best["below_valid_target"] = True; return best

def hillshade(z, res, az=315.0, alt=45.0):
    zz = np.where(np.isfinite(z), z, np.nanmean(z)); gy, gx = np.gradient(zz, res); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    p95 = float(np.nanpercentile(slope[np.isfinite(z)], 95)) if np.isfinite(z).any() else 0.0; zf = 1.0
    if p95 < 5.0 and p95 > 0: zf = np.tan(np.radians(20.0)) / np.tan(np.radians(p95))
    gx, gy = gx * zf, gy * zf; sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    azr, altr = np.radians(az), np.radians(alt); hs = np.sin(altr) * np.cos(sl) + np.cos(altr) * np.sin(sl) * np.cos(azr - asp)
    return np.where(np.isfinite(z), np.clip(hs, 0, 1), np.nan), zf, p95

def scalebar(ax, x0, y0, length_px, label=True):
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", linewidth=5, solid_capstyle="butt"); ax.plot([x0, x0 + length_px], [y0, y0], color="black", linewidth=2.2, solid_capstyle="butt")
    if label: ax.text(x0 + length_px / 2, y0 - 0.02 * (ax.get_ylim()[0] - ax.get_ylim()[1]), f"{int(R8.SCALEBAR_M)} m", ha="center", va="bottom", fontsize=8, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

def tile_map(tiles, vals, H, W):
    ng, mg = (H - T.MARGIN) // TILE + 1, (W - T.MARGIN) // TILE + 1; A = np.full((ng, mg), np.nan)
    for tr0, tc0, v in zip(tiles.tr0.values, tiles.tc0.values, vals): A[(tr0 - T.MARGIN) // TILE, (tc0 - T.MARGIN) // TILE] = v
    return A, (0 - 0.5 + T.MARGIN, mg * TILE - 0.5 + T.MARGIN, ng * TILE - 0.5 + T.MARGIN, 0 - 0.5 + T.MARGIN)   # imshow extent (left, right, bottom, top) in px

def one_pair(pid):
    t0 = time.time(); m = D["m"]; row = C4.pair_row(m, pid); unit = C4.UNIT_OF[pid]; tl = D["tiles"][D["tiles"].pair == pid]
    with rasterio.open(f"{R8.R04_FIELDS}/{pid}/F_hr.tif") as s: Fhr = s.read(1); tr = s.transform; hr_res = float(abs(tr.a)); H, W = s.shape
    with rasterio.open(f"{R8.R04_FIELDS}/{pid}/joint_v1.tif") as s: J = (s.read(1) == 1) & np.isfinite(Fhr)
    f_lo, f_hi = float(tl.f_lr.iloc[0]), float(tl.f_hi.iloc[0])
    rms = float(np.sqrt(np.mean(Fhr[J].astype(float) ** 2)))
    met = dict(pair=pid, unit=unit, n_tiles=len(tl), hr_res=hr_res, k=int(tl.k.iloc[0]), lr_native=float(tl.lr_native.iloc[0]), f_lr=f_lo, f_hi=f_hi, rms_Fhr_m=rms, n_joint_px=int(J.sum()),
               A_fine_median_m=float(tl.A_fine.median()), A_fine_p10_m=float(tl.A_fine.quantile(0.1)), A_fine_p90_m=float(tl.A_fine.quantile(0.9)))
    met.update(spectrum_sectors(Fhr, list(zip(tl.tr0.values, tl.tc0.values)), f_lo, f_hi)); met.update(edge_ratio(Fhr, J))
    win = choose_window(J, hr_res); met["window"] = win; r0, c0, h, w = win["r0"], win["c0"], win["h"], win["w"]
    paths = resolve_pair_paths(C4.DATA, row)
    with rasterio.open(paths["hr"]) as s:
        assert s.shape == (H, W) and s.transform == tr, "HR grid != field mosaic grid"
        hr = s.read(1, window=rasterio.windows.Window(c0, r0, w, h)).astype(np.float32); nod = s.nodata
        if nod is not None and np.isfinite(nod): hr = np.where(hr == nod, np.nan, hr)
    if paths.get("hr_valid"):
        with rasterio.open(paths["hr_valid"]) as s: hr = np.where(s.read(1, window=rasterio.windows.Window(c0, r0, w, h)) == 1, hr, np.nan)
    hs, zf, p95 = hillshade(hr, hr_res); met["hillshade"] = dict(z_factor=float(zf), p95_slope_deg=float(p95))
    Fw = Fhr[r0:r0 + h, c0:c0 + w]; del Fhr, J
    bar_px = R8.SCALEBAR_M / hr_res; ext = (c0 - 0.5, c0 + w - 0.5, r0 + h - 0.5, r0 - 0.5)
    def panels(axs, label):
        axs[0].imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest"); axs[1].imshow(Fw, cmap="RdBu_r", vmin=-3 * rms, vmax=3 * rms, extent=ext, interpolation="nearest")
        for ax in axs[:2]:
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_axis_off(); scalebar(ax, c0 + 0.04 * w, r0 + 0.95 * h, bar_px, label=label)
    # set A (blind)
    figA, axs = plt.subplots(1, 2, figsize=(10, 5.3), layout="constrained"); panels(axs, label=True); figA.suptitle(D["code"][pid], fontsize=11)
    os.makedirs(f"{R8.OUT_FIELDS}/pages", exist_ok=True); fa = f"{R8.OUT_FIELDS}/pages/A_{D['code'][pid]}.png"; figA.savefig(fa, dpi=DPI); plt.close(figA)
    # set B (labelled)
    pred = D["pred"].set_index("tile_id").GBT_ship_A_fine.reindex(tl.tile_id).values
    lo = np.log10(max(1e-4, min(np.nanmin(tl.A_fine.values), np.nanmin(pred)))); hi = np.log10(max(np.nanmax(tl.A_fine.values), np.nanmax(pred)))
    figB, axs = plt.subplots(1, 4, figsize=(20, 5.8), layout="constrained"); panels(axs, label=True)
    for ax, vals, ttl in ((axs[2], tl.A_fine.values, "tile A_fine (m, log10)"), (axs[3], pred, "GBT_ship prediction (m, log10)")):
        A, e = tile_map(tl, np.log10(np.maximum(vals, 1e-4)), H, W); im = ax.imshow(A, cmap="cividis", vmin=lo, vmax=hi, extent=e, interpolation="nearest"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_axis_off(); ax.set_title(ttl, fontsize=9)
    cb = figB.colorbar(im, ax=axs[3], fraction=0.05, pad=0.02, shrink=0.8); cb.set_label("log10 A_fine (m)", fontsize=8)
    cb2 = figB.colorbar(axs[1].images[0], ax=axs[1], fraction=0.05, pad=0.02, shrink=0.8); cb2.set_label("F_hr (m)", fontsize=8)
    axs[0].set_title("HR hillshade" + (f" (vertical exaggeration x{zf:.1f})" if zf != 1 else ""), fontsize=9); axs[1].set_title(f"F_hr, scale +/- 3 RMS ({3*rms:.3g} m)", fontsize=9)
    figB.suptitle(f"{pid}  ({unit}; code {D['code'][pid]})  k={met['k']}, lr_native={met['lr_native']:.1f} m, hr_res={hr_res:.2f} m" + (f"; {win['side_m']/1000:.1f} km subwindow at the valid-area centroid (valid {win['valid_frac']:.2f})" if win["subwindow"] else "; whole pair"), fontsize=10)
    figB.text(0.005, 0.005, f"sector fraction {met['sector_frac_max']:.3f} (isotropy {R8.ISO_FRAC:.3f}; sector {met['sector_deg_max'][0]}-{met['sector_deg_max'][1]} deg) | edge ratio {met['edge_ratio']:.2f} (RMS <=3 px {met['rms_near_m']:.4f} m / >10 px {met['rms_far_m']:.4f} m) | A_fine median {met['A_fine_median_m']:.4f} m, p10-p90 {met['A_fine_p10_m']:.4f}-{met['A_fine_p90_m']:.4f} m | {len(tl)} tiles", fontsize=8)
    fb = f"{R8.OUT_FIELDS}/pages/B_{unit}_{pid}.png"; figB.savefig(fb, dpi=DPI); plt.close(figB)
    met["page_A"] = fa; met["page_B"] = fb; met["seconds"] = time.time() - t0
    return met

def assemble_pdf(pngs, out):
    with PdfPages(out) as pdf:
        for p in pngs:
            img = plt.imread(p); fig = plt.figure(figsize=(img.shape[1] / DPI, img.shape[0] / DPI), dpi=DPI); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.set_axis_off(); pdf.savefig(fig, dpi=DPI); plt.close(fig)

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1"); workers = int(os.environ.get("R08_FIELD_WORKERS", 4)); t0 = time.time()
    m = R8.start("fields_r08"); os.makedirs(R8.OUT_FIELDS, exist_ok=True); os.makedirs(f"{R8.FIGURES}/key", exist_ok=True)
    tiles = R8.load_tiles(); P, rules = R8.load_pred_tile(); PE = pd.read_parquet(R8.PRED_TILE_EW_R05)
    pairs = sorted(tiles.pair.unique()); C4.assert_no_sealed(pairs); assert len(pairs) == 39
    rng = np.random.default_rng(R8.SEED_FIGSET_A); order = [pairs[i] for i in rng.permutation(len(pairs))]; code = {p: f"P{i+1:02d}" for i, p in enumerate(order)}
    key = dict(seed=R8.SEED_FIGSET_A, order=[dict(code=code[p], pair=p, unit=C4.UNIT_OF[p]) for p in order])
    R8.dump(key, f"{R8.FIGURES}/key/figset_A_key.json"); R8.dump(key, f"{R8.OUT_FIELDS}/figset_A_key.json")
    D.update(m=m, tiles=tiles, pred=pd.concat([P[["tile_id", "GBT_ship_A_fine"]], PE[["tile_id", "GBT_ship_A_fine"]]], ignore_index=True), code=code)
    with Pool(workers) as pool: mets = pool.map(one_pair, pairs, chunksize=1)
    R8.dump(dict(criteria=dict(sector_deg=R8.SECTOR_DEG, isotropy_frac=R8.ISO_FRAC, edge_near_px=R8.EDGE_NEAR_PX, edge_far_px=R8.EDGE_FAR_PX, subwindow_m=R8.SUBWIN_M, subwindow_trigger_m=R8.SUBWIN_TRIGGER_M, subwindow_valid_frac=R8.SUBWIN_VALID_FRAC, scalebar_m=R8.SCALEBAR_M), pairs=mets), f"{R8.OUT_FIELDS}/field_metrics.json")
    flat = [{k: v for k, v in r.items() if k not in ("sector_fracs", "window", "hillshade", "page_A", "page_B")} | dict(subwindow=r["window"]["subwindow"], window_side_m=r["window"]["side_m"], window_valid_frac=r["window"]["valid_frac"], hillshade_z_factor=r["hillshade"]["z_factor"], low_amplitude=(r["unit"] in R8.LOW_AMP)) for r in mets]
    pd.DataFrame(flat).to_csv(f"{R8.OUT_FIELDS}/field_metrics.csv", index=False)
    by = {r["pair"]: r for r in mets}
    assemble_pdf([by[p]["page_A"] for p in order], f"{R8.FIGURES}/R08_figset_A_blind.pdf")
    assemble_pdf([by[p]["page_B"] for p in sorted(pairs, key=lambda p: (C4.UNIT_OF[p], p))], f"{R8.FIGURES}/R08_figset_B_labelled.pdf")
    for r in mets: print(f"  {r['unit']} {r['pair']}: sector {r['sector_frac_max']:.3f} edge {r['edge_ratio']:.2f} A_fine {r['A_fine_median_m']:.4f} m ({r['seconds']:.0f} s)")
    print(f"done ({time.time()-t0:.0f} s)")
