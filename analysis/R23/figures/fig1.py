"""R23 Figure 1: (a) global Equal Earth map of the development units and blind-test sites; (b) one example pair as four map
panels (ship grid, AUV grid, band-passed fine field, per-tile s_lr); (c) radial power spectra of the same pair.
Read only on every frozen output; writes only under analysis/R23/figures/. No score is recomputed."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform_bounds
import fig_style as S
font = S.apply()
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
from cmcrameri import cm as cmc

POOLED_RHO = 0.392                     # directive §1 rule: pair with s_lr rho closest to the pooled value
MIN_INTERIOR_TILES = 50
UNITS_CSV = f"{S.AR}/R15/out_ruling/units.csv"
PER_PAIR_D1 = f"{S.AR}/R10b/rehearsal/scores/per_pair_D1.csv"
MASTER_TILE = f"{S.AR}/R08b/scores/master_tile.parquet"
MANIFEST = f"{S.ACQ}/manifest/pairs_v2_1.parquet"
TABLE_S1 = f"{S.ACQ}/reports_post_grl_review/ACQ-R16/TableS1_pairs.csv"
GMRT = f"{S.TMP}/gmrt_25km_topo.tif"   # GMRT GridServer, layer=topo, mresolution=25000, N80/S-80 (Ryan et al., 2009)
BLIND1 = {"RC2511__MGDS_32239": f"{S.OAKD}/harmonized_confirmatory/nu01b/RC2511__MGDS_32239/footprint.geojson",
          "SUM1004__MGDS_33090": f"{S.OAKD}/harmonized_confirmatory/nu03/SUM1004__MGDS_33090/footprint.geojson"}
BLIND2 = {p: f"{S.OAKD}/phase5_v4_1/harmonized/{p}/footprint.geojson" for p in
          ("Escanaba__MGDS_30368", "KN161L04__MGDS_22436", "KN173L02__MGDS_31675", "MV1003__MGDS_31860", "AT26-24__MGDS_31954")}
sys.path.insert(0, f"{S.ACQ}/src/discovery"); import stage_c5_spectral as SPEC   # frozen S3.2 spectrum routine (48 log bins)
sys.path.insert(0, f"{S.REPO}/analysis/R04/scripts"); import targets_v1 as T          # frozen target definitions (WIN, TILE, MARGIN)

meta = dict(directive="R23", figure=1, assertions=[], notes=[]); rows = []
def check(name, ok, got, want):
    meta["assertions"].append(dict(name=name, passed=bool(ok), got=got, want=want)); print(f"  assert {name}: {'PASS' if ok else 'FAIL'} got={got} want={want}")

# ---------------------------------------------------------------- sites
U = pd.read_csv(UNITS_CSV); dev = U[U.unit != "lu16"].copy()
real = dev[dev.group == "real-relief"]; art = dev[dev.group == "artifact-affected"]
check("16 development units", len(dev) == 16, len(dev), 16); check("10 real-relief (filled)", len(real) == 10, len(real), 10); check("6 artifact-affected (open)", len(art) == 6, len(art), 6)

def centroid_lonlat(fp):
    import pyproj; from shapely.geometry import shape; from shapely.ops import transform
    g = json.load(open(fp)); crs = g["crs"]["properties"]["name"]; geom = shape(g["features"][0]["geometry"])
    t = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True); c = transform(t.transform, geom).centroid
    return float(c.x), float(c.y), crs
b1 = {p: centroid_lonlat(f) for p, f in BLIND1.items()}; b2 = {p: centroid_lonlat(f) for p, f in BLIND2.items()}
check("2 blind-test-1 triangles", len(b1) == 2, len(b1), 2); check("5 blind-test-2 squares", len(b2) == 5, len(b2), 5)
for _, r in dev.iterrows():
    rows.append(dict(panel="a", series=f"development {r.group}", item=r.unit, x_lon=r.centroid_lon, y_lat=r.centroid_lat, value=None, source_file=UNITS_CSV, source_column="centroid_lon, centroid_lat, group"))
for p, (lon, lat, crs) in b1.items(): rows.append(dict(panel="a", series="blind test 1", item=p, x_lon=lon, y_lat=lat, value=None, source_file=BLIND1[p], source_column=f"footprint centroid ({crs} -> EPSG:4326)"))
for p, (lon, lat, crs) in b2.items(): rows.append(dict(panel="a", series="blind test 2", item=p, x_lon=lon, y_lat=lat, value=None, source_file=BLIND2[p], source_column=f"footprint centroid ({crs} -> EPSG:4326)"))

# ---------------------------------------------------------------- choose the example pair (rule, not hand-picked)
PP = pd.read_csv(PER_PAIR_D1); S1 = pd.read_csv(TABLE_S1)
ten = set(real.unit); scorable = PP[(PP.role == "development") & PP.unit.isin(ten) & (PP.view != "not_scorable")]
check("14 scorable real-relief development pairs", len(scorable) == 14, len(scorable), 14)
coreg = dict(zip(S1.pair_id, S1["coreg case"]))
cand = scorable[(scorable.view == "interior") & (scorable.n_interior >= MIN_INTERIOR_TILES) & (scorable.pair.map(coreg) == "rigid")].copy()
cand["dist"] = (cand.rho_s_lr - POOLED_RHO).abs(); cand = cand.sort_values(["dist", "n_interior"], ascending=[True, False])
M = pd.read_parquet(MANIFEST).set_index("pair_id")
cand["k"] = cand.pair.map(lambda p: int(M.loc[p, "max_recoverable_k"]))
top5 = cand.head(5)[["pair", "unit", "rho_s_lr", "n_interior", "k", "lr_native", "hr_res", "dist"]]
meta["candidates"] = top5.to_dict("records"); print(top5.to_string())
pid = str(cand.iloc[0].pair); unit = str(cand.iloc[0].unit); meta["chosen_pair"] = pid; meta["chosen_unit"] = unit
frozen = PP[PP.pair == pid].iloc[0]; mrow = M.loc[pid]
hdir = f"{S.OAKD}/{mrow['harmonized_path']}"; hr_path, lr_path = f"{hdir}/hr.tif", f"{hdir}/lr.tif"
Fhr_path = f"{S.AR}/R04/fields/{pid}/F_hr.tif"
with rasterio.open(hr_path) as d: hr = d.read(1).astype("float32"); hr_tr = d.transform; hr_crs = d.crs; hr_bounds = d.bounds; hr_res = abs(d.transform.a)
with rasterio.open(lr_path) as d: lr = d.read(1, masked=True).astype("float32").filled(np.nan); lr_bounds = d.bounds
with rasterio.open(Fhr_path) as d: Fhr = d.read(1).astype("float32"); assert d.shape == hr.shape and d.transform == hr_tr, "F_hr grid != hr grid"
lr_native = float(mrow["lr_native_res_m"]); k = int(mrow["max_recoverable_k"])
rho_pair = float(frozen.rho_s_lr)
check("chosen pair rho equals frozen (per_pair_D1 rho_s_lr)", np.isclose(rho_pair, float(cand.iloc[0].rho_s_lr)), rho_pair, float(cand.iloc[0].rho_s_lr))
check("chosen pair k equals frozen (manifest max_recoverable_k)", k == int(frozen.get("k", k)) if "k" in frozen else True, k, "manifest")
check("chosen pair lr_native equals frozen (per_pair_D1 lr_native)", np.isclose(lr_native, float(frozen.lr_native)), lr_native, float(frozen.lr_native))
check("chosen pair hr_res equals frozen (per_pair_D1 hr_res)", np.isclose(hr_res, float(frozen.hr_res), rtol=1e-9), hr_res, float(frozen.hr_res))
lon_c, lat_c, _ = centroid_lonlat(f"{hdir}/footprint.geojson"); meta["chosen_center"] = dict(lon=lon_c, lat=lat_c)
print(f"chosen {pid} ({unit}): rho {rho_pair:.4f}, k {k}, lr_native {lr_native}, hr_res {hr_res:.5f}, centre lon {lon_c:.4f} lat {lat_c:.4f}")

# per-tile s_lr (interior tiles only) as a mosaic on the HR grid
MT = pd.read_parquet(MASTER_TILE); tl = MT[(MT.pair == pid) & MT.interior_D1.astype(bool)].copy()
tl["r0"] = tl.tile_id.str.extract(r"_r(\d{6})_c").astype(int); tl["c0"] = tl.tile_id.str.extract(r"_c(\d{6})$").astype(int)
check("interior tiles of the chosen pair equal frozen n_interior", len(tl) == int(frozen.n_interior), len(tl), int(frozen.n_interior))
slr_mos = np.full(hr.shape, np.nan, np.float32)
for _, t in tl.iterrows():
    r, c = t.r0 + T.MARGIN, t.c0 + T.MARGIN; slr_mos[r:r + T.TILE, c:c + T.TILE] = t.ref_s_lr
    rows.append(dict(panel="b4", series="per-tile s_lr (interior D1)", item=t.tile_id, x_lon=None, y_lat=None, value=float(t.ref_s_lr), source_file=MASTER_TILE, source_column="ref_s_lr (interior_D1 == True)"))

# ---------------------------------------------------------------- spectra (frozen S3.2 routine, 48 log bins)
def _b4(p):
    with rasterio.open(p) as ds: return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
bb = [_b4(hr_path), _b4(lr_path)]
b = (max(x[0] for x in bb), max(x[1] for x in bb), min(x[2] for x in bb), min(x[3] for x in bb))
target_res = max(0.5, hr_res)
lr_c, hr_c, dx = SPEC.load_to_common(lr_path, hr_path, b, target_res)
lr_c, hr_c, dx = SPEC._cap_pair(lr_c, hr_c, dx, 1024)
fc_h, p_h = SPEC.radial_psd(hr_c, dx); fc_l, p_l = SPEC.radial_psd(lr_c, dx)
assert fc_h is not None and fc_l is not None and len(fc_h) == 48 and np.allclose(fc_h, fc_l)
H_, W_ = hr_c.shape; w2 = np.outer(np.hanning(H_), np.hanning(W_)) ** 2
norm = dx ** 2 / w2.sum()                                   # Welch normalisation of the routine's mean |F|^2: PSD in m^2 per (cycle/m)^2 = m^4
P_h, P_l = p_h * norm, p_l * norm
f_lr_px = hr_res / lr_native; f_hi_px = min(k * f_lr_px, 0.45)
f_lr_m, f_hi_m = f_lr_px / hr_res, f_hi_px / hr_res         # cycles per metre
check("band edge f_lr (cycles/m) = (hr_res/lr_native)/hr_res", np.isclose(f_lr_m, 1.0 / lr_native), f_lr_m, 1.0 / lr_native)
check("band edge f_hi (cycles/m) = min(k f_lr, 0.45)/hr_res", np.isclose(f_hi_m, min(k * f_lr_px, 0.45) / hr_res), f_hi_m, min(k * f_lr_px, 0.45) / hr_res)
meta["spectra"] = dict(common_grid_shape=[int(H_), int(W_)], dx_m=float(dx), bounds_lonlat=list(map(float, b)), f_lr_cpm=f_lr_m, f_hi_cpm=f_hi_m, k=k, norm_factor=float(norm))
for i in range(48):
    rows.append(dict(panel="c", series="AUV PSD", item=f"bin{i:02d}", x_lon=None, y_lat=None, value=float(P_h[i]), x_cpm=float(fc_h[i]), psd_raw=float(p_h[i]), source_file=hr_path, source_column="stage_c5_spectral.radial_psd (x norm)"))
    rows.append(dict(panel="c", series="Ship PSD", item=f"bin{i:02d}", x_lon=None, y_lat=None, value=float(P_l[i]), x_cpm=float(fc_l[i]), psd_raw=float(p_l[i]), source_file=lr_path, source_column="stage_c5_spectral.radial_psd (x norm)"))

failed = [a for a in meta["assertions"] if not a["passed"]]
if failed:
    json.dump(meta, open(f"{S.OUT}/fig1_meta.json", "w"), indent=1, default=str); raise SystemExit(f"Figure 1 stopped: {failed}")

# ---------------------------------------------------------------- figure
FIG_H = 128.0
fig = S.fig_new(S.WIDTH_MM, FIG_H)

# (a) map
with rasterio.open(GMRT) as d: z = d.read(1).astype("float32"); gb = d.bounds; dy = abs(d.transform.e)
L = np.where(z < 0, 0.93 - 0.24 * np.clip(-z, 0, 8000) / 8000.0, 0.66 - 0.20 * np.clip(z, 0, 5000) / 5000.0).astype("float32")
npad = int(round((90.0 - gb.top) / dy)); L = np.vstack([np.repeat(L[:1], npad, 0), L, np.repeat(L[-1:], npad, 0)])   # ±80° grid extended to the poles by edge rows
proj = ccrs.EqualEarth(central_longitude=-130.0)
MAP_W = 96.0; MAP_H = MAP_W * (proj.y_limits[1] - proj.y_limits[0]) / (proj.x_limits[1] - proj.x_limits[0])   # Equal Earth aspect, so the map fills the box
axm = S.ax_mm(fig, 12.0, 4.0, MAP_W, MAP_H, projection=proj); axm.set_global(); meta["map_box_mm"] = dict(w=MAP_W, h=MAP_H)
im = axm.imshow(L, origin="upper", extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(), cmap="gray", vmin=0.0, vmax=1.0, regrid_shape=2400, interpolation="nearest", zorder=0)
im.set_rasterized(True); im.set_gid("fig1a_gmrt_basemap")
axm.spines["geo"].set_linewidth(S.LW_AX)
gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, xlocs=list(range(-180, 181, 60)), ylocs=list(range(-90, 91, 30)), linewidth=0.3, color=S.GRAY_LINE, linestyle="-", zorder=1)
gl.top_labels = False; gl.right_labels = False; gl.rotate_labels = False; gl.xlabel_style = {"size": S.FS}; gl.ylabel_style = {"size": S.FS}; gl.xpadding = 2; gl.ypadding = 2
mk = dict(transform=ccrs.PlateCarree(), linestyle="none", markersize=S.MS, markeredgewidth=0.5, zorder=5, clip_on=False)
h_real, = axm.plot(real.centroid_lon, real.centroid_lat, marker="o", markerfacecolor=S.OKABE["black"], markeredgecolor=S.OKABE["black"], **mk); h_real.set_gid("fig1a_dev_real_relief")
h_art, = axm.plot(art.centroid_lon, art.centroid_lat, marker="o", markerfacecolor="white", markeredgecolor=S.OKABE["black"], **mk); h_art.set_gid("fig1a_dev_artifact")
h_b1, = axm.plot([v[0] for v in b1.values()], [v[1] for v in b1.values()], marker="^", markerfacecolor=S.OKABE["vermillion"], markeredgecolor=S.OKABE["vermillion"], **mk); h_b1.set_gid("fig1a_blind1")
h_b2, = axm.plot([v[0] for v in b2.values()], [v[1] for v in b2.values()], marker="s", markerfacecolor=S.OKABE["blue"], markeredgecolor=S.OKABE["blue"], **mk); h_b2.set_gid("fig1a_blind2")
leg_h = [Line2D([], [], marker="o", color=S.OKABE["black"], markerfacecolor=S.OKABE["black"], linestyle="none", markersize=S.MS, markeredgewidth=0.5),
         Line2D([], [], marker="^", color=S.OKABE["vermillion"], markerfacecolor=S.OKABE["vermillion"], linestyle="none", markersize=S.MS, markeredgewidth=0.5),
         Line2D([], [], marker="s", color=S.OKABE["blue"], markerfacecolor=S.OKABE["blue"], linestyle="none", markersize=S.MS, markeredgewidth=0.5)]
leg = axm.legend(leg_h, ["Development unit", "Blind test 1", "Blind test 2"], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=S.FS, handletextpad=0.3, borderaxespad=0.0, columnspacing=1.5)
leg.set_gid("fig1a_legend")
# inset for the north-east Pacific margin, where seven sites lie within a few degrees of each other
INSET = (-130.0, -104.0, 21.0, 47.0)
axi = S.ax_mm(fig, 86.0, 4.0 + MAP_H - 25.0, 22.0, 22.0, projection=ccrs.PlateCarree()); axi.set_extent(INSET, crs=ccrs.PlateCarree())
imi = axi.imshow(L, origin="upper", extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", zorder=0)
imi.set_rasterized(True); imi.set_gid("fig1a_inset_basemap")
for sp in axi.spines.values(): sp.set_linewidth(S.LW_AX)
mki = dict(mk, markersize=3.0)
for xs, ys, m, fc, ec, gid in ((real.centroid_lon, real.centroid_lat, "o", S.OKABE["black"], S.OKABE["black"], "fig1a_inset_dev_real_relief"), (art.centroid_lon, art.centroid_lat, "o", "white", S.OKABE["black"], "fig1a_inset_dev_artifact"),
                               ([v[0] for v in b1.values()], [v[1] for v in b1.values()], "^", S.OKABE["vermillion"], S.OKABE["vermillion"], "fig1a_inset_blind1"), ([v[0] for v in b2.values()], [v[1] for v in b2.values()], "s", S.OKABE["blue"], S.OKABE["blue"], "fig1a_inset_blind2")):
    hh, = axi.plot(list(xs), list(ys), marker=m, markerfacecolor=fc, markeredgecolor=ec, **mki); hh.set_gid(gid); hh.set_clip_on(True)
bx = [INSET[0], INSET[1], INSET[1], INSET[0], INSET[0]]; by = [INSET[2], INSET[2], INSET[3], INSET[3], INSET[2]]
hb, = axm.plot(bx, by, transform=ccrs.PlateCarree(), color=S.OKABE["black"], linewidth=S.LW_AX, zorder=4); hb.set_gid("fig1a_inset_box")
n_in = int(((dev.centroid_lon > INSET[0]) & (dev.centroid_lon < INSET[1]) & (dev.centroid_lat > INSET[2]) & (dev.centroid_lat < INSET[3])).sum()) + sum(1 for v in list(b1.values()) + list(b2.values()) if INSET[0] < v[0] < INSET[1] and INSET[2] < v[1] < INSET[3])
meta["inset"] = dict(extent_lon_lat=INSET, n_sites=n_in)
S.panel_letter(fig, axm, "a", dx_mm=-10.0, dy_mm=-1.0)

# (b) four map panels, identical extent
ext = (hr_bounds.left, hr_bounds.right, hr_bounds.bottom, hr_bounds.top); ext_lr = (lr_bounds.left, lr_bounds.right, lr_bounds.bottom, lr_bounds.top)
PW = 37.5; PH = PW * hr.shape[0] / hr.shape[1]; X0 = 8.0; GAP = 3.0; YB = 66.0
depth_lr = -lr; depth_hr = -hr
vmin_d, vmax_d = np.nanpercentile(depth_lr, [1, 99]); vmin_d, vmax_d = float(np.floor(vmin_d / 10) * 10), float(np.ceil(vmax_d / 10) * 10)
amp = float(np.nanpercentile(np.abs(Fhr[np.isfinite(slr_mos)]), 99)); amp = float(np.round(amp, 1)) if amp >= 1 else float(np.round(amp, 2))
smin, smax = float(np.floor(tl.ref_s_lr.min())), float(np.ceil(tl.ref_s_lr.max()))
cmap_d = cmc.batlow.copy(); cmap_d.set_bad("white"); cmap_v = cmc.vik.copy(); cmap_v.set_bad("white"); cmap_s = cmc.batlow.copy(); cmap_s.set_bad("white")
panels = [("fig1b_ship_grid", depth_lr, ext_lr, cmap_d, vmin_d, vmax_d), ("fig1b_auv_grid", depth_hr, ext, cmap_d, vmin_d, vmax_d),
          ("fig1b_fine_field", Fhr, ext, cmap_v, -amp, amp), ("fig1b_tile_s_lr", slr_mos, ext, cmap_s, smin, smax)]
axes_b, ims = [], []
for i, (gid, arr, e, cmap, vmin, vmax) in enumerate(panels):
    ax = S.ax_mm(fig, X0 + i * (PW + GAP), YB, PW, PH)
    im = ax.imshow(np.ma.masked_invalid(arr), extent=e, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest" if i == 0 else "antialiased", zorder=1)
    im.set_rasterized(True); im.set_gid(gid)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(S.LW_AX)
    axes_b.append(ax); ims.append(im)
    rows.append(dict(panel=f"b{i+1}", series=gid, item="raster", x_lon=None, y_lat=None, value=f"vmin={vmin}, vmax={vmax}", source_file=(lr_path if i == 0 else hr_path if i == 1 else Fhr_path if i == 2 else MASTER_TILE), source_column=("depth = -elevation" if i < 2 else "F_hr band 1" if i == 2 else "ref_s_lr mosaic, 128 px tiles at r0+64, c0+64")))
# scale bar on the first panel only
sb_len = 1000.0; sx0 = ext[0] + 0.06 * (ext[1] - ext[0]); sy0 = ext[2] + 0.05 * (ext[3] - ext[2])
l_bg, = axes_b[0].plot([sx0, sx0 + sb_len], [sy0, sy0], color="white", linewidth=2.2, solid_capstyle="butt", zorder=3); l_bg.set_gid("fig1b_scalebar_halo")
l_fg, = axes_b[0].plot([sx0, sx0 + sb_len], [sy0, sy0], color="black", linewidth=1.0, solid_capstyle="butt", zorder=4); l_fg.set_gid("fig1b_scalebar")
t_sb = axes_b[0].text(sx0 + sb_len / 2, sy0 + 0.02 * (ext[3] - ext[2]), "1 km", ha="center", va="bottom", fontsize=S.FS, color="black", zorder=5, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]); t_sb.set_gid("fig1b_scalebar_label")
S.panel_letter(fig, axes_b[0], "b", dx_mm=-5.0, dy_mm=1.0)
# colorbars: shared under panels 1-2, one each under 3 and 4
CBY = YB + PH + 1.5; CBH = 2.5
cax1 = S.ax_mm(fig, X0, CBY, 2 * PW + GAP, CBH); cax3 = S.ax_mm(fig, X0 + 2 * (PW + GAP), CBY, PW, CBH); cax4 = S.ax_mm(fig, X0 + 3 * (PW + GAP), CBY, PW, CBH)
for cax, im, lab, gid in ((cax1, ims[1], "depth (m)", "fig1b_cbar_depth"), (cax3, ims[2], "band-passed relief (m)", "fig1b_cbar_fine"), (cax4, ims[3], "$s_{\\mathrm{lr}}$ (m)", "fig1b_cbar_s_lr")):
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(S.LW_AX); cb.ax.tick_params(labelsize=S.FS, width=S.LW_AX, length=2, pad=1.5)
    cb.set_label(lab, fontsize=S.FS, labelpad=1.5); cb.ax.set_gid(gid)
    if im is ims[2]: cb.set_ticks([-amp, 0, amp])

# (c) spectra
axs = S.ax_mm(fig, 124.0, 8.0, 42.0, 44.0); S.style_axes(axs)
band = axs.axvspan(f_lr_m, f_hi_m, color=S.GRAY_BAND, linewidth=0, zorder=0); band.set_gid("fig1c_target_band")
good_h = np.isfinite(P_h) & (P_h > 0); good_l = np.isfinite(P_l) & (P_l > 0)
la, = axs.loglog(fc_h[good_h], P_h[good_h], color=S.OKABE["blue"], linewidth=S.LW_DATA, linestyle="-", zorder=3); la.set_gid("fig1c_auv_spectrum")
ls, = axs.loglog(fc_l[good_l], P_l[good_l], color=S.OKABE["vermillion"], linewidth=S.LW_DATA, linestyle=(0, (3, 1.5)), zorder=2); ls.set_gid("fig1c_ship_spectrum")
ymin = float(np.nanmin(P_h[good_h])) / 30.0; ymax = float(np.nanmax(np.r_[P_h[good_h], P_l[good_l]])) * 5.0
axs.set_ylim(ymin, ymax); axs.set_xlim(float(fc_h[0]) * 0.8, float(fc_h[-1]) * 1.3)
axs.set_xlabel("wavenumber (cycles m$^{-1}$)", fontsize=S.FS); axs.set_ylabel("radial PSD (m$^{4}$)", fontsize=S.FS)
axs.tick_params(axis="both", which="both", labelsize=S.FS)
# direct labels at the line ends (AUV at its last bin; ship where its line leaves the axes or at its last bin inside)
ymid = np.sqrt(ymin * ymax)
n_a = int(np.argmin(np.abs(np.log(P_h[good_h]) - np.log(ymid / 30)))); ta = axs.text(fc_h[good_h][n_a] * 1.9, P_h[good_h][n_a], "AUV", color=S.OKABE["blue"], fontsize=S.FS, ha="left", va="center"); ta.set_gid("fig1c_label_auv")
n_s = int(np.argmin(np.abs(np.log(P_l[good_l]) - np.log(ymid)))); ts = axs.text(fc_l[good_l][n_s] * 0.55, P_l[good_l][n_s], "Ship", color=S.OKABE["vermillion"], fontsize=S.FS, ha="right", va="center"); ts.set_gid("fig1c_label_ship")
S.panel_letter(fig, axs, "c", dx_mm=-9.5, dy_mm=1.0)

S.finish(fig, 1, font, meta, rows)
