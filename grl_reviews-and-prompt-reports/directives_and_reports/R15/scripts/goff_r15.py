"""R15 §1.1, §2, §3 (descriptive; reviewer request; not pre-registered). One process, CPU only.
§1.1 grid facts of Goff (2020) Data Set S1 read from the file header and values.
§2.1 unit location: union of the pairs' high-resolution footprints (<harmonized_path>/footprint.geojson; see note N1 in the
     record), centroid and area in a Lambert azimuthal equal-area projection centred on the unit, median depth of the frozen
     harmonized ship grid (lr.tif where lr_valid) sampled every ~DEPTH_SPACING_M, rounded to 10 m.
§2.2 Goff value: (a) bilinear at the centroid (NaN if any of the four nodes is missing); (b) median of the nodes with
     |dlon| <= 0.5 and |dlat| <= 0.5 deg of the centroid, with the count of non-missing nodes (primary).
§2.3 domain class (no computed / interpolated layer exists): computed = age at centroid not NaN and >= 1 Myr and no
     Kim & Wessel seamount centre within 2 x radius (radius = major axis / 2) of the footprint union; possibly interpolated =
     a value exists but a proxy fails; no value = no non-missing node in the neighbourhood.
§3   Spearman rho with a 2,000-resample bootstrap over units (default_rng(20261014)); must-hold first."""
from __future__ import annotations
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot   # font manager scans system font dirs BEFORE the guard wraps os.scandir (its os.walk needs a plain iterator)
import r15_common as R
import numpy as np, pandas as pd
C4, R8 = R.C4, R.R8
DEPTH_SPACING_M = 20.0

# ------------------------------------------------------------------ grids
def grid_facts():
    import netCDF4
    d = netCDF4.Dataset(R.GOFF_GRID); f = dict(file=os.path.basename(R.GOFF_GRID), sha256=R.GOFF_GRID_SHA, data_model=d.data_model, global_attrs={a: str(getattr(d, a)) for a in d.ncattrs()},
                                             dimensions={n: len(v) for n, v in d.dimensions.items()},
                                             variables={n: dict(dims=list(v.dimensions), dtype=str(v.dtype), attrs={a: str(v.getncattr(a)) for a in v.ncattrs()}) for n, v in d.variables.items()})
    x = np.asarray(d["x"][:], float); y = np.asarray(d["y"][:], float); z = np.asarray(np.ma.filled(d["z"][:], np.nan), float)
    dx = np.diff(x); dy = np.diff(y); assert np.allclose(dx, dx[0], atol=1e-9) and np.allclose(dy, dy[0], atol=1e-9)
    has_node_offset = any("node_offset" in v.ncattrs() for v in d.variables.values()) or "node_offset" in d.ncattrs()
    valid = (z != R.GOFF_NODATA) & np.isfinite(z); v = z[valid]
    f.update(node_spacing_deg=[float(dx[0]), float(dy[0])], node_spacing_arcmin=[float(dx[0] * 60), float(dy[0] * 60)], x_first_last=[float(x[0]), float(x[-1])], y_first_last=[float(y[0]), float(y[-1])],
             registration=("gridline (no node_offset attribute: GMT default; first node at -180.0 / -75.0 exactly, actual_range equals the node coordinates)" if not has_node_offset else "see node_offset"),
             longitude_columns_cover_360=bool(abs(len(x) * dx[0] - 360.0) < 1e-6), crs="none recorded (no crs / grid_mapping attribute; x / y in degrees by range and spacing: geographic lon / lat)",
             units="none recorded (z long_name 'z'; values are RMS heights in m per the SI)", n_nodes=int(z.size), n_nan=int(np.isnan(z).sum()), n_sentinel_minus999=int((z == R.GOFF_NODATA).sum()),
             n_valid=int(valid.sum()), frac_valid=float(valid.mean()), n_valid_negative=int((v < 0).sum()), valid_min=float(v.min()), valid_max=float(v.max()),
             valid_percentiles={p: float(np.percentile(v, p)) for p in (1, 5, 25, 50, 75, 95, 99)}, other_layers=[n for n in d.variables if n not in ("x", "y", "z")],
             computed_interpolated_flag=bool([n for n in d.variables if n not in ("x", "y", "z")]))
    d.close(); zz = np.where(valid, z, np.nan)
    return f, x, y, zz

def load_age():
    import netCDF4
    d = netCDF4.Dataset(R.AGE_GRID); names = list(d.variables)
    xn = "lon" if "lon" in names else "x"; yn = "lat" if "lat" in names else "y"; zn = [n for n in names if n not in (xn, yn)][0]
    x = np.asarray(d[xn][:], float); y = np.asarray(d[yn][:], float); z = np.asarray(np.ma.filled(d[zn][:].astype(float), np.nan), float)
    facts = dict(file=os.path.basename(R.AGE_GRID), sha256=R.AGE_GRID_SHA, url=R.AGE_GRID_URL, variable=zn, attrs={a: str(d[zn].getncattr(a)) for a in d[zn].ncattrs()}, shape=list(z.shape),
                 x_first_last=[float(x[0]), float(x[-1])], y_first_last=[float(y[0]), float(y[-1])], spacing_deg=float(np.diff(x)[0]), n_nan=int(np.isnan(z).sum()),
                 version="2020.1 (GTS2012 timescale, 6 arcmin, gridline); Seton et al. (2020) G3 doi:10.1029/2020GC009214 per README.txt")
    d.close()
    if y[0] > y[-1]: y = y[::-1]; z = z[::-1]
    if abs(x[-1] - x[0] - 360.0) < 1e-9: x = x[:-1]; z = z[:, :-1]          # drop the duplicated 180 column so longitude wraps cleanly
    return facts, x, y, z

def bilinear(x, y, z, lon, lat):
    """Bilinear on a gridline-registered global lon grid (wrapping); NaN if any of the four nodes is NaN or lat outside."""
    dx = x[1] - x[0]; dy = y[1] - y[0]; nx = len(x)
    fx = ((lon - x[0]) % 360.0) / dx; fy = (lat - y[0]) / dy
    if fy < 0 or fy > len(y) - 1: return float("nan"), []
    i0 = int(math.floor(fx)); j0 = min(int(math.floor(fy)), len(y) - 2); tx = fx - i0; ty = fy - j0
    i0 %= nx; i1 = (i0 + 1) % nx
    q = np.array([[z[j0, i0], z[j0, i1]], [z[j0 + 1, i0], z[j0 + 1, i1]]])
    nodes = [(float(x[i]), float(y[j]), float(z[j, i])) for j in (j0, j0 + 1) for i in (i0, i1)]
    if not np.isfinite(q).all(): return float("nan"), nodes
    return float((1 - ty) * ((1 - tx) * q[0, 0] + tx * q[0, 1]) + ty * ((1 - tx) * q[1, 0] + tx * q[1, 1])), nodes

def neighbourhood(x, y, z, lon, lat, half=R.NEIGH_DEG):
    dlon = (x - lon + 180.0) % 360.0 - 180.0; ci = np.where(np.abs(dlon) <= half + 1e-9)[0]; ri = np.where(np.abs(y - lat) <= half + 1e-9)[0]
    sub = z[np.ix_(ri, ci)]; fin = sub[np.isfinite(sub)]
    return dict(n_nodes=int(sub.size), n_valid=int(fin.size), median=(float(np.median(fin)) if fin.size else float("nan")), min=(float(fin.min()) if fin.size else float("nan")),
                max=(float(fin.max()) if fin.size else float("nan")), n_negative=int((fin < 0).sum()))

def load_kw():
    rows = []; seg = 0
    for line in open(R.KW):
        if line.startswith(">"): seg += int(line.split()[2]); continue
        if line.startswith("#") or not line.strip(): continue
        p = line.split(); rows.append(dict(lon=float(p[0]), lat=float(p[1]), azimuth=float(p[2]), major_km=float(p[3]), minor_km=float(p[4]), height_m=float(p[5]), regional_depth_m=float(p[8]), crust_age_kw=float(p[9]), id=p[10]))
    K = pd.DataFrame(rows); assert len(K) == seg == 24643, (len(K), seg); return K   # segment headers sum to 24,643; the header comment says 24,646 (file discrepancy, recorded)

def haversine_deg(lon1, lat1, lon2, lat2):
    l1, p1, l2, p2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

# ------------------------------------------------------------------ units
def footprint(m, pid):
    from shapely.geometry import shape
    row = C4.pair_row(m, pid); fp = os.path.join(C4.DATA, str(row["harmonized_path"]), "footprint.geojson"); g = json.load(open(fp))
    assert len(g["features"]) == 1, (pid, len(g["features"]))
    return dict(path=fp, sha256=R.sha256_file(fp), crs=g["crs"]["properties"]["name"], geom=shape(g["features"][0]["geometry"]), manifest_footprint_wkt=row["footprint_wkt"])

def to_crs(geom, src, dst):
    import pyproj; from shapely.ops import transform
    t = pyproj.Transformer.from_crs(src, dst, always_xy=True); return transform(t.transform, geom)

def laea(lon, lat): return f"+proj=laea +lat_0={lat:.6f} +lon_0={lon:.6f} +datum=WGS84 +units=m +no_defs"

def unit_location(m, pids):
    from shapely.ops import unary_union
    fps = {p: footprint(m, p) for p in pids}
    c0 = [to_crs(f["geom"], f["crs"], "EPSG:4326").centroid for f in fps.values()]
    lon0 = float(np.degrees(np.arctan2(np.mean([math.sin(math.radians(c.x)) for c in c0]), np.mean([math.cos(math.radians(c.x)) for c in c0])))); lat0 = float(np.mean([c.y for c in c0]))
    for _ in range(2):                                     # re-centre the projection on the union centroid
        P = laea(lon0, lat0); U = unary_union([to_crs(f["geom"], f["crs"], P) for f in fps.values()]); cg = to_crs(U.centroid, P, "EPSG:4326"); lon0, lat0 = float(cg.x), float(cg.y)
    P = laea(lon0, lat0); U = unary_union([to_crs(f["geom"], f["crs"], P) for f in fps.values()])
    per_pair = {p: dict(centroid_lon=float(c.x), centroid_lat=float(c.y), area_km2=float(to_crs(f["geom"], f["crs"], laea(c.x, c.y)).area / 1e6), footprint=f["path"], footprint_sha256=f["sha256"], crs=f["crs"],
                        manifest_footprint_wkt=f["manifest_footprint_wkt"]) for (p, f), c in zip(fps.items(), c0)}
    spread = max([haversine_deg(a["centroid_lon"], a["centroid_lat"], lon0, lat0) for a in per_pair.values()]) * 111.195
    return dict(lon=lon0, lat=lat0, area_km2=float(U.area / 1e6), proj=P, union=U, per_pair=per_pair, max_pair_centroid_to_unit_centroid_km=float(spread))

def pair_sample(m, pid, gx, gy, gz, ax_, ay, az, K):
    """Ruling H2(b) (Steve, 2026-10-13): Goff (a) / (b), crustal age and seamount proxies at ONE pair's footprint centroid
    (its own footprint, unit-centred LAEA re-centred on the pair), and the pair's class under the §2.3 rule and its sensitivity."""
    loc = unit_location(m, [pid])
    ga, _ = bilinear(gx, gy, gz, loc["lon"], loc["lat"]); nb = neighbourhood(gx, gy, gz, loc["lon"], loc["lat"])
    age, _ = bilinear(ax_, ay, az, loc["lon"], loc["lat"]); sm = seamounts(K, loc)
    fails = []
    if not np.isfinite(age): fails.append("off oceanic crust (age NaN)")
    elif age < R.AGE_MIN_MYR: fails.append(f"crustal age {age:.2f} < 1 Myr")
    fails_s = list(fails)
    if sm["within"]: fails.append(f"seamount within 2r ({', '.join(s['id'] for s in sm['within'])})")
    if sm["within_sensitivity"]: fails_s.append(f"seamount within 2 x major ({', '.join(s['id'] for s in sm['within_sensitivity'])})")
    has = nb["n_valid"] > 0
    cls = "no value" if not has else ("computed" if not fails else "possibly interpolated")
    cls_s = "no value" if not has else ("computed" if not fails_s else "possibly interpolated")
    return dict(pair=pid, centroid_lon=loc["lon"], centroid_lat=loc["lat"], area_km2=loc["area_km2"], goff_a=ga, goff_b=nb["median"], goff_b_n_valid=nb["n_valid"], goff_b_n_nodes=nb["n_nodes"], goff_b_n_negative=nb["n_negative"],
                crustal_age_myr=age, smt_nearest_id=(sm["nearest"] or {}).get("id"), smt_nearest_dist_km=(sm["nearest"] or {}).get("dist_km", np.nan), smt_nearest_radius_km=(sm["nearest"] or {}).get("radius_km", np.nan),
                smt_within_2r=len(sm["within"]), smt_within_2major=len(sm["within_sensitivity"]), domain_class=cls, class_reason="; ".join(fails), domain_class_sensitivity=cls_s, class_reason_sensitivity="; ".join(fails_s), seamounts=sm)

def aggregate_pairs(PP, key="domain_class"):
    """Unit class = worst class among the pairs that have a value (possibly interpolated > computed); no such pair -> no value."""
    v = [p for p in PP if p["goff_b_n_valid"] > 0]
    if not v: return "no value", v
    return ("possibly interpolated" if any(p[key] == "possibly interpolated" for p in v) else "computed"), v

def unit_depth(m, pids):
    import rasterio
    from rasterio.enums import Resampling
    from preprocessing.colocated_utils import resolve_pair_paths
    vals, per = [], {}
    for p in pids:
        row = C4.pair_row(m, p); pp = resolve_pair_paths(C4.DATA, row)
        with rasterio.open(pp["lr"]) as s, rasterio.open(pp["lr_valid"]) as sv:
            res = abs(s.transform.a); f = max(1, int(round(DEPTH_SPACING_M / res))); shp = (max(1, s.height // f), max(1, s.width // f))
            a = s.read(1, out_shape=shp, resampling=Resampling.nearest).astype(float); nd = s.nodata
            v = sv.read(1, out_shape=shp, resampling=Resampling.nearest).astype(bool)
        ok = v & np.isfinite(a) & ((a != nd) if nd is not None else True); w = a[ok]; vals.append(w)
        per[p] = dict(lr=pp["lr"], lr_valid=pp["lr_valid"], res_m=res, decimation=f, n=int(ok.sum()), median_elev_m=(float(np.median(w)) if w.size else float("nan")))
    allv = np.concatenate(vals); med = float(np.median(allv))
    return dict(median_elev_m=med, depth_m=float(round(-med / 10.0) * 10.0), n_samples=int(allv.size), per_pair=per)

def seamounts(K, loc):
    from shapely.geometry import Point
    dd = haversine_deg(K.lon.values, K.lat.values, loc["lon"], loc["lat"]); cand = K[dd <= R.SMT_SEARCH_DEG].copy()
    import pyproj
    t = pyproj.Transformer.from_crs("EPSG:4326", loc["proj"], always_xy=True)
    xs, ys = t.transform(cand.lon.values, cand.lat.values)
    cand["dist_km"] = [loc["union"].distance(Point(a, b)) / 1e3 for a, b in zip(xs, ys)]
    cand["radius_km"] = cand.major_km / 2.0; cand["margin_km"] = cand.dist_km - R.SMT_RADIUS_FACTOR * cand.radius_km
    cand["margin_sens_km"] = cand.dist_km - R.SMT_RADIUS_FACTOR * cand.major_km
    if not len(cand): return dict(n_candidates=0, nearest=None, within=[], within_sensitivity=[])
    n = cand.sort_values("dist_km").iloc[0]
    within = cand[cand.margin_km <= 0].sort_values("dist_km"); within_s = cand[cand.margin_sens_km <= 0].sort_values("dist_km")
    rec = lambda r: dict(id=r.id, lon=r.lon, lat=r.lat, dist_km=float(r.dist_km), radius_km=float(r.radius_km), major_km=float(r.major_km), height_m=float(r.height_m))
    return dict(n_candidates=int(len(cand)), nearest=rec(n), within=[rec(r) for r in within.itertuples()], within_sensitivity=[rec(r) for r in within_s.itertuples()],
                min_margin_km=float(cand.margin_km.min()))

# ------------------------------------------------------------------ unit values (R08 §4, R08b interior)
def unit_values():
    tiles = pd.read_parquet(R8.R05_TILES)
    U = tiles.groupby("unit").agg(A_fine_median_m=("A_fine", "median"), s_lr_median_m=("s_lr", "median"), n_tiles=("A_fine", "size"), n_pairs=("pair", "nunique"))
    ref = pd.read_csv(R.oak_verify(R.OAK_R08_LEVEL, "level_per_unit.csv")).set_index("unit")
    for c in ("A_fine_median_m", "s_lr_median_m"):
        d = np.abs(U.loc[ref.index, c].values - ref[c].values).max(); assert d <= 1e-12, (c, d)
    ti = pd.read_parquet(R.oak_verify(R.OAK_R08B_INT, "tiles_interior.parquet"))
    J = ti.merge(tiles[["tile_id", "A_fine", "s_lr"]].rename(columns={"A_fine": "A_fine_t"}), on="tile_id", how="left", validate="one_to_one")
    assert np.allclose(J.A_fine.values, J.A_fine_t.values, rtol=0, atol=1e-12, equal_nan=True), "R08b A_fine != R05 tiles A_fine"
    I = J[J.interior_D1.astype(bool)].groupby("unit").agg(A_fine_int_median_m=("A_fine", "median"), s_lr_int_median_m=("s_lr", "median"), n_int_tiles=("A_fine", "size"))
    U = U.join(I, how="left"); U["n_int_tiles"] = U.n_int_tiles.fillna(0).astype(int)
    band = {}
    for u, d in tiles.groupby("unit"):
        b = d.groupby("pair").agg(hr_res=("hr_res", "first"), f_lr=("f_lr", "first"), f_hi=("f_hi", "first"), k=("k", "first"), lr_native=("lr_native", "first"))
        b["lam_long_m"] = b.hr_res / b.f_lr; b["lam_short_m"] = b.hr_res / b.f_hi
        band[u] = dict(lam_short_m=float(b.lam_short_m.min()), lam_long_m=float(b.lam_long_m.max()), per_pair=b.reset_index().to_dict("records"))
    return U, band, dict(level_per_unit=f"{R.OAK_R08_LEVEL}/level_per_unit.csv", tiles_interior=f"{R.OAK_R08B_INT}/tiles_interior.parquet", tiles=R8.R05_TILES)

# ------------------------------------------------------------------ statistics
def set_stats(D, col_fine, col_slr, label):
    out = dict(set=label, n=int(len(D)), units=list(D.unit))
    if len(D) < 5: out["note"] = "fewer than 5 units: table only, no correlation (§3.1)"; return out
    out["goff_vs_fine"] = R.spearman_boot(D.goff_b, D[col_fine]); out["goff_vs_s_lr"] = R.spearman_boot(D.goff_b, D[col_slr])
    out["s_lr_vs_fine"] = R.spearman_boot(D[col_slr], D[col_fine]); out["partial_goff_fine_given_s_lr"] = R.partial_boot(D.goff_b, D[col_fine], D[col_slr])
    return out

def figure(T, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    COL = {"real-relief": "#0072B2", "artifact-affected": "#E69F00", "EW0207 (separate)": "#555555"}
    MK = {"computed": "o", "possibly interpolated": "s"}
    fig, ax = plt.subplots(figsize=(6.6, 5.2)); seen = set()
    P = T[np.isfinite(T.goff_b) & (T.goff_b > 0)]
    for r in P.itertuples():
        grp = r.group; cls = r.domain_class; mk = MK.get(cls, "x")
        lab = f"{grp}, {cls}"; first = lab not in seen; seen.add(lab)
        face = COL[grp] if cls == "computed" else "white"
        ax.scatter(r.goff_b, r.A_fine_median_m, s=58, marker=mk, facecolor=face, edgecolor=COL[grp], linewidth=1.6, zorder=3, label=lab if first else None)
        ax.annotate(r.unit if r.unit != R.EW_UNIT else "EW0207", (r.goff_b, r.A_fine_median_m), textcoords="offset points", xytext=(6, 3), fontsize=7.5, color="#333333")
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, FormatStrFormatter
    ax.xaxis.set_major_locator(FixedLocator([40, 60, 100, 150, 200, 300])); ax.xaxis.set_major_formatter(FormatStrFormatter("%g")); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Goff (2020) predicted abyssal hill RMS height (m)\n(median of nodes within 0.5° of each pair centroid; unit = median over pairs with a value)" if RULING else "Goff (2020) predicted abyssal hill RMS height, median of nodes within 0.5° (m)", fontsize=9)
    ax.set_ylabel("unit-median fine relief A_fine (m)")
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    miss = [u for u in T.unit if u not in set(P.unit)]
    order = [f"{g}, {c}" for g in ("real-relief", "artifact-affected", "EW0207 (separate)") for c in ("computed", "possibly interpolated")]
    h, l = ax.get_legend_handles_labels(); hl = sorted(zip(h, l), key=lambda t: order.index(t[1]))
    ax.legend([a for a, _ in hl], [b for _, b in hl], frameon=False, fontsize=7.5, loc="lower right", title="filled = computed; open = possibly interpolated", title_fontsize=7.5)
    if miss: ax.text(0.01, -0.22, "not plotted (no positive Goff value in the 0.5° neighbourhood): " + ", ".join("EW0207" if u == R.EW_UNIT else u for u in miss), transform=ax.transAxes, fontsize=7, color="#555555")
    fig.tight_layout(); fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------ main
RULING = "--ruling" in sys.argv          # Steve's rulings of 2026-10-13 (H1 a, H2 b, H3 keep); without the flag: the pre-ruling run as first reported
OUT = f"{R.ROOT}/out_ruling" if RULING else R.OUT

def must_hold_pre(U16):
    """Pre-ruling must-hold (directive seed 20261014, tolerance 0.01) and its diagnostic, as first reported (§1-§7 of the report)."""
    mh = R.spearman_boot(U16.s_lr_median_m, U16.A_fine_median_m)
    dev = [abs(mh["rho"] - R.R08_RHO), abs(mh["ci95"][0] - R.R08_CI[0]), abs(mh["ci95"][1] - R.R08_CI[1])]
    must = dict(rho=mh["rho"], ci95=mh["ci95"], n=mh["n"], r08_rho=R.R08_RHO, r08_ci95=list(R.R08_CI), abs_dev=dev, tol=R.MUST_HOLD_TOL, passed=bool(max(dev) <= R.MUST_HOLD_TOL), seed=R.SEED)
    r08seed = R8.spearman_boot(U16.s_lr_median_m, U16.A_fine_median_m, seed=R8.SEED, n_boot=R.N_BOOT); lo, hi = [], []
    for sd in range(20261000, 20261100):
        b = R8.spearman_boot(U16.s_lr_median_m, U16.A_fine_median_m, seed=sd, n_boot=R.N_BOOT); lo.append(b["ci95"][0]); hi.append(b["ci95"][1])
    q = lambda v: dict(min=min(v), p5=float(np.percentile(v, 5)), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=max(v))
    must["diagnostic"] = dict(with_r08_seed=dict(seed=R8.SEED, **r08seed), seeds="20261000..20261099 (100 seeds, 2,000 resamples each)", ci_lo=q(lo), ci_hi=q(hi),
                              frac_seeds_within_tol=float(np.mean([(abs(a - R.R08_CI[0]) <= R.MUST_HOLD_TOL) and (abs(b - R.R08_CI[1]) <= R.MUST_HOLD_TOL) for a, b in zip(lo, hi)])))
    print(f"must-hold: rho {mh['rho']:+.3f} [{mh['ci95'][0]:+.3f}, {mh['ci95'][1]:+.3f}] vs R08 +0.697 [+0.305, +0.866]; max dev {max(dev):.4f} -> {'PASS' if must['passed'] else 'FAIL'}", flush=True)
    return must

def must_hold_ruling(U16):
    """Ruling H1 (a): R08's seed 20261009; must reproduce OAK R08 level_correlations.json exactly. The 20261014 value is reported beside it."""
    r08 = json.load(open(R.oak_verify(R.OAK_R08_LEVEL, "level_correlations.json")))["correlations"]["level_raw_vs_s_lr"]
    mh = R8.spearman_boot(U16.s_lr_median_m, U16.A_fine_median_m, seed=R8.SEED, n_boot=R.N_BOOT)
    dev = [abs(mh["rho"] - r08["rho"]), abs(mh["ci95"][0] - r08["ci95"][0]), abs(mh["ci95"][1] - r08["ci95"][1])]
    must = dict(rho=mh["rho"], ci95=mh["ci95"], n=mh["n"], r08=r08, abs_dev=dev, tol="exact (<= 1e-12)", passed=bool(max(dev) <= 1e-12 and mh["n"] == r08["n"] == 16), seed=R8.SEED,
                ruling="H1 (a), Steve 2026-10-13", seed_20261014_row=R.spearman_boot(U16.s_lr_median_m, U16.A_fine_median_m))
    print(f"must-hold (ruling H1 a, seed {R8.SEED}): rho {mh['rho']:+.6f} [{mh['ci95'][0]:+.6f}, {mh['ci95'][1]:+.6f}] vs R08 {r08['rho']:+.6f} [{r08['ci95'][0]:+.6f}, {r08['ci95'][1]:+.6f}]; max dev {max(dev):.1e} -> {'PASS' if must['passed'] else 'FAIL'}", flush=True)
    return must

if __name__ == "__main__":
    t0 = time.time(); m, prov = R.start("goff_r15_ruling" if RULING else "goff_r15"); os.makedirs(OUT, exist_ok=True)
    U, band, vsrc = unit_values(); U16 = U.loc[R.UNITS16]
    must = must_hold_ruling(U16) if RULING else must_hold_pre(U16); R.dump(must, f"{OUT}/must_hold.json")
    if RULING and not must["passed"]: raise SystemExit("STOP (R15 §3 must hold, ruling H1 a): R08 §4 not reproduced exactly")
    # §1.1 grid facts
    gf, gx, gy, gz = grid_facts(); af, ax_, ay, az = load_age(); K = load_kw()
    R.dump(dict(goff=gf, age=af, kw=dict(file=os.path.basename(R.KW), sha256=R.KW_SHA, url=R.KW_URL, version="KWSMTS v0.1 (Kim & Wessel 2011, GJI 186, 615-631)", n=len(K), n_note="24,643 rows = sum of the 24 plate-segment headers; the file header comment states 24,646",
                                         radius_rule="radius = major axis / 2 (columns 4 / 5 are the major / minor axes of the basal ellipse, km)", major_km_range=[float(K.major_km.min()), float(K.major_km.max())])), f"{OUT}/grid_facts.json")
    assert 2 * K.major_km.max() / 111.195 < R.SMT_SEARCH_DEG, "seamount pre-filter too narrow"
    # §2 per unit (location, area and depth from the union of the unit's footprints in both modes)
    rows, recs, prow = [], {}, []
    for u in R.UNITS16 + [R.EW_UNIT]:
        pids = (C4.UNITS[u] if u in C4.UNITS else C4.EW0207[u])
        loc = unit_location(m, pids); dep = unit_depth(m, pids)
        grp = "EW0207 (separate)" if u == R.EW_UNIT else ("real-relief" if u in R.TEN else "artifact-affected")
        uv = U.loc[u]
        base = dict(unit=u, group=grp, n_pairs=len(pids), setting=" / ".join(dict.fromkeys(str(C4.pair_row(m, p)["terrain_class"]) for p in pids)), site=" / ".join(dict.fromkeys(str(C4.pair_row(m, p)["site_name"]) for p in pids)),
                    centroid_lon=loc["lon"], centroid_lat=loc["lat"], area_km2=loc["area_km2"], max_pair_offset_km=loc["max_pair_centroid_to_unit_centroid_km"], depth_m=dep["depth_m"],
                    A_fine_median_m=float(uv.A_fine_median_m), s_lr_median_m=float(uv.s_lr_median_m), A_fine_int_median_m=float(uv.A_fine_int_median_m) if uv.n_int_tiles else np.nan,
                    s_lr_int_median_m=float(uv.s_lr_int_median_m) if uv.n_int_tiles else np.nan, n_int_tiles=int(uv.n_int_tiles), lam_short_m=band[u]["lam_short_m"], lam_long_m=band[u]["lam_long_m"])
        if RULING:
            # H2 (b): sample every pair; unit class = worst class among the pairs that have a value; unit Goff = median of those pairs' values
            PP = [pair_sample(m, p, gx, gy, gz, ax_, ay, az, K) for p in pids]
            for p in PP: prow.append(dict(unit=u, **{k: v for k, v in p.items() if k != "seamounts"}))
            cls, val = aggregate_pairs(PP); cls_s, _ = aggregate_pairs(PP, "domain_class_sensitivity")
            ga = [p["goff_a"] for p in val if np.isfinite(p["goff_a"])]; ages = [p["crustal_age_myr"] for p in PP]
            near = min(PP, key=lambda p: (p["smt_nearest_dist_km"] if np.isfinite(p["smt_nearest_dist_km"]) else np.inf))
            r = dict(base, crustal_age_myr=(ages[0] if len(PP) == 1 else np.nan), crustal_age_pairs=";".join(f"{a:.2f}" for a in ages), oceanic_crust_pairs=f"{sum(np.isfinite(a) for a in ages)} of {len(PP)}",
                     smt_nearest_id=near["smt_nearest_id"], smt_nearest_dist_km=near["smt_nearest_dist_km"], smt_nearest_radius_km=near["smt_nearest_radius_km"],
                     domain_class=cls, class_reason=" | ".join(f"{p['pair']}: {p['class_reason']}" for p in val if p["class_reason"]) if len(PP) > 1 else (PP[0]["class_reason"]), domain_class_sensitivity=cls_s,
                     n_pairs_with_value=len(val), goff_a=(float(np.median(ga)) if ga else np.nan), goff_b=(float(np.median([p["goff_b"] for p in val])) if val else np.nan),
                     goff_b_n_valid=int(sum(p["goff_b_n_valid"] for p in val)), goff_b_n_nodes=int(sum(p["goff_b_n_nodes"] for p in PP)), goff_b_n_negative=int(sum(p["goff_b_n_negative"] for p in PP)))
            recs[u] = dict(location={k: v for k, v in loc.items() if k != "union"}, depth=dep, pairs=PP, band=band[u])
        else:
            ga, ga_nodes = bilinear(gx, gy, gz, loc["lon"], loc["lat"]); nb = neighbourhood(gx, gy, gz, loc["lon"], loc["lat"])
            age, age_nodes = bilinear(ax_, ay, az, loc["lon"], loc["lat"]); sm = seamounts(K, loc)
            fails = []
            if not np.isfinite(age): fails.append("off oceanic crust (age NaN)")
            elif age < R.AGE_MIN_MYR: fails.append(f"crustal age {age:.2f} < 1 Myr")
            if sm["within"]: fails.append(f"seamount within 2r ({', '.join(s['id'] for s in sm['within'])})")
            cls = "no value" if nb["n_valid"] == 0 else ("computed" if not fails else "possibly interpolated")
            fails_s = [f for f in fails if not f.startswith("seamount")] + ([f"seamount within 2 x major ({', '.join(s['id'] for s in sm['within_sensitivity'])})"] if sm["within_sensitivity"] else [])
            cls_s = "no value" if nb["n_valid"] == 0 else ("computed" if not fails_s else "possibly interpolated")
            pp = {p: dict(goff_b=neighbourhood(gx, gy, gz, v["centroid_lon"], v["centroid_lat"])["median"], n_valid=neighbourhood(gx, gy, gz, v["centroid_lon"], v["centroid_lat"])["n_valid"], **{k: v[k] for k in ("centroid_lon", "centroid_lat", "area_km2")}) for p, v in loc["per_pair"].items()}
            r = dict(base, crustal_age_myr=age, oceanic_crust=bool(np.isfinite(age)), smt_nearest_id=(sm["nearest"] or {}).get("id"), smt_nearest_dist_km=(sm["nearest"] or {}).get("dist_km", np.nan), smt_nearest_radius_km=(sm["nearest"] or {}).get("radius_km", np.nan),
                     smt_within_2r=len(sm["within"]), smt_within_2major=len(sm["within_sensitivity"]), domain_class=cls, class_reason="; ".join(fails) if fails else "", domain_class_sensitivity=cls_s,
                     goff_a=ga, goff_b=nb["median"], goff_b_n_valid=nb["n_valid"], goff_b_n_nodes=nb["n_nodes"], goff_b_min=nb["min"], goff_b_max=nb["max"], goff_b_n_negative=nb["n_negative"])
            recs[u] = dict(location={k: v for k, v in loc.items() if k != "union"}, depth=dep, goff_bilinear_nodes=ga_nodes, goff_neighbourhood=nb, age_bilinear_nodes=age_nodes, seamounts=sm, per_pair_goff=pp, band=band[u])
        r["ratio_goff_over_fine"] = r["goff_b"] / r["A_fine_median_m"] if np.isfinite(r["goff_b"]) else np.nan
        rows.append(r); print(f"{u}: class {r['domain_class']} (sens {r['domain_class_sensitivity']}) goff a {r['goff_a']:.1f} b {r['goff_b']:.1f} fine {r['A_fine_median_m']:.4f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/units.csv", index=False); R.dump(recs, f"{OUT}/units_record.json")
    if RULING:
        PT = pd.DataFrame(prow); PT.to_csv(f"{OUT}/pairs.csv", index=False)
        # check: single-pair units must give the pre-ruling values exactly (pair footprint = unit union)
        T0 = pd.read_csv(f"{R.OUT}/units.csv").set_index("unit"); singles = [u for u in T.unit if T.set_index("unit").loc[u, "n_pairs"] == 1]
        for c in ("goff_a", "goff_b", "goff_b_n_valid", "domain_class", "domain_class_sensitivity"):
            a = T.set_index("unit").loc[singles, c]; b = T0.loc[singles, c]
            ok = (a.astype(str) == b.astype(str)) | (pd.isna(a) & pd.isna(b)) if a.dtype == object else np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9, equal_nan=True)
            assert bool(np.all(ok)), (c, list(a), list(b))
        print(f"check: {len(singles)} single-pair units reproduce the pre-ruling Goff values and classes exactly", flush=True)
    D = T[T.unit.isin(R.UNITS16)]
    sets = dict(primary_n=int((D.domain_class == "computed").sum()), secondary_n=int(D.domain_class.isin(["computed", "possibly interpolated"]).sum()), classes=dict(zip(D.unit, D.domain_class)))
    if not must["passed"]:
        R.dump(dict(must_hold=must, set_sizes=sets, note="§3 not computed: must-hold failed (HOLD)"), f"{OUT}/stats.json"); print(f"set sizes {sets['primary_n']} / {sets['secondary_n']}; stopped before §3 ({time.time() - t0:.0f} s)"); sys.exit(0)
    # §3 comparison (16 development units; EW0207 never pooled); new statistics with seed 20261014 (R.SEED)
    prim = D[D.domain_class == "computed"]; sec = D[D.domain_class.isin(["computed", "possibly interpolated"])]
    S = dict(set_sizes=sets, primary=set_stats(prim, "A_fine_median_m", "s_lr_median_m", "computed"), secondary=set_stats(sec, "A_fine_median_m", "s_lr_median_m", "computed + possibly interpolated"))
    for nm, b in (("primary_ten_interior", prim), ("secondary_ten_interior", sec)):
        sub = b[b.unit.isin(R.TEN) & (b.n_int_tiles > 0)]
        S[nm] = set_stats(sub, "A_fine_int_median_m", "s_lr_int_median_m", f"{nm.split('_')[0]} set ∩ ten real-relief, R08b interior D1 medians (lu14 has no interior tile)")
        S[nm].pop("partial_goff_fine_given_s_lr", None)
    S["sensitivity_classes"] = dict(primary_n=int((D.domain_class_sensitivity == "computed").sum()), secondary_n=int(D.domain_class_sensitivity.isin(["computed", "possibly interpolated"]).sum()),
                                    primary_units=list(D[D.domain_class_sensitivity == "computed"].unit), changed=[u for u, a, b in zip(D.unit, D.domain_class, D.domain_class_sensitivity) if a != b])
    S["must_hold"] = must; S["seed"] = R.SEED; S["n_boot"] = R.N_BOOT; S["unit_value_sources"] = vsrc
    R.dump(S, f"{OUT}/stats.json")
    os.makedirs(R.FIGURES, exist_ok=True); figure(T, f"{R.FIGURES}/R15_fig_goff_vs_fine_relief")
    for k in ("primary", "secondary", "primary_ten_interior", "secondary_ten_interior"):
        s_ = S[k]; print(f"{k}: n {s_['n']} {s_.get('note', '')} " + ("; ".join(f"{q} {s_[q]['rho']:+.3f} [{s_[q]['ci95'][0]:+.3f}, {s_[q]['ci95'][1]:+.3f}]" for q in ("goff_vs_fine", "goff_vs_s_lr", "s_lr_vs_fine", "partial_goff_fine_given_s_lr") if q in s_)), flush=True)
    print(f"done ({time.time() - t0:.0f} s)")
