"""R03 §5 — where σ beats roughness and where it loses (NA090, AT37-13, RR1506; T-ship, mosaic): tile selection,
GeoPackage of footprints with attributes, per-tile GeoTIFFs (harmonized CRS), and feature-mean tables. No interpretation."""
import sys, os, json, glob, numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.transform import Affine; from shapely.geometry import box
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C
OUT = f"{C.ROOT}/s5"; os.makedirs(OUT, exist_ok=True); UNITS = ["NA090", "AT37-13", "RR1506"]
pt = pd.read_parquet(f"{C.R02}/eval/pertile.parquet"); pt = pt[pt["mode"] == "mosaic"].set_index("key")
s3 = pd.read_parquet(f"{C.ROOT}/s3/s3_pertile.parquet").set_index("key"); Fz = pd.read_parquet(f"{C.ROOT}/s4/features_all.parquet").set_index("key")
featcols = [c for c in Fz.columns if c.endswith(("_a", "_d", "_a_hrgrid")) and Fz[c].dtype.kind == "f"] + ["s_lr", "dist_edge", "frac_valid", "depth"]
summary = []; sel_all = []
for unit in UNITS:
    d = pt[pt.unit == unit].copy(); n = len(d); d["rank_T"] = d.T_ship.rank(); d["rank_sigma"] = d.sigma_m.rank(); d["rank_C"] = d.C.rank()
    d["dev_sigma"] = (d.rank_sigma - d.rank_T).abs() / n; d["dev_C"] = (d.rank_C - d.rank_T).abs() / n
    k15 = max(1, int(round(0.15 * n))); worstC = d.nlargest(k15, "dev_C"); sig_wins = worstC.nsmallest(15, "dev_sigma"); worstS = d.nlargest(k15, "dev_sigma"); rough_wins = worstS.nsmallest(15, "dev_C")
    T = dict(np.load(f"{C.R02}/mosaics/{unit}/tiles_mosaic.npz", allow_pickle=True)); mos = np.load(f"{C.R02}/mosaics/{unit}/mosaic.npz", allow_pickle=True)
    tr = Affine.from_gdal(*mos["transform"].tolist()); crs = str(mos["crs"]); kidx = {str(k): i for i, k in enumerate(T["keys"])}
    recs = []; od = os.path.join(OUT, unit); os.makedirs(od, exist_ok=True)
    for grp, sel in (("sigma_wins", sig_wins), ("roughness_wins", rough_wins)):
        for key, r in sel.iterrows():
            r0 = int(key.split("_r")[-1][:6]); c0 = int(key.split("_c")[-1][:6]); i = kidx[key]; ttr = tr * Affine.translation(c0, r0)
            hr_off = T["hr_m"][i] - float(T["delta"][i]); lu = T["lr_up_m"][i]; sg = T["sigma_m"][i]; j = T["joint"][i].astype(bool)
            for nm, arr in (("hr_minus_delta", hr_off), ("ship_bicubic", lu), ("sigma_m", sg), ("abs_err_bicubic", np.abs(hr_off - lu))):
                with rasterio.open(os.path.join(od, f"{grp}__{key[:-4]}__{nm}.tif"), "w", driver="GTiff", height=256, width=256, count=1, dtype="float32", crs=crs, transform=ttr, nodata=np.nan, compress="deflate") as dst:
                    dst.write(np.where(j, arr, np.nan).astype(np.float32), 1)
            x0, y0 = ttr * (0, 0); x1, y1 = ttr * (256, 256); s = s3.loc[key] if key in s3.index else None; f = Fz.loc[key] if key in Fz.index else None
            recs.append(dict(key=key, unit=unit, group=grp, rank_T=float(r.rank_T), rank_sigma=float(r.rank_sigma), rank_C=float(r.rank_C), T_ship=float(r.T_ship), sigma_m=float(r.sigma_m), roughness_C=float(r.C), s_lr=float(r.s_lr),
                             T_res=float(s.T_res) if s is not None else np.nan, T_fine=float(s.T_fine) if s is not None else np.nan, A_fine=float(s.A_fine) if s is not None else np.nan,
                             shat_fine_a=float(f.shat_fine_a) if f is not None else np.nan, shat_fine_d=float(f.shat_fine_d) if f is not None else np.nan, geometry=box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))
    g = gpd.GeoDataFrame(recs, crs=crs); g.to_file(os.path.join(OUT, "s5_cases.gpkg"), layer=unit, driver="GPKG"); sel_all.append(g.drop(columns="geometry"))
    fm = Fz[Fz.index.isin(g.key)].join(g.set_index("key")["group"])[featcols + ["group"]].groupby("group").mean(numeric_only=True).T
    fm.columns = [f"{unit}: {c}" for c in fm.columns]; summary.append(fm)
    print(unit, "n", n, "selected", len(g), flush=True)
pd.concat(sel_all).to_csv(f"{OUT}/s5_selected_tiles.csv", index=False); S = pd.concat(summary, axis=1); S.to_csv(f"{OUT}/s5_feature_means.csv")
md = ["| feature | " + " | ".join(S.columns) + " |", "|---|" + "---|" * len(S.columns)] + [f"| {i} | " + " | ".join(f"{v:.3g}" for v in row) + " |" for i, row in S.iterrows()]
open(f"{OUT}/s5_feature_means.md", "w").write("\n".join(md)); print("wrote", OUT)
