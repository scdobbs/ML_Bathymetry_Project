"""R02 §3.7 — GeoTIFF mosaics for NA090, TN299 (both pairs), RR1506 and a Fig.-4-style hit/miss PNG on T-ship
for sigma and for the free combined baseline. CPU."""
import json, os, sys, glob
import numpy as np, pandas as pd, rasterio
from rasterio.transform import Affine
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R02"); import het_clean as hc
sys.path.insert(0, "/scratch/users/scdobbs/dir44"); import task5_metadata as t5
MOS = "/scratch/users/scdobbs/grl_review/R02/mosaics"; EV = "/scratch/users/scdobbs/grl_review/R02/eval"; OUT = "/scratch/users/scdobbs/grl_review/R02/visual_qa"
os.makedirs(OUT, exist_ok=True)
DATA = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy"
UNITS = {"NA090": ["NA090__MGDS_31212"], "TN299": ["TN299__MGDS_27339", "TN299__MGDS_31253"], "RR1506": ["RR1506__MGDS_29779"]}


def write_tif(path, arr, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan, compress="DEFLATE", tiled=True, BIGTIFF="IF_SAFER") as dst:
        dst.write(arr.astype(np.float32), 1)


def geotiffs():
    from preprocessing.colocated_utils import load_manifest
    pairs = load_manifest(DATA)
    for unit, pids in UNITS.items():
        for pid in pids:
            suffix = f"_{pid}" if len(pids) > 1 else ""
            m = dict(np.load(os.path.join(MOS, unit, f"mosaic{suffix}.npz"), allow_pickle=True))
            tr = Affine.from_gdal(*m["transform"].tolist()); crs = str(m["crs"]); j = m["joint"].astype(bool)
            row = pairs[pairs.pair_id == pid].iloc[0]
            with rasterio.open(os.path.join(DATA, row["harmonized_path_hr"])) as s: hr = s.read(1, masked=True).filled(np.nan).astype(np.float32)
            hr_off = hr - m["delta_m"]
            od = os.path.join(OUT, f"{unit}{suffix}"); os.makedirs(od, exist_ok=True)
            write_tif(os.path.join(od, "mu_m.tif"), np.where(j, m["mu_m"], np.nan), tr, crs)
            write_tif(os.path.join(od, "sigma_m.tif"), np.where(j, m["sigma_m"], np.nan), tr, crs)
            write_tif(os.path.join(od, "abs_err_cnn_m.tif"), np.where(j, np.abs(hr_off - m["mu_m"]), np.nan), tr, crs)
            write_tif(os.path.join(od, "abs_err_bicubic_m.tif"), np.where(j, np.abs(hr_off - m["lr_up_m"]), np.nan), tr, crs)
            print("  wrote", od, flush=True)


def hitmiss():
    df = pd.read_parquet(os.path.join(EV, "pertile.parquet")); df = df[df["mode"] == "mosaic"]
    res = json.load(open(os.path.join(EV, "results.json")))
    # rebuild the free-combined nested-LOUO GBT prediction on T_ship for the plotted units
    units7 = list(hc.FOLDS); d = df[df.unit.isin(units7)].reset_index(drop=True); cols = ["C", "curv", "aniso", "slope", "dist_edge", "frac_valid", "depth", "lr_native"]
    pred = np.full(len(d), np.nan)
    for u in units7:
        tr = d.unit != u; te = d.unit == u
        pred[te.values] = 10 ** t5.gbt_fit_predict(d.loc[tr, cols].values.astype(float), np.log10(d.loc[tr, "T_ship"].values + 1e-9), d.loc[te, cols].values.astype(float))
    d["free_combined"] = pred
    fig, axes = plt.subplots(len(UNITS), 2, figsize=(10, 4.2 * len(UNITS)))
    for i, unit in enumerate(UNITS):
        s = d[d.unit == unit].copy(); s["r0"] = s.key.str.extract(r"_r(\d{6})").astype(int); s["c0"] = s.key.str.extract(r"_c(\d{6})").astype(int)
        k = max(1, int(round(0.2 * len(s)))); true_top = set(s.sort_values("T_ship", ascending=False).key[:k])
        for jx, (name, col) in enumerate([("model σ", "sigma_m"), ("free combined (roughness+metadata GBT)", "free_combined")]):
            pred_top = set(s.sort_values(col, ascending=False).key[:k]); ax = axes[i, jx]
            for _, r in s.iterrows():
                c = "#1b9e77" if (r.key in true_top and r.key in pred_top) else ("#d95f02" if r.key in true_top else ("#7570b3" if r.key in pred_top else "#dddddd"))
                ax.add_patch(plt.Rectangle((r.c0, -r.r0), 256, 256, color=c, lw=0))
            ax.set_xlim(s.c0.min(), s.c0.max() + 256); ax.set_ylim(-s.r0.max() - 256, -s.r0.min()); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            hit = len(true_top & pred_top) / k; rho = s[[col, "T_ship"]].corr(method="spearman").iloc[0, 1]
            ax.set_title(f"{unit} — {name}\nhit {hit:.2f}  ρ {rho:+.2f}  (green hit, orange miss, purple false alarm)", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "hitmiss_Tship.png"), dpi=110); print("wrote hitmiss", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project")
    geotiffs(); hitmiss()
