"""R03 §6.2 — one-pair pilot: regrid TN399 (smallest NCEI raw volume, 0.17 GB / 10 swath files) with mbgrid -M at
lr_native = 45.8 m in the MB-System 5.8.2beta06 Apptainer sandbox; compare the mean surface with the R02 ship grid;
correlate tile-median SD with T-ship / T_fine / A_fine next to roughness C. FEASIBILITY NUMBERS, NOT EVIDENCE."""
import sys, os, glob, gzip, json, math, time, subprocess, shutil, numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.enums import Resampling as RS
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C
sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project"); from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import load_manifest, resolve_pair_paths, reproject_from_array
from scipy.stats import spearmanr
PID = "TN399__MGDS_30373"; CRUISE = "TN399"; SANDBOX = "/home/groups/hilley/containers/mbsystem_sandbox"
OUT = f"{C.ROOT}/s6/pilot"; RAW = f"{OUT}/raw"; WD = os.path.join(os.environ.get("L_SCRATCH", "/tmp"), "r03_pilot"); os.makedirs(WD, exist_ok=True)
BIND = ["--bind", f"/scratch,/home/groups,/oak,{WD}"]
def mb(args, **kw):
    r = subprocess.run(["apptainer", "exec", *BIND, SANDBOX, *args], capture_output=True, text=True, cwd=WD, **kw); return r
T0 = time.time(); timing = {}
m = load_manifest(C.DATA); row = m[m.pair_id == PID].iloc[0]; p = resolve_pair_paths(C.DATA, row); lr_native = float(row.lr_native_res_m)
# 1. decompress + datalist (as stage_c_full.prepare_cruise; format 58 from the filename)
files = sorted(glob.glob(f"{RAW}/*.mb58.gz")); dl = os.path.join(WD, "datalist.mb-1")
with open(dl, "w") as f:
    for gz in files:
        out = os.path.join(WD, os.path.basename(gz)[:-3])
        if not os.path.exists(out):
            with gzip.open(gz, "rb") as s, open(out, "wb") as d: shutil.copyfileobj(s, d, 1 << 22)
        f.write(f"{os.path.basename(out)} 58\n")
timing["decompress_s"] = time.time() - T0; print(f"{len(files)} swath files, {sum(os.path.getsize(g) for g in files)/1e6:.0f} MB gz", flush=True)
# 2. region = harmonized lr.tif bounds + 1 km, in lon/lat; cell = lr_native (m) in degrees (pipeline convention)
with rasterio.open(p["lr"]) as s: lr_arr = s.read(1).astype(np.float32); lr_nd = s.nodata; lr_tr = s.transform; lr_crs = s.crs; b = s.bounds
w, so, e, n = transform_bounds(lr_crs, "EPSG:4326", *b); mlat = (so + n) / 2; dlon = 1 / (111320 * max(0.2, math.cos(math.radians(mlat)))); dlat = 1 / 111320
R = f"{w-1000*dlon}/{e+1000*dlon}/{so-1000*dlat}/{n+1000*dlat}"; dx, dy = lr_native * dlon, lr_native * dlat
t1 = time.time(); r = mb(["mbgrid", "-I", "datalist.mb-1", "-O", "pilot", "-R", R, "-E", f"{dx}/{dy}/degrees", "-A2", "-G3", "-F1", "-C0", "-M"]); timing["mbgrid_s"] = time.time() - t1
open(f"{OUT}/mbgrid_stdout.txt", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr); print("mbgrid rc", r.returncode, "in", round(timing["mbgrid_s"]), "s", flush=True)
grds = {k: os.path.join(WD, f"pilot{suf}.grd") for k, suf in (("mean", ""), ("num", "_num"), ("sd", "_sd"))}
for k, g in grds.items():
    if os.path.exists(g): shutil.copy(g, f"{OUT}/{os.path.basename(g)}")
    else: print("MISSING", g, flush=True)
def read_grd(path):
    for cand in (path, f"NETCDF:{path}"):
        try:
            with rasterio.open(cand) as ds: return ds.read(1, masked=True).filled(np.nan).astype(np.float64), ds.transform, ds.crs
        except Exception: continue
    raise RuntimeError(path)
mean, gtr, gcrs = read_grd(grds["mean"]); sd, _, _ = read_grd(grds["sd"]); num, _, _ = read_grd(grds["num"]); gcrs = gcrs or "EPSG:4326"
print("grid", mean.shape, "crs", gcrs, "valid frac", np.isfinite(mean).mean().round(3), flush=True)
# 3. reproject onto the harmonized lr.tif grid; RMS difference on the common valid area
def onto_lr(a, rs):
    dst = np.full(lr_arr.shape, np.nan, np.float32)
    reproject(source=a.astype(np.float32), destination=dst, src_transform=gtr, src_crs=gcrs, dst_transform=lr_tr, dst_crs=lr_crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=rs); return dst
mean_lr = onto_lr(mean, RS.bilinear); sd_lr = onto_lr(sd, RS.nearest); num_lr = onto_lr(num, RS.nearest)
ship = np.where(lr_arr == lr_nd, np.nan, lr_arr); ok = np.isfinite(ship) & np.isfinite(mean_lr); diff = mean_lr[ok] - ship[ok]
idx = pd.read_parquet(f"{C.R02}/patches_clean/patch_index.parquet"); s_lr_med = float(idx[idx.pair_id == PID].s_lr.median())
cmp = dict(n_common=int(ok.sum()), frac_ship_valid_covered=float(ok.sum() / np.isfinite(ship).sum()), rms_diff_m=float(np.sqrt(np.mean(diff ** 2))), median_diff_m=float(np.median(diff)), mad_m=float(np.median(np.abs(diff - np.median(diff)))),
           rms_after_median_offset_m=float(np.sqrt(np.mean((diff - np.median(diff)) ** 2))), s_lr_median_m=s_lr_med, rms_over_s_lr=float(np.sqrt(np.mean((diff - np.median(diff)) ** 2)) / s_lr_med),
           sd_grid_median_m=float(np.nanmedian(sd_lr[ok])), num_grid_median=float(np.nanmedian(num_lr[ok])), cell_m_harmonized=float(abs(lr_tr.a)), cell_m_regrid=lr_native)
print(cmp, flush=True)
# 4. tile-median SD on the R02 evaluation tiles of TN399 (same 200 tiles as R02/§3), vs T-ship, T_fine, A_fine and roughness C
s3 = pd.read_parquet(f"{C.ROOT}/s3/s3_pertile.parquet"); s3 = s3[s3.pid == PID]
g = ecp.pair_setup(C.DATA, row, 256); recs = []
for _, t in s3.iterrows():
    r0 = int(t.key.split("_r")[-1][:6]); c0 = int(t.key.split("_c")[-1][:6]); bnd = rasterio.windows.bounds(rasterio.windows.Window(c0, r0, 256, 256), g["transform"])
    sdt = reproject_from_array(sd_lr, lr_tr, lr_crs, bnd, g["lr_px"], g["dst_crs"], RS.average); nt = reproject_from_array(num_lr, lr_tr, lr_crs, bnd, g["lr_px"], g["dst_crs"], RS.average)
    v = np.isfinite(sdt)
    recs.append(dict(key=t.key, sd_median=float(np.nanmedian(sdt)) if v.any() else np.nan, sd_mean=float(np.nanmean(sdt)) if v.any() else np.nan, num_median=float(np.nanmedian(nt)) if np.isfinite(nt).any() else np.nan,
                     T_ship=t.T_ship, T_fine=t.T_fine, A_fine=t.A_fine, T_res=t.T_res, C=t.C, sigma_m=t.sigma_m, s_lr=t.s_lr))
d = pd.DataFrame(recs).dropna(subset=["sd_median"]); d.to_parquet(f"{OUT}/pilot_tiles.parquet")
rho = {tg: {pr: float(spearmanr(d[pr], d[tg])[0]) for pr in ("sd_median", "sd_mean", "num_median", "C", "sigma_m", "s_lr")} for tg in ("T_ship", "T_res", "T_fine", "A_fine")}
timing["total_s"] = time.time() - T0
json.dump(dict(pair=PID, cruise=CRUISE, n_swath_files=len(files), gz_mb=sum(os.path.getsize(x) for x in files) / 1e6, mbgrid_cmd=f"mbgrid -I datalist.mb-1 -O pilot -R {R} -E {dx}/{dy}/degrees -A2 -G3 -F1 -C0 -M",
               mbgrid_rc=r.returncode, comparison=cmp, n_tiles=int(len(d)), spearman=rho, timing_s=timing, label="FEASIBILITY, NOT EVIDENCE"), open(f"{OUT}/pilot_results.json", "w"), indent=1)
print(json.dumps(rho, indent=1)); print("timing", timing, flush=True)
