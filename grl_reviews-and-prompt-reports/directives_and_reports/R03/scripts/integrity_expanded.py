"""R02 §1.2 — integrity of the clean patch set (as R01a Phase A) + footprint comparison with the original set."""
import sys; sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common  # lockbox guard
import glob, os, json, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
P = "/scratch/users/scdobbs/grl_review/R03/patches_clean_expanded"; R = "/scratch/users/scdobbs/grl_review/R02/patches_clean"
OUT = "/scratch/users/scdobbs/grl_review/R03/integrity"; os.makedirs(OUT, exist_ok=True)
FILLS = [-9999.0, -32768.0, -32767.0, 3.4028235e38]
def one(f):
    d = np.load(f, allow_pickle=True); x = d["x"][0]; y = d["y"][0]; hv = d["hr_valid"][0].astype(bool); lvh = d["lr_valid_hr"][0].astype(bool)
    lr = d["lr_raw_m"][0]; lv = d["lr_valid"][0].astype(bool); st = np.asarray(d["stats"]).ravel(); j = hv & lvh
    ym = y * st[1] + st[2] + st[0]   # back to meters
    return dict(file=os.path.basename(f), pid=str(d["pair_id"]), split=str(d["split"]), unit=str(d["leakage_unit"]),
                x_nan=int(np.isnan(x).sum()), x_inf=int(np.isinf(x).sum()), y_nan_valid=int(np.isnan(y[hv]).sum()), y_inf=int(np.isinf(y).sum()),
                lr_nan_valid=int(np.isnan(lr[lv]).sum()), lr_fill_valid=int(sum(np.isclose(lr[lv], v, atol=1e-3).sum() for v in FILLS)),
                ym_le_m9000_valid=int((ym[hv] <= -9000).sum()), mask_nonbinary=int(((d["hr_valid"][0] != 0) & (d["hr_valid"][0] != 1)).sum() + ((d["lr_valid_hr"][0] != 0) & (d["lr_valid_hr"][0] != 1)).sum()),
                max_abs_x=float(np.abs(x).max()), max_abs_x_joint=float(np.abs(x[j]).max()) if j.any() else np.nan, max_abs_y=float(np.abs(y[hv]).max()) if hv.any() else np.nan,
                mu_lr=float(st[0]), s_lr=float(st[1]), delta=float(st[2]), hr_std_m=float(st[4]), s_floor_hit=bool(st[8] > 0.5),
                frac_hr_valid=float(hv.mean()), frac_joint=float(j.mean()), y_mean_joint=float(y[j].mean()) if j.any() else np.nan, y_median_joint=float(np.median(y[j])) if j.any() else np.nan)
if __name__ == "__main__":
    files = sorted(glob.glob(f"{P}/*/*.npz")); orig = {os.path.basename(f) for f in glob.glob(f"{R}/*/*.npz")}
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool: recs = pool.map(one, files, chunksize=64)
    df = pd.DataFrame(recs); df.to_parquet(f"{OUT}/perpatch.parquet")
    names = set(df.file); nat_orig = {n for n in orig if any(n.startswith(p) for p in df.pid.unique())}
    q = lambda s: {k: float(v) for k, v in zip(["min", "p50", "p95", "max"], [s.min(), s.median(), s.quantile(.95), s.max()])}
    summ = dict(n_clean=len(df), n_original_same_pairs=len(nat_orig), in_both=len(names & nat_orig), only_clean=sorted(names - nat_orig)[:20], only_original=sorted(nat_orig - names)[:20],
                n_only_clean=len(names - nat_orig), n_only_original=len(nat_orig - names),
                counts_all_zero_expected={k: int(df[k].sum()) for k in ["x_nan", "x_inf", "y_nan_valid", "y_inf", "lr_nan_valid", "lr_fill_valid", "ym_le_m9000_valid", "mask_nonbinary"]},
                per_unit={u: dict(n=int((df.unit == u).sum()), max_abs_x=q(df[df.unit == u].max_abs_x), max_abs_x_joint=q(df[df.unit == u].max_abs_x_joint.dropna()), max_abs_y=q(df[df.unit == u].max_abs_y.dropna()),
                                  s_floor_hit_frac=float(df[df.unit == u].s_floor_hit.mean()), s_lr=q(df[df.unit == u].s_lr), delta_m=q(df[df.unit == u].delta),
                                  y_median_joint=q(df[df.unit == u].y_median_joint.dropna())) for u in sorted(df.unit.unique())},
                per_split=df.groupby("split").size().to_dict())
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)[:6000])
