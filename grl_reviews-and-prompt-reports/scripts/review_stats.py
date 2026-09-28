"""GRL review assessment (2026-09-28): statistics the reviewer asked for, computed on frozen conventions.

Part 1 — fine-band transfer decomposition (reviewer asks 1 & 2).
  Exact dir44 task3 leakage-unit protocol (t3.collect / same standardization / same LOUO fit), native 8 units
  and expanded 15 units (dir55 boundary_suite unit maps). For every fold we add: Pearson r, Spearman rho,
  R2 after removing the per-unit mean of prediction and observation, R2 after removing mean and scale (= r^2),
  level bias and scale ratio. Computed in the paper's normalized units and in meters (x std_m).
Part 2 — anatomy of the 'total error' target (reviewer asks 3 & 4), from the dir55 repro per-pixel maps
  (het_production_v1_repro; NOT the production model, whose per-pixel maps were never persisted).
"""
from __future__ import annotations
import glob, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "/scratch/users/scdobbs/step0_bakeoff")
import step0_bakeoff as b0
sys.path.insert(0, "/scratch/users/scdobbs/dir44")
import task3_covariates as t3

OUT = Path("/scratch/users/scdobbs/grl_review"); OUT.mkdir(exist_ok=True)
PATCHES = t3.PATCHES
EPS = 1e-9
NATIVE_UNITS = {
    "NA090": ["NA090__MGDS_31212"], "NA080": ["NA080__MGDS_31290"], "TN399": ["TN399__MGDS_30373"],
    "RR1506": ["RR1506__MGDS_29779"], "AT37-13": ["AT37-13__MGDS_31199"], "EW0207": ["EW0207__MGDS_32556"],
    "FK181031": ["FK181031__MGDS_24618"], "TN299": ["TN299__MGDS_27339", "TN299__MGDS_31253"],
}
NEW_UNITS = {
    "lu02_abyssal": ["discol_so242_1"], "lu06_nodule": ["ccz_so268_1"], "lu07_volcanic": ["EW9801__MGDS_31425"],
    "lu09_vent": ["tag_m127"], "lu11_margin": ["NA076__MGDS_32317"], "lu14_margin": ["AR26__MGDS_31838"],
    "lu18_arctic": ["2009_Amundsen__MGDS_30046"],
}


# ---------------- Part 1 ----------------
def collect_with_std(pairs, splits=("train", "val", "test")):
    """t3.collect logic verbatim, plus std_m (stats[1]) so the target can be expressed in meters."""
    files = []
    for sp in splits:
        files += sorted(glob.glob(os.path.join(PATCHES, sp, "*.npz")))
    by = {}
    for pid in pairs:
        fs = [f for f in files if b0.pid_of(f) == pid]
        if len(fs) > t3.FIT_CAP:
            fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, t3.FIT_CAP).round().astype(int))]
        base, y, std = [], [], []
        for f in fs:
            d = np.load(f, allow_pickle=True)
            pb = b0.patch_bands(d)
            ft = b0.features_and_target(pb)
            if ft is None:
                continue
            base.append([np.log10(ft["C"] + EPS), ft["curv"], ft["aniso"]])
            y.append(ft["yfine"]); std.append(pb["std_m"])
        if len(y) >= 8:
            by[pid] = dict(base=np.array(base), y=np.array(y), std=np.array(std))
    return by


def merge_units(by_all, units):
    by_unit = {}
    for u, ps in units.items():
        recs = [by_all[p] for p in ps if p in by_all]
        if recs:
            by_unit[u] = dict(base=np.vstack([r["base"] for r in recs]),
                              y=np.concatenate([r["y"] for r in recs]),
                              std=np.concatenate([r["std"] for r in recs]))
    return by_unit


def design(rec, mode, meters):
    b = rec["base"].copy()
    if meters:  # C and curvature scale with std_m; anisotropy is unitless
        b[:, 0] = b[:, 0] + np.log10(rec["std"])
        b[:, 1] = b[:, 1] * rec["std"]
    return b if mode == "base" else b[:, :1]


def target(rec, meters):
    y = rec["y"] * (rec["std"] if meters else 1.0)
    return np.log10(y + EPS)


def fold_metrics(yte, p):
    yb, pb_ = yte.mean(), p.mean()
    st = np.sum((yte - yb) ** 2)
    r2_raw = 1 - np.sum((yte - p) ** 2) / st
    r2_dm = 1 - np.sum((yte - yb - (p - pb_)) ** 2) / st
    r = pearsonr(p, yte)[0]; rho = spearmanr(p, yte)[0]
    slope = np.sum((p - pb_) * (yte - yb)) / max(np.sum((p - pb_) ** 2), EPS)
    return dict(n=int(len(yte)), r2_raw=r2_raw, r2_demeaned=r2_dm, r2_demeaned_rescaled=r * r * np.sign(r),
                pearson=r, spearman=rho, level_bias_log10=pb_ - yb, scale_ratio=p.std() / yte.std(),
                ols_slope_obs_on_pred=slope)


def louo(by_unit, mode, meters):
    units = list(by_unit)
    Xall = np.vstack([design(by_unit[u], mode, meters) for u in units])
    mu = Xall.mean(0); sd = Xall.std(0) + EPS          # frozen convention (t3.louo): global standardization
    per, P, T = {}, [], []
    for held in units:
        tr = [u for u in units if u != held]
        Xtr = np.vstack([design(by_unit[u], mode, meters) for u in tr])
        ytr = np.concatenate([target(by_unit[u], meters) for u in tr])
        A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
        beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        Xte = design(by_unit[held], mode, meters); yte = target(by_unit[held], meters)
        p = np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ beta
        per[held] = fold_metrics(yte, p); P.append(p); T.append(yte)
    P = np.concatenate(P); T = np.concatenate(T)
    pooled = 1 - np.sum((T - P) ** 2) / np.sum((T - T.mean()) ** 2)
    keys = [k for k in next(iter(per.values())) if k != "n"]
    summ = {k: dict(mean=float(np.mean([per[u][k] for u in units])),
                    median=float(np.median([per[u][k] for u in units])),
                    n_positive=int(sum(per[u][k] > 0 for u in units))) for k in keys}
    return dict(n_units=len(units), pooled_r2=float(pooled), per_fold=per, summary=summ)


def part1():
    out = {}
    t0 = time.time()
    by_all = collect_with_std([p for v in {**NATIVE_UNITS, **NEW_UNITS}.values() for p in v])
    print(f"collected {len(by_all)} pairs in {time.time()-t0:.0f}s: " +
          ", ".join(f"{p}={len(by_all[p]['y'])}" for p in by_all), flush=True)
    for tag, units in [("native8", NATIVE_UNITS), ("expanded15", {**NATIVE_UNITS, **NEW_UNITS})]:
        bu = merge_units(by_all, units)
        for mode in ("base", "logC"):
            for meters in (False, True):
                key = f"{tag}/{mode}/{'meters' if meters else 'normalized'}"
                res = louo(bu, mode, meters); out[key] = res
                s = res["summary"]
                print(f"[{key}] N={res['n_units']} pooled={res['pooled_r2']:.3f} | per-fold mean/median: "
                      f"R2raw {s['r2_raw']['mean']:.2f}/{s['r2_raw']['median']:.2f}  "
                      f"R2demeaned {s['r2_demeaned']['mean']:.2f}/{s['r2_demeaned']['median']:.2f}  "
                      f"r {s['pearson']['mean']:.2f}/{s['pearson']['median']:.2f} ({s['pearson']['n_positive']}/{res['n_units']}>0)  "
                      f"rho {s['spearman']['mean']:.2f}/{s['spearman']['median']:.2f} ({s['spearman']['n_positive']}/{res['n_units']}>0)  "
                      f"scale_ratio med {s['scale_ratio']['median']:.2f}", flush=True)
    # per-unit spread of the target, normalized vs meters (reviewer: orders of magnitude?)
    bu = merge_units(by_all, {**NATIVE_UNITS, **NEW_UNITS})
    out["target_spread"] = {u: dict(median_yfine_norm=float(np.median(bu[u]["y"])),
                                    median_yfine_m=float(np.median(bu[u]["y"] * bu[u]["std"])),
                                    median_std_m=float(np.median(bu[u]["std"]))) for u in bu}
    return out


# ---------------- Part 2 ----------------
def patch_index():
    idx = {}
    for sp in ("train", "val", "test"):
        for f in glob.glob(os.path.join(PATCHES, sp, "*.npz")):
            idx[os.path.basename(f)[:-4]] = f
    return idx


def band_share(err, m, f_lr, f_hi):
    """Fraction of detrended error variance below f_lr/2, in [f_lr/2,f_lr], and in [f_lr,f_hi]."""
    tot = np.mean(b0._detrend(err, m)[m] ** 2) + EPS
    lo = b0.band_rms(err, m, 1e-6, f_lr / 2) ** 2 / tot
    ny = b0.band_rms(err, m, f_lr / 2, f_lr) ** 2 / tot
    fi = b0.band_rms(err, m, f_lr, f_hi) ** 2 / tot
    return lo, ny, fi


def part2():
    idx = patch_index(); rows = []
    for unit in NATIVE_UNITS:
        pp = np.load(f"/scratch/users/scdobbs/dir55/repro/het_folds/perpixel_{unit}.npz", allow_pickle=True)
        sig, mu, pid, row, col = pp["sigma"].astype(np.float32), pp["mean"].astype(np.float32), pp["pid"], pp["row"], pp["col"]
        for i in range(len(pid)):
            key = f"{pid[i]}_r{int(row[i]):06d}_c{int(col[i]):06d}"
            if key not in idx:
                continue
            d = np.load(idx[key], allow_pickle=True); pb = b0.patch_bands(d); m = pb["valid"]
            if m.sum() < 64 or pb["f_hi"] <= pb["f_lr"]:
                continue
            hr, lu = pb["hr"], pb["lr_up"]
            err = hr - mu[i]; e = np.abs(err[m]); s = sig[i][m]
            bic = (hr - lu)[m]
            e2 = e ** 2; order = np.argsort(e2)[::-1]; tot = e2.sum() + EPS
            n1 = max(1, int(0.01 * len(e))); n5 = max(1, int(0.05 * len(e)))
            # distance of the worst-1% pixels to the nearest invalid pixel / patch edge
            padded = np.pad(m, 1, constant_values=False)
            dist = distance_transform_edt(padded)[1:-1, 1:-1][m]
            lo, ny, fi = band_share(err, m, pb["f_lr"], pb["f_hi"])
            rows.append(dict(unit=unit, pid=str(pid[i]), row=int(row[i]), col=int(col[i]), n_valid=int(m.sum()),
                             std_m=pb["std_m"], mean_sigma=float(s.mean()),
                             resid_rms=float(np.sqrt(e2.mean())), resid_median=float(np.median(e)),
                             resid_rms_clip5=float(np.sqrt(np.mean(np.minimum(e, 5.0) ** 2))),
                             share_top1pct=float(e2[order[:n1]].sum() / tot), share_top5pct=float(e2[order[:n5]].sum() / tot),
                             frac_gt5=float(np.mean(e > 5)), frac_gt2=float(np.mean(e > 2)),
                             cov90=float(np.mean(e <= 1.6449 * s)),
                             dist_top1_median=float(np.median(dist[order[:n1]])), dist_all_median=float(np.median(dist)),
                             frac_top1_within8px=float(np.mean(dist[order[:n1]] <= 8)),
                             bic_rms=float(np.sqrt(np.mean(bic ** 2))), bic_offset=float(bic.mean()),
                             bic_rms_dm=float(bic.std()), mean_vs_lrup_rms=float(np.sqrt(np.mean((mu[i] - lu)[m] ** 2))),
                             err_share_lowband=lo, err_share_nyqband=ny, err_share_fineband=fi,
                             hr_fine=b0.band_rms(hr, m, pb["f_lr"], pb["f_hi"]), C=b0.coarse_features(lu, m, pb["f_lr"])[0]))
        print(f"  {unit}: {sum(r['unit']==unit for r in rows)} patches", flush=True)
    df = pd.DataFrame(rows); df.to_parquet(OUT / "part2_error_anatomy_perpatch.parquet")

    def within_r2(x, y, g):
        xd = x - x.groupby(g).transform("mean"); yd = y - y.groupby(g).transform("mean")
        b = np.sum(xd * yd) / np.sum(xd * xd); return float(1 - np.sum((yd - b * xd) ** 2) / np.sum(yd ** 2))
    ls = np.log10(df.mean_sigma); g = df.unit
    summ = dict(n_patches=int(len(df)),
                medians=df.drop(columns=["unit", "pid"]).median(numeric_only=True).to_dict(),
                per_unit_medians=df.groupby("unit").median(numeric_only=True).to_dict(orient="index"),
                within_unit_r2_logsigma_vs={k: within_r2(ls, np.log10(df[k] + EPS), g) for k in
                    ["resid_rms", "resid_median", "resid_rms_clip5", "bic_rms", "bic_rms_dm", "hr_fine", "mean_vs_lrup_rms"]},
                within_unit_r2_logC_vs={k: within_r2(np.log10(df.C + EPS), np.log10(df[k] + EPS), g) for k in
                    ["resid_rms", "resid_median", "bic_rms", "bic_rms_dm"]},
                per_unit_spearman_logsigma_vs={u: {k: float(spearmanr(np.log10(s.mean_sigma), np.log10(s[k] + EPS))[0])
                    for k in ["resid_rms", "resid_median", "resid_rms_clip5", "bic_rms", "bic_rms_dm", "share_top1pct"]}
                    for u, s in df.groupby("unit")})
    print(json.dumps({k: summ[k] for k in ["medians", "within_unit_r2_logsigma_vs", "within_unit_r2_logC_vs"]}, indent=1, default=float))
    return summ


if __name__ == "__main__":
    res = {"part1_transfer_decomposition": part1()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    res["part2_error_anatomy"] = part2()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT / "results.json")
