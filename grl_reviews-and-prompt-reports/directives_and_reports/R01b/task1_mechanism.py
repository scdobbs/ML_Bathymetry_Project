"""R01b Task 1 — mechanism (R01 B2–B5) on the reproduction model's per-pixel maps and weights. CPU only."""
from __future__ import annotations
import glob, os, sys, json, time
import numpy as np, pandas as pd, torch
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, "/scratch/users/scdobbs/step0_bakeoff"); import step0_bakeoff as b0
sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project/models/het_production_v1/step0_bakeoff"); import het_louo as hl
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
P = "/scratch/users/scdobbs/grl_review/R01/patches_restored/patches_native_phase2"
W = "/scratch/users/scdobbs/dir55/repro/het_folds"
OUT = "/scratch/users/scdobbs/grl_review/R01b/task1"; os.makedirs(OUT, exist_ok=True)
UNITS = ["NA090","NA080","TN399","RR1506","AT37-13","EW0207","FK181031","TN299"]
idx = {os.path.basename(f)[:-4]: f for sp in ("train","val","test") for f in glob.glob(os.path.join(P, sp, "*.npz"))}
def key(pid, r, c): return f"{pid}_r{int(r):06d}_c{int(c):06d}"

# ---------------- B2: effective receptive field by gradient probing ----------------
def b2():
    rng = np.random.default_rng(0); rows = []; prof_all = []
    per_unit = {u: np.load(f"{W}/perpixel_{u}.npz", allow_pickle=True) for u in UNITS}
    picks = []
    for u in UNITS:
        pp = per_unit[u]; n = len(pp["pid"])
        for i in rng.choice(n, size=3, replace=False): picks.append((u, i))
    picks = picks[:20]
    H = 256; yy, xx = np.mgrid[:H, :H]; rr = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    for u, i in picks:
        pp = per_unit[u]; f = idx[key(pp["pid"][i], pp["row"][i], pp["col"][i])]
        d = np.load(f, allow_pickle=True)
        model = hl.HeteroscedasticUNetNative(base_ch=32); model.load_state_dict(torch.load(f"{W}/ckpt_{u}.pt", map_location="cpu")); model.eval()
        x = torch.from_numpy(d["lr_up"]).float()[None].requires_grad_(True)
        s = torch.from_numpy(np.atleast_1d(d["scale"]).astype(np.float32))[None]
        mean, logvar = model(x, s)
        corr = mean - x                      # learned correction head_mean(h); mean = x + head
        for name, out in (("mu_minus_x", corr), ("logvar", logvar)):
            if x.grad is not None: x.grad.zero_()
            out[0, 0, 128, 128].backward(retain_graph=True)
            g = x.grad[0, 0].abs().numpy(); tot = g.sum() + 1e-30
            order = np.argsort(rr.ravel()); cum = np.cumsum(g.ravel()[order]) / tot; rs = rr.ravel()[order]
            r95 = float(rs[np.searchsorted(cum, 0.95)]); r99 = float(rs[np.searchsorted(cum, 0.99)])
            within = {R: float(g[rr <= R].sum() / tot) for R in (8, 16, 32, 48, 64, 96)}
            rows.append(dict(unit=u, i=int(i), quantity=name, r95=r95, r99=r99, **{f"mass_within_{R}": v for R, v in within.items()},
                             frac_mass_beyond_64=1 - within[64]))
        del model
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/B2_receptive_field.csv", index=False)
    summ = df.groupby("quantity")[["r95", "r99", "mass_within_8", "mass_within_16", "mass_within_32", "mass_within_64", "frac_mass_beyond_64"]].agg(["median", "max"]).round(3)
    print("B2\n", summ.to_string(), flush=True)
    return dict(table=df.to_dict(orient="records"), summary=json.loads(summ.to_json()))

# ---------------- B3/B4/B5 on per-pixel maps ----------------
def b345():
    rows = []; prof = {u: {} for u in UNITS}; pix_rows = []
    yy, xx = np.mgrid[:256, :256]; dborder = np.minimum(np.minimum(yy, 255 - yy), np.minimum(xx, 255 - xx))
    for u in UNITS:
        pp = np.load(f"{W}/perpixel_{u}.npz", allow_pickle=True)
        accum = {}  # distance -> sums for B4/B5
        for i in range(len(pp["pid"])):
            k = key(pp["pid"][i], pp["row"][i], pp["col"][i])
            if k not in idx: continue
            d = np.load(idx[k], allow_pickle=True); pb = b0.patch_bands(d); m = pb["valid"]
            if m.sum() < 64: continue
            hr, lu = pb["hr"], pb["lr_up"]; mu = pp["mean"][i].astype(np.float32); sg = pp["sigma"][i].astype(np.float32)
            err = np.abs(hr - mu); st = np.asarray(d["stats"]).ravel(); std_m = float(st[1])
            lv = d["lr_valid"][0].astype(bool); L = lv.shape[0]
            ci = (np.arange(256) * L // 256); lv_hr = lv[ci][:, ci]
            step = m & binary_dilation(~lv_hr, iterations=max(1, int(np.ceil(256 / L)))) if (~lv).any() else np.zeros_like(m)
            dhole = distance_transform_edt(m) if (~m).any() else np.full(m.shape, 999.0)
            off = float(np.median((hr - lu)[m]))
            e = err[m]; n1 = max(1, int(0.01 * len(e))); top = np.zeros(m.sum(), bool); top[np.argsort(e)[::-1][:n1]] = True
            absin = np.abs(lu)[m]; db = dborder[m]; dh = dhole[m]; stp = step[m]; sgm = sg[m]
            rows.append(dict(unit=u, key=k, std_m=std_m, abs_off_norm=abs(off), max_abs_in=float(absin.max()),
                             frac_mu_gt5=float((np.abs(mu[m]) > 5).mean()), frac_err_gt5=float((e > 5).mean()),
                             top1_dborder_med=float(np.median(db[top])), all_dborder_med=float(np.median(db)),
                             top1_dhole_med=float(np.median(dh[top])), all_dhole_med=float(np.median(dh)),
                             top1_absin_med=float(np.median(absin[top])), all_absin_med=float(np.median(absin)),
                             top1_in_step=float(stp[top].mean()), all_in_step=float(stp.mean()),
                             top1_within8_border=float((db[top] <= 8).mean()), top1_within8_hole=float((dh[top] <= 8).mean()),
                             sigma_clamp_hi=float((sgm >= 20.0).mean()), sigma_clamp_lo=float((sgm <= 0.0068).mean()),
                             sigma_med=float(np.median(sgm)), sigma_p99=float(np.quantile(sgm, 0.99)),
                             resid_rms=float(np.sqrt(np.mean(e ** 2)))))
            # B5 partial Spearman within patch, rim (<=32) vs interior (>64)
            for tag, sel in (("rim32", db <= 32), ("interior64", db > 64)):
                if sel.sum() > 200:
                    R = np.c_[rankdata(absin[sel]), rankdata(db[sel]), np.ones(sel.sum())]
                    re = rankdata(e[sel]); rs = rankdata(sgm[sel])
                    be, *_ = np.linalg.lstsq(R, re, rcond=None); bs, *_ = np.linalg.lstsq(R, rs, rcond=None)
                    rows[-1][f"partial_rho_{tag}"] = float(np.corrcoef(re - R @ be, rs - R @ bs)[0, 1])
                    rows[-1][f"rho_{tag}"] = float(spearmanr(e[sel], sgm[sel])[0])
            # B4/B5 profiles
            for dd in range(0, 65):
                s2 = m & (dborder == dd)
                if s2.any():
                    a = accum.setdefault(dd, [0, 0.0, [], 0.0]); a[0] += int(s2.sum()); a[1] += float(err[s2].sum()); a[2].append(np.median(err[s2])); a[3] += float(sg[s2].sum())
        prof[u] = {dd: dict(n=a[0], mean_err=a[1] / a[0], median_err_of_patchmedians=float(np.median(a[2])), mean_sigma=a[3] / a[0]) for dd, a in accum.items()}
        print(f"  {u}: {sum(r['unit']==u for r in rows)} patches", flush=True)
    df = pd.DataFrame(rows); df.to_parquet(f"{OUT}/B3_perpatch.parquet")
    json.dump(prof, open(f"{OUT}/B4_profiles.json", "w"), indent=1)
    # B3 summaries
    d7 = df[df.unit != "EW0207"]
    def summ(x):
        return dict(n=int(len(x)), top1_dborder_med=float(x.top1_dborder_med.median()), all_dborder_med=float(x.all_dborder_med.median()),
                    top1_within8_border=float(x.top1_within8_border.median()), top1_within8_hole=float(x.top1_within8_hole.median()),
                    top1_dhole_med=float(x.top1_dhole_med.median()), top1_absin_med=float(x.top1_absin_med.median()), all_absin_med=float(x.all_absin_med.median()),
                    top1_in_step=float(x.top1_in_step.mean()), all_in_step=float(x.all_in_step.mean()),
                    frac_mu_gt5_med=float(x.frac_mu_gt5.median()), frac_mu_gt5_mean=float(x.frac_mu_gt5.mean()),
                    sigma_clamp_hi_mean=float(x.sigma_clamp_hi.mean()), sigma_clamp_lo_mean=float(x.sigma_clamp_lo.mean()),
                    partial_rho_rim32_med=float(x.partial_rho_rim32.median()), partial_rho_interior64_med=float(x.partial_rho_interior64.median()),
                    rho_rim32_med=float(x.rho_rim32.median()), rho_interior64_med=float(x.rho_interior64.median()))
    out = dict(all8=summ(df), excl_EW0207=summ(d7), per_unit={u: summ(df[df.unit == u]) for u in UNITS})
    # deciles
    for col in ("std_m", "max_abs_in", "abs_off_norm"):
        dec = pd.qcut(df[col], 10, labels=False, duplicates="drop")
        out[f"frac_mu_gt5_by_decile_{col}"] = df.groupby(dec).agg(lo=(col, "min"), hi=(col, "max"), frac_mu_gt5_mean=("frac_mu_gt5", "mean"), frac_err_gt5_mean=("frac_err_gt5", "mean"), n=("key", "size")).round(4).to_dict(orient="index")
    # spearman across patches of frac_mu_gt5 with the three factors
    out["patch_level_spearman_frac_mu_gt5"] = {c: float(spearmanr(df[c], df.frac_mu_gt5)[0]) for c in ("std_m", "max_abs_in", "abs_off_norm")}
    json.dump(out, open(f"{OUT}/B3_B5_summary.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("all8", "excl_EW0207", "patch_level_spearman_frac_mu_gt5")}, indent=1), flush=True)
    return out

if __name__ == "__main__":
    t0 = time.time(); r2 = b2(); print(f"B2 done {time.time()-t0:.0f}s", flush=True)
    r = b345(); print(f"B3-5 done {time.time()-t0:.0f}s", flush=True)
