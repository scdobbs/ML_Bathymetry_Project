"""Follow-up: are the worst-1% error pixels at the PATCH BORDER (CNN padding effect) or at INTERIOR invalid pixels
(ship-coverage holes / AUV holes)? Also: how much of the per-patch RMS error survives when a border rim and a
mask-edge rim are excluded, and does sigma's within-unit skill survive that exclusion?"""
import glob, os, sys, json
import numpy as np, pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion
sys.path.insert(0, "/scratch/users/scdobbs/step0_bakeoff"); import step0_bakeoff as b0
P = "/scratch/groups/hilley/bathy_diffusion_2026/patches_native_phase2"
idx = {os.path.basename(f)[:-4]: f for sp in ("train","val","test") for f in glob.glob(os.path.join(P, sp, "*.npz"))}
UNITS = ["NA090","NA080","TN399","RR1506","AT37-13","EW0207","FK181031","TN299"]
rows = []
for unit in UNITS:
    pp = np.load(f"/scratch/users/scdobbs/dir55/repro/het_folds/perpixel_{unit}.npz", allow_pickle=True)
    for i in range(len(pp["pid"])):
        key = f"{pp['pid'][i]}_r{int(pp['row'][i]):06d}_c{int(pp['col'][i]):06d}"
        if key not in idx: continue
        d = np.load(idx[key], allow_pickle=True); pb = b0.patch_bands(d); m = pb["valid"]
        if m.sum() < 64: continue
        err = np.abs(pb["hr"] - pp["mean"][i].astype(np.float32)); sig = pp["sigma"][i].astype(np.float32)
        H, W = m.shape
        yy, xx = np.mgrid[:H, :W]; d_border = np.minimum(np.minimum(yy, H-1-yy), np.minimum(xx, W-1-xx))
        d_hole = distance_transform_edt(m) if (~m).any() else np.full(m.shape, 1e9)   # distance to interior invalid
        e = err[m]; n1 = max(1, int(0.01*len(e))); top = np.argsort(e)[::-1][:n1]
        db, dh = d_border[m][top], d_hole[m][top]
        interior = m & (d_border > 8) & (d_hole > 8)
        rows.append(dict(unit=unit, n_valid=int(m.sum()), has_hole=bool((~m).any()), frac_interior=float(interior.mean()),
                         top1_near_border=float(np.mean(db <= 8)), top1_near_hole=float(np.mean(dh <= 8)),
                         top1_near_either=float(np.mean((db <= 8) | (dh <= 8))),
                         resid_rms=float(np.sqrt(np.mean(e**2))),
                         resid_rms_interior=float(np.sqrt(np.mean(err[interior]**2))) if interior.sum() > 64 else np.nan,
                         sigma_mean=float(sig[m].mean()), sigma_mean_interior=float(sig[interior].mean()) if interior.sum() > 64 else np.nan,
                         bic_rms_dm_interior=float((pb["hr"]-pb["lr_up"])[interior].std()) if interior.sum() > 64 else np.nan,
                         C=b0.coarse_features(pb["lr_up"], m, pb["f_lr"])[0]))
df = pd.DataFrame(rows); df.to_parquet("/scratch/users/scdobbs/grl_review/edge_followup_perpatch.parquet")
def within_r2(x, y, g):
    ok = np.isfinite(x) & np.isfinite(y); x, y, g = x[ok], y[ok], g[ok]
    xd = x - x.groupby(g).transform("mean"); yd = y - y.groupby(g).transform("mean")
    b = np.sum(xd*yd)/np.sum(xd*xd); return float(1 - np.sum((yd-b*xd)**2)/np.sum(yd**2))
out = dict(n=len(df), frac_patches_with_hole=float(df.has_hole.mean()),
           medians=df.median(numeric_only=True).to_dict(),
           within_r2=dict(sigma_vs_resid=within_r2(np.log10(df.sigma_mean), np.log10(df.resid_rms), df.unit),
                          sigma_int_vs_resid_int=within_r2(np.log10(df.sigma_mean_interior), np.log10(df.resid_rms_interior), df.unit),
                          sigma_int_vs_bic_dm_int=within_r2(np.log10(df.sigma_mean_interior), np.log10(df.bic_rms_dm_interior), df.unit),
                          C_vs_resid_int=within_r2(np.log10(df.C), np.log10(df.resid_rms_interior), df.unit),
                          C_vs_bic_dm_int=within_r2(np.log10(df.C), np.log10(df.bic_rms_dm_interior), df.unit)))
json.dump(out, open("/scratch/users/scdobbs/grl_review/edge_followup.json", "w"), indent=1)
print(json.dumps(out, indent=1))
