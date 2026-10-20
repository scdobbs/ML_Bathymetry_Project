"""R03 §3 — what T-ship is made of: band split of the residual (HR − δ − bicubic) on the R02 mosaic re-cut tiles,
native 7 units + EW0207 (separately). Filter = R02 T32B convention exactly: plane detrend on valid px, masked px set to
0, hard FFT annulus mask, RMS on valid px (b0.bandpass / band_rms). No taper (R02 used none)."""
import sys, json, os, glob, numpy as np, pandas as pd
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C, r03_metrics as M
sys.path.insert(0, "/scratch/users/scdobbs/step0_bakeoff"); import step0_bakeoff as b0
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R02"); import het_clean as hc
from scipy.stats import spearmanr
OUT = f"{C.ROOT}/s3"; os.makedirs(OUT, exist_ok=True); U7 = list(C.NATIVE7); UNITS = {**C.NATIVE7, **C.EW0207}
pt = pd.read_parquet(f"{C.R02}/eval/pertile.parquet"); pt = pt[pt["mode"] == "mosaic"].set_index("key")
rows = []
for unit, pids in UNITS.items():
    for pid in pids:
        suf = f"_{pid}" if len(pids) > 1 else ""; T = dict(np.load(f"{C.R02}/mosaics/{unit}/tiles_mosaic{suf}.npz", allow_pickle=True))
        for i, key in enumerate(T["keys"]):
            key = str(key)
            if key not in pt.index: continue
            r = pt.loc[key]; d = np.load(glob.glob(f"{C.R02}/patches_clean/*/{key}")[0], allow_pickle=True); st = np.asarray(d["stats"]).ravel()
            k, lr_native, hr_res = float(st[5]), float(st[6]), float(st[7]); f_lr = hr_res / lr_native; f_hi = min(k * f_lr, 0.45)
            hr = T["hr_m"][i]; lu = T["lr_up_m"][i]; mu = T["mu_m"][i]; j = T["joint"][i].astype(bool) & np.isfinite(mu); i16 = T["interior16"][i].astype(bool) & j
            hr_off = hr - float(T["delta"][i]); resid = np.where(j, hr_off - lu, 0.0)
            def bands(m):
                if m.sum() < 64: return [np.nan] * 7
                T2 = float(np.mean(resid[m] ** 2)); tf = b0.band_rms(resid, m, f_lr, f_hi); th = b0.band_rms(resid, m, f_hi, 1.0); tlp = b0.band_rms(resid, m, 0.0, f_lr)
                tres = float(np.sqrt(max(T2 - tf ** 2 - th ** 2, 0.0)))     # everything not in [f_lr, Nyq]: plane + resolved band (sums in quadrature to T-ship)
                af = b0.band_rms(hr_off, m, f_lr, f_hi); bf = b0.band_rms(lu, m, f_lr, f_hi)
                return [np.sqrt(T2), tres, tlp, tf, th, af, bf]
            Tt, Tres, Tres_lp, Tfine, Thi, Afine, Bfine = bands(j); Tt16, Tres16, Tres_lp16, Tfine16, Thi16, Afine16, Bfine16 = bands(i16)
            rows.append(dict(unit=unit, pid=pid, key=key, f_lr=f_lr, f_hi=f_hi, k=k, frac_joint=float(j.mean()), frac_i16=float(i16.mean()),
                             T_ship=Tt, T_res=Tres, T_res_lp=Tres_lp, T_fine=Tfine, T_hi=Thi, A_fine=Afine, B_fine=Bfine, E_noise=float(np.sqrt(max(Bfine ** 2 - Afine ** 2, 0.0))),
                             T_ship_i16=Tt16, T_res_i16=Tres16, T_fine_i16=Tfine16, A_fine_i16=Afine16,
                             sigma_m=r.sigma_m, s_lr=r.s_lr, C=r.C, curv=r.curv, aniso=r.aniso, slope=r.slope, dist_edge=r.dist_edge, frac_valid=r.frac_valid, depth=r.depth, lr_native=r.lr_native, T_ship_R02=r.T_ship))
    print(unit, len(rows), flush=True)
df = pd.DataFrame(rows); df.to_parquet(f"{OUT}/s3_pertile.parquet"); res = {}; md = []
# 1. band shares per unit
md.append("### §3.1 Band shares of T-ship² per unit (medians over tiles; mosaic tiles, joint-valid px)\n\n| unit | n | median T_res²/T² (plane+resolved) | median T_res_lp²/T² (detrended low-pass only) | median T_fine²/T² | median T_hi²/T² (above f_hi) | median A_fine (m) | median T_fine (m) | median B_fine/A_fine | tiles frac_joint<0.9 | Spearman(T_res, T_fine) | Spearman(T_res, A_fine) | Spearman(T_fine, A_fine) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for u in U7 + ["EW0207"]:
    d = df[df.unit == u]; T2 = d.T_ship ** 2
    rr = dict(n=len(d), res=float((d.T_res ** 2 / T2).median()), res_lp=float((d.T_res_lp ** 2 / T2).median()), fine=float((d.T_fine ** 2 / T2).median()), hi=float((d.T_hi ** 2 / T2).median()),
              A_fine=float(d.A_fine.median()), T_fine=float(d.T_fine.median()), BA=float((d.B_fine / d.A_fine).median()), low_valid=float((d.frac_joint < 0.9).mean()),
              rho_res_fine=float(spearmanr(d.T_res, d.T_fine)[0]), rho_res_A=float(spearmanr(d.T_res, d.A_fine)[0]), rho_fine_A=float(spearmanr(d.T_fine, d.A_fine)[0]))
    res[f"bands/{u}"] = rr
    md.append(f"| {u} | {rr['n']} | {rr['res']:.3f} | {rr['res_lp']:.3f} | {rr['fine']:.3f} | {rr['hi']:.3f} | {rr['A_fine']:.3f} | {rr['T_fine']:.3f} | {rr['BA']:.2f} | {rr['low_valid']:.2f} | {rr['rho_res_fine']:+.2f} | {rr['rho_res_A']:+.2f} | {rr['rho_fine_A']:+.2f} |")
# 2. predictors vs T_res, T_fine, A_fine (native 7, mosaic)
d7 = df[df.unit.isin(U7)].reset_index(drop=True); g = d7.unit.values
for tgt in ("T_ship", "T_res", "T_fine", "A_fine", "T_res_i16", "T_fine_i16", "A_fine_i16"):
    dd = d7.dropna(subset=[tgt]).reset_index(drop=True); gg = dd.unit.values; y = dd[tgt].values
    P = {"sigma_m": dd.sigma_m.values, "roughness-3": M.nested_linear(dd, ["C", "curv", "aniso"], tgt, U7, gg), "free combined GBT": M.nested_gbt(dd, ["C", "curv", "aniso", "slope", "dist_edge", "frac_valid", "depth", "lr_native"], tgt, U7, gg), "s_lr alone": dd.s_lr.values}
    S, Cm = M.compare(P, y, gg, U7, [("sigma_m", "roughness-3"), ("sigma_m", "free combined GBT"), ("sigma_m", "s_lr alone"), ("roughness-3", "s_lr alone")])
    res[f"pred/{tgt}"] = dict(scores=S, comparisons=Cm, n=int(len(dd))); md += [f"\n### §3.2 Predictors of {tgt} (native 7 units, mosaic, {len(dd)} tiles)\n", M.md_pred_table(S, U7), "", M.md_cmp_table(Cm)]
    print(f"[{tgt}] " + " ".join(f"{k}={v['within_r2']:.3f}" for k, v in S.items()), flush=True)
# 3. which band does each predictor track (per unit rho with T_res vs T_fine)
md.append("\n### §3.3 Which band each predictor tracks: per-unit Spearman ρ with T_res / T_fine / A_fine\n\n| unit | σ_m: ρ(T_res) / ρ(T_fine) / ρ(A_fine) | roughness C: ρ(T_res) / ρ(T_fine) / ρ(A_fine) | s_lr: ρ(T_res) / ρ(T_fine) / ρ(A_fine) | ρ(σ_m, E_noise) / ρ(C, E_noise) | ρ(σ_m, A_fine \\| E_noise) / ρ(C, A_fine \\| E_noise) (partial, log) |\n|---|---|---|---|---|---|")
def partial(a, b, c):
    la, lb, lc = (np.log10(np.asarray(v) + M.EPS) for v in (a, b, c)); A = np.c_[lc, np.ones(len(lc))]
    ra = la - A @ np.linalg.lstsq(A, la, rcond=None)[0]; rb = lb - A @ np.linalg.lstsq(A, lb, rcond=None)[0]; return float(spearmanr(ra, rb)[0])
for u in U7 + ["EW0207"]:
    d = df[df.unit == u]; f = lambda p, t: f"{spearmanr(d[p], d[t])[0]:+.2f}"
    rr = dict(sigma=[f("sigma_m", t) for t in ("T_res", "T_fine", "A_fine")], C=[f("C", t) for t in ("T_res", "T_fine", "A_fine")], s_lr=[f("s_lr", t) for t in ("T_res", "T_fine", "A_fine")],
              noise=[f("sigma_m", "E_noise"), f("C", "E_noise")], partialA=[f"{partial(d.sigma_m, d.A_fine, d.E_noise + 1e-3):+.2f}", f"{partial(d.C, d.A_fine, d.E_noise + 1e-3):+.2f}"])
    res[f"track/{u}"] = rr; md.append(f"| {u} | {' / '.join(rr['sigma'])} | {' / '.join(rr['C'])} | {' / '.join(rr['s_lr'])} | {' / '.join(rr['noise'])} | {' / '.join(rr['partialA'])} |")
json.dump(res, open(f"{OUT}/s3_results.json", "w"), indent=1, default=float); open(f"{OUT}/s3_tables.md", "w").write("\n".join(md)); print("wrote", OUT)
