"""R03 §2 — decomposition of σ_m's skill into s_lr and σ_norm (native 7 units, R02 per-tile outputs, no inference)."""
import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C, r03_metrics as M
OUT = f"{C.ROOT}/s2"; os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(f"{C.R02}/eval/pertile.parquet"); U7 = list(C.NATIVE7)
df["sigma_norm"] = df.sigma_m / df.s_lr          # exact: sigma_m = mean over scored px of sigma_norm * s_lr, s_lr constant per tile
res = {}; md = []
for mode in ("mosaic", "tile"):
    for tgt in ("T_ship", "T_cnn"):
        d = df[(df["mode"] == mode) & df.unit.isin(U7)].reset_index(drop=True); g = d.unit.values; y = d[tgt].values
        P = {"log s_lr": d.s_lr.values, "log sigma_norm": d.sigma_norm.values, "log sigma_m (R02)": d.sigma_m.values,
             "roughness-3 (R02)": M.nested_linear(d, ["C", "curv", "aniso"], tgt, U7, g),
             "s_lr + roughness-3": M.nested_linear(d, ["C", "curv", "aniso", "s_lr"], tgt, U7, g)}
        # within-unit partial: residual of log sigma_norm on log s_lr, residual of log T on log s_lr (per-unit OLS)
        ls, lsn, lt = np.log10(d.s_lr.values), np.log10(d.sigma_norm.values), np.log10(y + M.EPS); eS = np.zeros(len(d)); eT = np.zeros(len(d)); partial_unit = {}
        for u in U7:
            s = g == u; A = np.c_[ls[s], np.ones(s.sum())]
            bS, *_ = np.linalg.lstsq(A, lsn[s], rcond=None); bT, *_ = np.linalg.lstsq(A, lt[s], rcond=None)
            eS[s] = lsn[s] - A @ bS; eT[s] = lt[s] - A @ bT; partial_unit[u] = float(np.corrcoef(eS[s], eT[s])[0, 1])
        pooled = float(np.corrcoef(eS, eT)[0, 1])
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            idx = rng.integers(0, 7, 7); sel = np.concatenate([np.where(g == U7[i])[0] for i in idx]); bs.append(np.corrcoef(eS[sel], eT[sel])[0, 1])
        S, Cm = M.compare(P, y, g, U7, [("log sigma_m (R02)", "log s_lr"), ("log sigma_norm", "log s_lr"), ("s_lr + roughness-3", "roughness-3 (R02)"), ("log sigma_m (R02)", "s_lr + roughness-3")])
        frac = S["log s_lr"]["within_r2"] / max(S["log sigma_m (R02)"]["within_r2"], 1e-9)
        res[f"{mode}/{tgt}"] = dict(scores=S, comparisons=Cm, partial_sigma_norm_given_s_lr=dict(pooled_r=pooled, pooled_r2=pooled ** 2, ci95_r=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], per_unit_r=partial_unit),
                                    frac_of_sigma_m_r2_from_s_lr=frac, n=int(len(d)))
        md += [f"\n### §2 {tgt} / {mode} ({len(d)} tiles, 7 native units)\n", M.md_pred_table(S, U7), "\nDifferences (bootstrap over units, 2,000 resamples):\n", M.md_cmp_table(Cm),
               f"\nFraction of σ_m's within-unit R² available from log s_lr alone: **{frac:.3f}** ({S['log s_lr']['within_r2']:.3f} / {S['log sigma_m (R02)']['within_r2']:.3f}). "
               f"Within-unit partial correlation of log σ_norm with log {tgt} given log s_lr: pooled r = **{pooled:+.3f}** (partial R² {pooled**2:.3f}), 95% CI over units [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]; per unit: " + ", ".join(f"{u} {partial_unit[u]:+.2f}" for u in U7) + "."]
        print(f"[{mode}/{tgt}] R2 s_lr={S['log s_lr']['within_r2']:.3f} sigma_norm={S['log sigma_norm']['within_r2']:.3f} sigma_m={S['log sigma_m (R02)']['within_r2']:.3f} rough3={S['roughness-3 (R02)']['within_r2']:.3f} s_lr+rough3={S['s_lr + roughness-3']['within_r2']:.3f} partial r={pooled:+.3f}", flush=True)
json.dump(res, open(f"{OUT}/s2_results.json", "w"), indent=1, default=float); open(f"{OUT}/s2_tables.md", "w").write("\n".join(md)); print("wrote", OUT)
