"""R04 §5 Check B — what sigma_norm learned. Native 7 units, R02 mosaic sigma (sigma_norm = sigma_m / s_lr, R03 §2
definition), the R03 T-ship tile set (s3_pertile: slope = R02 bicubic tile slope, T_res, A_fine R03 convention), joined
with the R04 A_fine where the tile lies on the stride-128 grid. Within-unit partial Spearman correlations (OLS
residualization on the log controls inside each unit, Spearman of the residuals), pooled median over units with a
bootstrap-over-units 95 % CI (2,000 resamples, seed 20261001)."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r04_common as C
OUT = f"{C.ROOT}/checks"; os.makedirs(OUT, exist_ok=True); EPS = 1e-9; N_BOOT = 2000; SEED = 20261001

def partial_rho(x, y, controls):
    L = lambda v: np.log10(np.asarray(v, float) + EPS); A = np.c_[np.ones(len(x)), *[L(c) for c in controls]]; lx, ly = L(x), L(y)
    rx = lx - A @ np.linalg.lstsq(A, lx, rcond=None)[0]; ry = ly - A @ np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(spearmanr(rx, ry)[0])

def pooled(per_unit):
    v = np.array([x for x in per_unit.values() if np.isfinite(x)]); rng = np.random.default_rng(SEED); n = len(v)
    bs = np.array([np.median(v[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    return dict(median=float(np.median(v)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_units=int(n), per_unit=per_unit)

if __name__ == "__main__":
    m = C.start("check_b"); s3 = pd.read_parquet(f"{C.R03}/s3/s3_pertile.parquet"); s3 = s3[s3.unit.isin(C.NATIVE7.values())].copy()
    s3["sigma_norm"] = s3.sigma_m / s3.s_lr; s3["r0"] = s3.key.str.extract(r"_r(\d{6})_").astype(int); s3["c0"] = s3.key.str.extract(r"_c(\d{6})").astype(int)
    t4 = pd.read_parquet(f"{C.ROOT}/tiles/tiles_v1.parquet")[["pair", "r0", "c0", "A_fine", "A_fine_Fhr_256", "T_fine"]].rename(columns={"pair": "pid", "A_fine": "A_fine_R04_tile128", "A_fine_Fhr_256": "A_fine_R04_256", "T_fine": "T_fine_R04_tile128"})
    s3 = s3.merge(t4, on=["pid", "r0", "c0"], how="left"); units = sorted(s3.unit.unique()); R = {}
    items = [("1. log sigma_norm vs log slope | log s_lr", "slope", ["s_lr"]), ("2a. log sigma_norm vs log T_res | log s_lr", "T_res", ["s_lr"]), ("2b. log sigma_norm vs log T_res | log s_lr, log slope", "T_res", ["s_lr", "slope"]),
             ("3. log sigma_norm vs log A_fine (R03 convention, 256 window) | log s_lr, log slope", "A_fine", ["s_lr", "slope"]), ("3'. log sigma_norm vs log A_fine (R04 tile 128, m) | log s_lr, log slope", "A_fine_R04_tile128", ["s_lr", "slope"]),
             ("3''. log sigma_norm vs log A_fine (R04 F_hr RMS over the 256 window) | log s_lr, log slope", "A_fine_R04_256", ["s_lr", "slope"]), ("4 (extra). log sigma_norm vs log T_fine (R04 tile 128) | log s_lr, log slope", "T_fine_R04_tile128", ["s_lr", "slope"])]
    md = ["# Check B — what sigma_norm learned (native 7 units, R02 mosaic sigma, R03 T-ship tile set)\n", "| item | pooled median partial ρ [95% CI] | n units | per unit " + " / ".join(units) + " | n rows |", "|---|---|---|---|---|"]
    for lab, tgt, ctl in items:
        per = {}; n = 0
        for u in units:
            d = s3[(s3.unit == u)].dropna(subset=[tgt] + ctl + ["sigma_norm"]); d = d[(d[tgt] > 0) & (d.sigma_norm > 0)]
            per[u] = partial_rho(d.sigma_norm, d[tgt], [d[c] for c in ctl]) if len(d) >= 10 else np.nan; n += len(d)
        r = pooled(per); r["n_rows"] = n; R[lab] = r
        md.append(f"| {lab} | {r['median']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['n_units']} | " + " / ".join(f"{per[u]:+.2f}" for u in units) + f" | {n} |")
    json.dump(R, open(f"{OUT}/check_b.json", "w"), indent=1, default=float); open(f"{OUT}/check_b.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
