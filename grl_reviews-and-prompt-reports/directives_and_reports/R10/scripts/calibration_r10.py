"""R10 §A6.3 — block-count calibration for B_min. Every development pair and EW0207 with >= 64 non-empty 4 x 4-tile blocks on
its scored view at D1 (the B4 view rule with B_min = 8; s_lr vs the view's target). For B in {4, 6, 8, 12, 16}: 200 random
subsets of B distinct blocks (a fresh numpy.random.default_rng(20261012) per pair, consumed in the order of the B values);
on each subset s_lr's block-bootstrap CI as in B4 but with 1,000 resamples (a fresh default_rng(20261009) per subset).
Per pair and pooled (mean over pairs): coverage of the pair's full-view rho, the fraction of draws with CI lower bound > 0,
the median CI width. Rule: B_min = 8 if pooled coverage at B = 8 >= 0.90; else the smaller of 12 and 16 reaching 0.90; else HOLD."""
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, phase4_score as PS, eval_v1_1 as E
BS = (4, 6, 8, 12, 16); N_DRAW = 200; N_BOOT = 1000; SEED_DRAW = 20261012; MIN_BLOCKS = 64
IN = f"{P4.ROOT['rehearsal']}/score_inputs/inputs_D1.parquet"; OUT = f"{P4.ROOT['rehearsal']}/calibration"

def one_pair(args):
    pid, t = args; t = t[np.isfinite(t.A_fine.values)]; vi = t[t.interior.values.astype(bool)]; vf = t[np.isfinite(t.A_fine_int.values)]
    if len(vi) >= P4.MIN_TILES and len(set(PS.blocks_of(vi))) >= P4.B_MIN_DEFAULT: view, tab, tgt = "interior", vi, "A_fine"
    elif len(vf) >= P4.MIN_TILES and len(set(PS.blocks_of(vf))) >= P4.B_MIN_DEFAULT: view, tab, tgt = "fallback", vf, "A_fine_int"
    else: return None
    blk = PS.blocks_of(tab); keys = sorted(set(blk)); nb = len(keys)
    if nb < MIN_BLOCKS: return dict(pair=pid, view=view, n_blocks=nb, eligible=False)
    idx = {k: np.where(np.array([b == k for b in blk]))[0] for k in keys}; pred = tab.s_lr.values.astype(float); tg = tab[tgt].values.astype(float); full = E.rho(pred, tg)
    rng = np.random.default_rng(SEED_DRAW); out = dict(pair=pid, unit=t.unit.iloc[0], view=view, n_blocks=nb, n_tiles=int(len(tab)), full_rho=full, eligible=True, by_B={})
    for B in BS:
        cov = []; low = []; wid = []
        for d in range(N_DRAW):
            sel = rng.choice(nb, B, replace=False); brng = np.random.default_rng(P4.BOOT_SEED); vals = []
            for _ in range(N_BOOT):
                pick = brng.integers(0, B, B); ii = np.concatenate([idx[keys[sel[j]]] for j in pick]); vals.append(E.rho(pred[ii], tg[ii]))
            vals = np.array(vals, float); ok = vals[np.isfinite(vals)]
            if (len(vals) - len(ok)) / N_BOOT > P4.UNDEF_MAX_FRAC or len(ok) == 0: cov.append(np.nan); low.append(np.nan); wid.append(np.nan); continue
            lo, hi = np.percentile(ok, 2.5), np.percentile(ok, 97.5); cov.append(lo <= full <= hi); low.append(lo > 0); wid.append(hi - lo)
        out["by_B"][str(B)] = dict(coverage=float(np.nanmean(cov)), frac_lower_gt0=float(np.nanmean(low)), median_width=float(np.nanmedian(wid)), n_not_estimable=int(np.isnan(cov).sum()))
    return out

if __name__ == "__main__":
    t0 = time.time(); workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)); os.makedirs(OUT, exist_ok=True)
    pairs = list(P4.C4.DEV_PAIRS) + [P4.R8.EW_PAIR]; m, root, rec = P4.start("calibration_r10.py", "rehearsal", pairs)
    T = pd.read_parquet(IN); T = T[T.pair.isin(pairs)]
    with Pool(workers) as pool: res = pool.map(one_pair, [(pid, T[T.pair == pid]) for pid in pairs], chunksize=1)
    res = [r for r in res if r]; elig = [r for r in res if r["eligible"]]
    pooled = {str(B): dict(coverage=float(np.nanmean([r["by_B"][str(B)]["coverage"] for r in elig])), frac_lower_gt0=float(np.nanmean([r["by_B"][str(B)]["frac_lower_gt0"] for r in elig])), median_width=float(np.nanmedian([r["by_B"][str(B)]["median_width"] for r in elig]))) for B in BS}
    if pooled["8"]["coverage"] >= 0.90: bmin, hold = 8, False
    elif pooled["12"]["coverage"] >= 0.90: bmin, hold = 12, False
    elif pooled["16"]["coverage"] >= 0.90: bmin, hold = 16, False
    else: bmin, hold = None, True
    json.dump(dict(rule="B_min = 8 if pooled coverage at B = 8 >= 0.90; else the smaller of 12 / 16 with coverage >= 0.90; else HOLD", Bs=BS, n_draw=N_DRAW, n_boot=N_BOOT, seed_draw=SEED_DRAW, seed_boot=P4.BOOT_SEED, min_blocks=MIN_BLOCKS, n_eligible_pairs=len(elig), ineligible=[dict(pair=r["pair"], n_blocks=r["n_blocks"]) for r in res if not r["eligible"]], pooled=pooled, B_min=bmin, hold=hold, per_pair=res, seconds=time.time() - t0), open(f"{OUT}/calibration.json", "w"), indent=1, default=float)
    print(f"eligible pairs {len(elig)}; pooled coverage " + ", ".join(f"B={B}: {pooled[str(B)]['coverage']:.3f} (lower>0 {pooled[str(B)]['frac_lower_gt0']:.2f}, width {pooled[str(B)]['median_width']:.2f})" for B in BS) + f" -> B_min {bmin}{' HOLD' if hold else ''} ({time.time()-t0:.0f} s)", flush=True)
