"""R04 §4 Check A — differential control re-measured.
1. Reproduce the submitted numbers from the submitted pipeline's persisted outputs (het_production_v1 per-patch parquet
   and summary; no model is run): within-pair R^2 of log sigma on log total error (paper: 0.555), the cross-pair pooled
   R^2 (summary: 0.460), and the fine-band numbers (summary 1.2e-5; report-34 6-fold 0.005; within-pair 0.003).
2. 2 x 2 (+ T-ship row) on the native 7 units: {submitted sigma, R02 clean sigma_m} x {paper fine-band amplitude
   (AUV-normalized, per 256 patch), A_fine in meters (R04 tile 128; also R04 F_hr RMS over the 256 window and the R03
   convention)}, plus T-ship (R02 definition), in the paper's metric and in the §2 metrics.
Alignment: the production parquet carries no tile key; its rows follow the R02 files_for() order (sorted file names,
200-per-pair linspace subsample), the same rule R02's evaluation used, so rows are aligned positionally per pair and the
alignment is verified by n_valid (paper) == n_scored (R02) on every row."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r04_common as C, eval_v1 as E
OUT = f"{C.ROOT}/checks"; os.makedirs(OUT, exist_ok=True); EPS = 1e-9
HP = f"{C.DATA}/models/het_production_v1/step0_bakeoff/het_louo"

def r2_pooled(x, y):
    """het_louo.r2_heldout: pooled log-log R^2 with fitted slope (the 'cross-pair' number)."""
    lx, ly = np.log10(np.asarray(x) + EPS), np.log10(np.asarray(y) + EPS); ss_res = np.sum((ly - np.polyval(np.polyfit(lx, ly, 1), lx)) ** 2); ss_tot = np.sum((ly - ly.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

def within_pair_r2(x, y, g): return E.within_r2_pooled(np.asarray(x, float), np.asarray(y, float), np.asarray(g))

if __name__ == "__main__":
    m = C.start("check_a"); prod = pd.read_parquet(f"{HP}/het_louo_perpatch.parquet"); summ = json.load(open(f"{HP}/het_louo_summary.json"))
    prod["unit"] = prod.pair.map({**C.NATIVE7_PAIRS, "EW0207__MGDS_32556": "EW0207"}); R = {}
    # ---- 1. reproduction
    rep = {}
    for lab, d in (("8 units (paper set)", prod), ("native 7 (EW0207 excluded)", prod[prod.unit != "EW0207"])):
        rep[lab] = dict(n=int(len(d)), cross_pair_r2_sigma_vs_resid=r2_pooled(d.mean_sigma, d.resid_rms), cross_pair_r2_sigma_vs_hrfine=r2_pooled(d.mean_sigma, d.hr_fine),
                        within_pair_r2_sigma_vs_resid_by_pair=within_pair_r2(d.mean_sigma, d.resid_rms, d.pair), within_pair_r2_sigma_vs_resid_by_unit=within_pair_r2(d.mean_sigma, d.resid_rms, d.unit),
                        within_pair_r2_sigma_vs_hrfine_by_pair=within_pair_r2(d.mean_sigma, d.hr_fine, d.pair), within_pair_r2_sigma_vs_hrfine_by_unit=within_pair_r2(d.mean_sigma, d.hr_fine, d.unit),
                        median_unit_rho_sigma_vs_resid=float(np.median([E.rho(s.mean_sigma.values, s.resid_rms.values) for _, s in d.groupby("unit")])),
                        median_unit_rho_sigma_vs_hrfine=float(np.median([E.rho(s.mean_sigma.values, s.hr_fine.values) for _, s in d.groupby("unit")])))
    rep["recorded_summary"] = summ["cross_pair"] | {"compare_to_regressor_r2": summ.get("compare_to_regressor_r2")}
    rep["paper_text"] = "'Within-site total-error localization reaches R2 = 0.555' (manuscript line 161-162); abstract 'error-localization R2 = 0.555 versus 0.107'; the directive's '0.53 / 0.005-0.06' are not verbatim in the manuscript: 0.555 is the within-pair number, 0.46 the pooled cross-pair number in het_louo_summary.json, 1.2e-5 (8-fold) / 0.005 (report-34 6-fold) the cross-pair sigma~fine, 0.003 the within-pair sigma~fine (REVIEW_ASSESSMENT §2.4), 0.06 the fine-band transfer regressor (summary compare_to_regressor_r2)."
    R["reproduction"] = rep
    # ---- 2. alignment with R02 per-tile table (mosaic mode): positional per pair (same files_for rule: sorted names,
    # linspace 200-cap). Verified by the coarse-roughness feature both pipelines computed from the same LR tile
    # (corr(log C_paper x hr_std, log C_R02) >= 0.85 per pair; shuffled pairing gives 0.05-0.60). n_valid (paper mask =
    # hr_valid & nearest-up lr_valid of the npz) vs n_scored (R02 joint & mosaic coverage) differ on partially covered tiles.
    pt = pd.read_parquet(f"{C.R02}/eval/pertile.parquet"); pt = pt[pt["mode"] == "mosaic"].copy(); al = []; bad = {}; ver = {}
    for pid, s in prod.groupby("pair", sort=False):
        q = pt[pt.pid == pid].sort_values("key").reset_index(drop=True); s = s.reset_index(drop=True)
        if len(q) != len(s): bad[pid] = f"row count {len(s)} vs {len(q)}"; continue
        ok = (s.C.values > 0) & (q.C.values > 0); r = float(np.corrcoef(np.log10(s.C.values[ok] * q.hr_std_m.values[ok]), np.log10(q.C.values[ok]))[0, 1])
        dn = s.n_valid.values - q.n_scored.values; ver[pid] = dict(corr_C=r, n_rows=int(len(q)), n_valid_mismatch_rows=int((dn != 0).sum()), rel_dn_p99=float(np.percentile(np.abs(dn) / q.n_scored.values, 99)))
        if r < 0.85: bad[pid] = f"alignment not verified (corr C = {r:.3f})"; continue
        al.append(pd.DataFrame(dict(pair=pid, key=q.key.values, sigma_sub=s.mean_sigma.values, resid_sub=s.resid_rms.values, hr_fine_paper=s.hr_fine.values, n_valid=s.n_valid.values, n_scored=q.n_scored.values,
                                    hr_std_m=q.hr_std_m.values, sigma_sub_m=s.mean_sigma.values * q.hr_std_m.values, resid_sub_m=s.resid_rms.values * q.hr_std_m.values,
                                    sigma_m_R02=q.sigma_m.values, T_ship=q.T_ship.values, s_lr=q.s_lr.values)))
    A = pd.concat(al, ignore_index=True); A["r0"] = A.key.str.extract(r"_r(\d{6})_").astype(int); A["c0"] = A.key.str.extract(r"_c(\d{6})").astype(int)
    R["alignment"] = dict(n_aligned=int(len(A)), pairs_aligned=sorted(A.pair.unique()), failures=bad, verification=ver)
    # R04 targets (tile 128 central; F_hr RMS over the 256 window; R03 convention) by window key
    tiles = pd.read_parquet(f"{C.ROOT}/tiles/tiles_v1.parquet")[["pair", "r0", "c0", "unit", "A_fine", "A_fine_Fhr_256", "A_fine_r03conv_256", "sigma_m_R02"]].rename(columns={"A_fine": "A_fine_tile128", "sigma_m_R02": "sigma_m_R02_tile128"})
    A = A.merge(tiles, on=["pair", "r0", "c0"], how="left"); A = A[A.pair.isin(C.NATIVE7_PAIRS)].copy(); A["unit"] = A.pair.map({p: next(u for u, ps in C.UNITS.items() if p in ps) for p in C.NATIVE7_PAIRS}); A["tile_id"] = A.key; A["block_id"] = -1
    R["alignment"]["n_native7"] = int(len(A)); R["alignment"]["n_with_r04_tile"] = int(A.A_fine_tile128.notna().sum()); A.to_parquet(f"{OUT}/check_a_table.parquet")
    # ---- 3. the table
    units = sorted(A.unit.unique()); models = {"submitted sigma (het_production_v1, AUV-normalized, per 256 patch)": "sigma_sub", "submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side)": "sigma_sub_m", "R02 clean sigma_m (mosaic, m, per 256 tile)": "sigma_m_R02", "R02 clean sigma_m (mosaic, m, per 128 tile)": "sigma_m_R02_tile128"}
    targets = {"paper total error resid_rms (AUV-normalized, 256 patch)": "resid_sub", "paper fine-band amplitude (AUV-normalized, 256 patch)": "hr_fine_paper", "A_fine tile 128 (m, R04)": "A_fine_tile128", "A_fine F_hr RMS over the 256 window (m, R04)": "A_fine_Fhr_256", "A_fine R03 convention, 256 window (m)": "A_fine_r03conv_256", "T-ship (R02 definition, m, 256 mosaic tile)": "T_ship"}
    tab = {}; md = ["# Check A — differential control re-measured (native 7 units)\n", "## 1. Reproduction of the submitted numbers (from het_production_v1 persisted outputs)\n", "| set | n | cross-pair pooled R² σ~total error | cross-pair pooled R² σ~fine | within-pair R² σ~total (by pair / by unit) | within-pair R² σ~fine (by pair / by unit) | median unit ρ σ~total / σ~fine |", "|---|---|---|---|---|---|---|"]
    for lab in ("8 units (paper set)", "native 7 (EW0207 excluded)"):
        r = rep[lab]; md.append(f"| {lab} | {r['n']} | {r['cross_pair_r2_sigma_vs_resid']:.3f} | {r['cross_pair_r2_sigma_vs_hrfine']:.5f} | {r['within_pair_r2_sigma_vs_resid_by_pair']:.3f} / {r['within_pair_r2_sigma_vs_resid_by_unit']:.3f} | {r['within_pair_r2_sigma_vs_hrfine_by_pair']:.3f} / {r['within_pair_r2_sigma_vs_hrfine_by_unit']:.3f} | {r['median_unit_rho_sigma_vs_resid']:.3f} / {r['median_unit_rho_sigma_vs_hrfine']:.3f} |")
    md += [f"\nRecorded in het_louo_summary.json: {rep['recorded_summary']}. {rep['paper_text']}\n", f"Alignment: {R['alignment']['n_aligned']} rows aligned positionally; verification per pair (corr of log C x hr_std vs log C_R02; n_valid-mismatch rows; p99 relative mismatch): " + "; ".join(f"{p} {v['corr_C']:.3f} / {v['n_valid_mismatch_rows']} / {v['rel_dn_p99']:.3f}" for p, v in ver.items()) + f"; failures {bad}; native-7 rows {len(A)}, of which {R['alignment']['n_with_r04_tile']} fall on the stride-128 R04 grid.\n",
           "## 2. Models x targets (native 7 units; paper metric = within-pair log-log R², unit-demeaned pooled; §2 metrics = median within-unit Spearman ρ / median within-unit log-log R² / median top-20 % capture; n rows)\n", "| model | target | paper metric: within-pair R² | median ρ | median log-log R² | median capture20 | pooled ρ | n |", "|---|---|---|---|---|---|---|---|"]
    for mn, mc in models.items():
        for tn, tc in targets.items():
            d = A[np.isfinite(A[mc]) & np.isfinite(A[tc]) & (A[tc] > 0)]
            if len(d) < 20: tab[f"{mn} | {tn}"] = "COULD NOT CHECK (n < 20)"; md.append(f"| {mn} | {tn} | COULD NOT CHECK | | | | | {len(d)} |"); continue
            S = E.score_one(d, mc, tc, "tile", sorted(d.unit.unique())); wr = within_pair_r2(d[mc], d[tc], d.unit)
            tab[f"{mn} | {tn}"] = dict(within_pair_r2=wr, median_rho=S["median"]["rho"], median_r2_loglog=S["median"].get("r2_loglog"), median_capture20=S["median"]["capture20"], pooled_rho=S["pooled"]["rho"], n=int(len(d)), per_unit_rho=S["per_unit"])
            md.append(f"| {mn} | {tn} | {wr:.3f} | {S['median']['rho']:.3f} | {S['median'].get('r2_loglog', np.nan):.3f} | {S['median']['capture20']:.3f} | {S['pooled']['rho']:.3f} | {len(d)} |")
    R["table"] = tab
    g = lambda mn, tn: tab.get(f"{mn} | {tn}", {}); sub = list(models)[0]; subm = list(models)[1]; r02 = list(models)[2]
    def v(mn, tn, k): x = g(mn, tn); return x.get(k, np.nan) if isinstance(x, dict) else np.nan
    T = list(targets)
    verdict = (f"Submitted σ (paper's units): within-pair R² {v(sub, T[0], 'within_pair_r2'):.3f} on its own total error vs {v(sub, T[1], 'within_pair_r2'):.3f} on the paper's fine-band amplitude, and {v(sub, T[5], 'within_pair_r2'):.3f} on T-ship (m) vs {v(sub, T[2], 'within_pair_r2'):.3f} on A_fine tile (m); "
               f"submitted σ in meters: {v(subm, T[5], 'within_pair_r2'):.3f} on T-ship vs {v(subm, T[2], 'within_pair_r2'):.3f} on A_fine tile ({v(subm, T[3], 'within_pair_r2'):.3f} on the 256 window); "
               f"R02 clean σ_m: {v(r02, T[5], 'within_pair_r2'):.3f} on T-ship vs {v(r02, T[2], 'within_pair_r2'):.3f} on A_fine tile ({v(r02, T[3], 'within_pair_r2'):.3f} on the 256 window); "
               f"median within-unit ρ: R02 σ_m {v(r02, T[5], 'median_rho'):.2f} (T-ship) vs {v(r02, T[2], 'median_rho'):.2f} (A_fine tile); submitted σ in m {v(subm, T[5], 'median_rho'):.2f} vs {v(subm, T[2], 'median_rho'):.2f}.")
    R["verdict_numbers"] = verdict; md += ["", "## 3. Verdict (numbers; one sentence in the report)", "", verdict]
    json.dump(R, open(f"{OUT}/check_a.json", "w"), indent=1, default=float); open(f"{OUT}/check_a.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
