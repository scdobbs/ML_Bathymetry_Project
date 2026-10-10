"""R09 Part A2 / A3 scoring. Candidates s_lr (ref_s_lr), GBT_ship, P-V1 seed-averaged (existing predictions, R08b master table)
scored through eval_v1_1 (within-pair rho, pair eligibility >= 10 tiles with finite target and prediction) against:
  (i)  A_fine on all tiles;            (ii) A_fine on R08b interior tiles at D1;
  (iii) A_fine_int at D1 (tiles with a defined interior-pixel target);
  A3:  A_fine on interior tiles at D125; A_fine_int at D125.
Unit sets: ten (lu14 shown beside, out of every median), nine, six, all 16. Paired median delta (iii) - (ii) per predictor
with the eval_v1_1 bootstrap over units (2,000, seed 20261001), units better, exact sign test. Validation rule: |paired
median delta| <= 0.05 for both s_lr and GBT_ship on the primary units scored both ways."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r09_common as R9, r04_common as C4, eval_v1_1 as E
sys.path.insert(0, R9.R08B_SCRIPTS); from score_r08b import median_ci_vals, compare_vals

CAND = R9.CANDIDATES; PREDS = list(CAND.values()); units16 = list(R9.ALL16)
VIEWS = {"all_Afine": ("A_fine", None), "intD1_Afine": ("A_fine", "interior_D1"), "intpx_D1": ("A_fine_int_D1", None), "intD125_Afine": ("A_fine", "interior_D125"), "intpx_D125": ("A_fine_int_D125", None)}

def per_unit(tab, target, pred):
    el = sorted(k for k, v in E.eligible_pairs(tab, pred, target).items() if v["eligible"])
    S = E.score_one(tab, pred, target, "tile", units16, elig_pairs=el); return {u: S["per_unit"][u].get("rho", np.nan) for u in units16}, el

if __name__ == "__main__":
    t0 = time.time(); m = R9.start("score_r09")
    M = pd.read_parquet(f"{R9.RB.OUT_SCORES}/master_tile.parquet"); M = M[["unit", "pair", "tile_id", "block_id", "A_fine"] + PREDS].copy()
    TI = pd.read_parquet(f"{R9.OUT_INT}/tiles_int.parquet").set_index("tile_id")
    for c in ("interior_D1", "n_int_px_D1", "A_fine_int_D1", "interior_D125", "n_int_px_D125", "A_fine_int_D125"): M[c] = TI[c].reindex(M.tile_id).values
    assert M.interior_D1.notna().all(); M.to_parquet(f"{R9.OUT_SCORES}/master_tile_r09.parquet")
    # R08b consistency: the interior_D1 view must reproduce R08b's per-unit rho for the three candidates
    S8 = json.load(open(R9.R08B_SCORES)); PU = {}; EL = {}
    for vname, (tgt, flag) in VIEWS.items():
        tab = M if flag is None else M[M[flag] == True]
        PU[vname] = {}; EL[vname] = {}
        for p in PREDS: PU[vname][p], EL[vname][p] = per_unit(tab, tgt, p)
    for p in PREDS:
        for u in units16:
            a, b = PU["intD1_Afine"][p][u], S8["per_unit"]["D1"][p].get(u, np.nan)
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12, (p, u, a, b)
    # coverage per pair / unit
    cov_p = TI.groupby("pair").agg(unit=("unit", "first"), n_tiles=("A_fine", "size"), interior_D1=("interior_D1", "sum"), defined_int_D1=("A_fine_int_D1", lambda s: int(s.notna().sum())), interior_D125=("interior_D125", "sum"), defined_int_D125=("A_fine_int_D125", lambda s: int(s.notna().sum()))).reset_index()
    cov_u = TI.groupby("unit").agg(n_tiles=("A_fine", "size"), interior_D1=("interior_D1", "sum"), defined_int_D1=("A_fine_int_D1", lambda s: int(s.notna().sum())), interior_D125=("interior_D125", "sum"), defined_int_D125=("A_fine_int_D125", lambda s: int(s.notna().sum()))).reset_index()
    cov_p.to_csv(f"{R9.OUT_SCORES}/coverage_per_pair.csv", index=False); cov_u.to_csv(f"{R9.OUT_SCORES}/coverage_per_unit.csv", index=False)
    # medians per set (lu14 out of every median), comparisons
    OUT = dict(candidates=CAND, views=list(VIEWS), per_unit=PU, eligible=EL, median={}, intpx_minus_intD1={}, intD125_minus_all={}, intpx125_minus_all={}, lu14={})
    for name, uset in R9.UNIT_SETS.items():
        us = [u for u in uset if u not in R9.NO_INTERIOR]
        OUT["median"][name] = {v: {p: median_ci_vals([PU[v][p][u] for u in us]) for p in PREDS} for v in VIEWS}
        OUT["intpx_minus_intD1"][name] = {p: compare_vals([PU["intpx_D1"][p][u] for u in us], [PU["intD1_Afine"][p][u] for u in us], us) for p in PREDS}
        OUT["intD125_minus_all"][name] = {p: compare_vals([PU["intD125_Afine"][p][u] for u in us], [PU["all_Afine"][p][u] for u in us], us) for p in PREDS}
        OUT["intpx125_minus_all"][name] = {p: compare_vals([PU["intpx_D125"][p][u] for u in us], [PU["all_Afine"][p][u] for u in us], us) for p in PREDS}
    OUT["lu14"] = {v: {p: PU[v][p]["lu14"] for p in PREDS} for v in VIEWS}
    # validation rule on the primary units scored both ways
    v = {p: OUT["intpx_minus_intD1"]["ten"][p] for p in ("ref_s_lr", "GBT_ship_A_fine")}
    OUT["validation"] = dict(rule=f"|paired median delta (A_fine_int D1 - interior-tile D1)| <= {R9.VALIDATION_TOL} for both s_lr and GBT_ship on the primary units scored both ways", tol=R9.VALIDATION_TOL,
                             s_lr=dict(paired_median=v["ref_s_lr"]["paired_median"], ci95=v["ref_s_lr"]["paired_ci95"], n_units=v["ref_s_lr"]["n_units"], passes=bool(abs(v["ref_s_lr"]["paired_median"]) <= R9.VALIDATION_TOL)),
                             GBT_ship=dict(paired_median=v["GBT_ship_A_fine"]["paired_median"], ci95=v["GBT_ship_A_fine"]["paired_ci95"], n_units=v["GBT_ship_A_fine"]["n_units"], passes=bool(abs(v["GBT_ship_A_fine"]["paired_median"]) <= R9.VALIDATION_TOL)))
    OUT["validation"]["validated"] = bool(OUT["validation"]["s_lr"]["passes"] and OUT["validation"]["GBT_ship"]["passes"])
    R9.R8.dump(OUT, f"{R9.OUT_SCORES}/scores_r09.json")
    for name in R9.UNIT_SETS: print(f"[{name}] " + "; ".join(f"{p}: all {OUT['median'][name]['all_Afine'][p]['value']:.3f} intD1 {OUT['median'][name]['intD1_Afine'][p]['value']:.3f} intpx {OUT['median'][name]['intpx_D1'][p]['value']:.3f} (paired Δ {OUT['intpx_minus_intD1'][name][p].get('paired_median', np.nan):+.3f}) | D125 tiles {OUT['median'][name]['intD125_Afine'][p]['value']:.3f} (n {OUT['median'][name]['intD125_Afine'][p]['n_units']}) intpx125 {OUT['median'][name]['intpx_D125'][p]['value']:.3f} (n {OUT['median'][name]['intpx_D125'][p]['n_units']})" for p in PREDS))
    print("validation:", OUT["validation"]); print(f"done ({time.time()-t0:.0f} s)")
