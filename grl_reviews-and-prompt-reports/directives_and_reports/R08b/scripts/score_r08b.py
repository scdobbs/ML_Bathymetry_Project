"""R08b §3-§4 — re-score every existing prediction on interior tiles (D1 primary, D2 sensitivity) and on all tiles, through
eval_v1_1.score_one (within-pair rho, unit = median over eligible pairs). Eligibility on each subset = eval_v1_1.eligible_pairs
(>= 10 tiles with finite target and finite prediction), per predictor. Unit sets: ten (primary), nine, six, all 16;
EW0207 one-unit table. Statistics per predictor and set: median rho with the eval_v1_1 bootstrap (2,000 resamples over
units, seed 20261001). Comparisons two ways: difference of medians (eval_v1_1.compare's bootstrap) and the median of the
per-unit paired differences with its own bootstrap, units better, exact one-sided sign test. Nothing is refit."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r08b_common as RB, r04_common as C4, eval_v1_1 as E

FAMILIES = {
    "free references": ["ref_s_lr", "ref_C", "ref_shipstd128", "ref_rsd128"],
    "bar": ["GBT_ship_A_fine", "GBT_noship_A_fine"],
    "bar without edge features (R08)": ["GBT_ship_dropEV_A_fine"],
    "round 1 (R06)": ["V0", "V1_s20261002", "V1_s20261003", "V1_s20261004", "V1_seedavg", "V2_s20261002", "V3"],
    "pretraining and seed completion (R07)": ["P-V1_s20261002", "P-V1_s20261003", "P-V1_s20261004", "P-V1_seedavg", "P-V2_s20261002", "P-V2_s20261003", "P-V2_s20261004", "P-V2_seedavg", "V2_s20261003", "V2_s20261004", "V2_seedavg"],
    "old model (R02, native 7)": ["sigma_m_R02"],
    "within-site (R08, feasible pairs)": ["within_site"],
}
PREDS = [p for v in FAMILIES.values() for p in v]
SUBSETS = ["all", "D1", "D2"]
PAIRED = [("GBT_ship_A_fine", "ref_s_lr"), ("V2_seedavg", "GBT_ship_A_fine"), ("V2_seedavg", "ref_s_lr"), ("P-V1_seedavg", "GBT_ship_A_fine"), ("GBT_ship_dropEV_A_fine", "GBT_ship_A_fine")]

def master_table():
    P5 = pd.read_parquet(RB.PRED_SOURCES["R05_tile"][0]); P6 = pd.read_parquet(RB.PRED_SOURCES["R06_tile"][0]); P7 = pd.read_parquet(RB.PRED_SOURCES["R07_tile"][0])
    W = pd.read_parquet(RB.PRED_SOURCES["R08_within"][0]); G = pd.read_parquet(RB.PRED_SOURCES["R08_dropgroup"][0])
    for X in (P6, P7, W, G): assert (X.tile_id.values == P5.tile_id.values).all()
    M = P5[["unit", "pair", "tile_id", "block_id", "A_fine", "sigma_m_R02", "ref_C", "ref_s_lr", "ref_shipstd128", "ref_rsd128", "GBT_ship_A_fine", "GBT_noship_A_fine"]].copy()
    for c in ("V0", "V1_s20261002", "V1_s20261003", "V1_s20261004", "V1_seedavg", "V3"): M[c] = P6[c].values
    M["V2_s20261002"] = P6["V2"].values; assert np.allclose(P7["V2_s20261002"].values, P6["V2"].values, rtol=0, atol=0), "R07 V2_s20261002 != R06 V2"
    for c in FAMILIES["pretraining and seed completion (R07)"]: M[c] = P7[c].values
    M["within_site"] = W["within_site"].values; M["GBT_ship_dropEV_A_fine"] = G["GBT_ship_dropEV_A_fine"].values
    E5 = pd.read_parquet(RB.PRED_SOURCES["R05_tile_EW"][0]); E6 = pd.read_parquet(RB.PRED_SOURCES["R06_tile_EW"][0]); E7 = pd.read_parquet(RB.PRED_SOURCES["R07_tile_EW"][0])
    assert (E6.tile_id.values == E5.tile_id.values).all() and (E7.tile_id.values == E5.tile_id.values).all()
    EW = E5[["unit", "pair", "tile_id", "block_id", "A_fine", "ref_C", "ref_s_lr", "ref_shipstd128", "ref_rsd128", "GBT_ship_A_fine", "GBT_noship_A_fine"]].copy()
    for c, s in (("V0", "V0"), ("V1_s20261002", "V1"), ("V2_s20261002", "V2"), ("V3", "V3")): EW[c] = E6[s].values
    for c in [c for c in E7.columns if c.startswith(("P-V1", "P-V2")) or c in ("V2_s20261003", "V2_s20261004", "V2_seedavg")]: EW[c] = E7[c].values
    return M, EW

def median_ci_vals(v, seed=E.SEED, n_boot=E.N_BOOT):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; n = len(v)
    if n == 0: return dict(value=np.nan, ci95=[np.nan, np.nan], n_units=0)
    rng = np.random.default_rng(seed); bs = np.array([np.median(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return dict(value=float(np.median(v)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_units=int(n))

def compare_vals(va, vb, units, seed=E.SEED, n_boot=E.N_BOOT):
    """eval_v1_1.compare on per-unit values (difference of medians) plus the median of per-unit paired differences."""
    va = np.asarray(va, float); vb = np.asarray(vb, float); ok = np.isfinite(va) & np.isfinite(vb); va, vb = va[ok], vb[ok]; uu = [u for u, o in zip(units, ok) if o]; n = len(va)
    if n == 0: return dict(n_units=0)
    rng = np.random.default_rng(seed); idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    bs = np.array([np.median(va[i]) - np.median(vb[i]) for i in idx]); d = va - vb; bsp = np.array([np.median(d[i]) for i in idx])
    better = int(np.sum(va > vb)); worse = int(np.sum(va < vb)); ties = int(np.sum(va == vb))
    p = float(binomtest(better, better + worse, 0.5, alternative="greater").pvalue) if better + worse > 0 else np.nan
    return dict(n_units=n, diff_median=float(np.median(va) - np.median(vb)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                paired_median=float(np.median(d)), paired_ci95=[float(np.percentile(bsp, 2.5)), float(np.percentile(bsp, 97.5))], units_a_better=better, units_b_better=worse, ties=ties, sign_test_p_one_sided=p, per_unit_diff={u: float(x) for u, x in zip(uu, d)})

if __name__ == "__main__":
    t0 = time.time(); m = RB.start("score_r08b"); units16 = list(RB.ALL16)
    M, EW = master_table(); TI = pd.read_parquet(f"{RB.OUT_INT}/tiles_interior.parquet").set_index("tile_id")
    for k in ("D1", "D2"): M[f"interior_{k}"] = TI[f"interior_{k}"].reindex(M.tile_id).values; EW[f"interior_{k}"] = TI[f"interior_{k}"].reindex(EW.tile_id).values
    assert M.interior_D1.notna().all() and EW.interior_D1.notna().all(); M.to_parquet(f"{RB.OUT_SCORES}/master_tile.parquet"); EW.to_parquet(f"{RB.OUT_SCORES}/master_tile_EW0207.parquet")
    SUB = {"all": M, "D1": M[M.interior_D1 == True], "D2": M[M.interior_D2 == True]}
    # ---- eligibility (target-based, for the drop ledger) and per-predictor eligibility sets
    elig_t = {s: {p: v for p, v in E.eligible_pairs(SUB[s].assign(_p=1.0), "_p", "A_fine").items()} for s in SUBSETS}
    drops = {}
    for s in ("D1", "D2"):
        gone = sorted(p for p in C4.DEV_PAIRS if not elig_t[s].get(p, {}).get("eligible", False))
        drops[s] = dict(pairs=[dict(pair=p, unit=C4.UNIT_OF[p], n_tiles_ok=elig_t[s].get(p, {}).get("n_tiles_ok", 0)) for p in gone], units=[u for u in units16 if all(p in gone for p in C4.UNITS[u])])
    elig = {s: {p: sorted(k for k, v in E.eligible_pairs(SUB[s], p, "A_fine").items() if v["eligible"]) for p in PREDS} for s in SUBSETS}
    RB.R8.dump(dict(target_based=elig_t, drops=drops, per_predictor=elig), f"{RB.OUT_SCORES}/eligibility.json")
    # ---- per-unit rho for every predictor and subset
    PU = {s: {} for s in SUBSETS}
    for s in SUBSETS:
        for p in PREDS:
            uu = list(RB.NATIVE7) if p == "sigma_m_R02" else units16
            S = E.score_one(SUB[s], p, "A_fine", "tile", uu, elig_pairs=elig[s][p]); PU[s][p] = {u: S["per_unit"][u].get("rho", np.nan) for u in uu}
        print(f"  scored {s} ({time.time()-t0:.0f} s)", flush=True)
    # ---- medians per set, comparisons
    OUT = dict(predictors=PREDS, families=FAMILIES, subsets=SUBSETS, unit_sets=RB.UNIT_SETS, per_unit=PU, median={}, interior_vs_all={}, paired={}, within_vs_gbt={}, reading_guide={})
    for name, uset in RB.UNIT_SETS.items():
        OUT["median"][name] = {s: {p: median_ci_vals([PU[s][p].get(u, np.nan) for u in uset]) for p in PREDS} for s in SUBSETS}
        OUT["interior_vs_all"][name] = {s: {p: compare_vals([PU[s][p].get(u, np.nan) for u in uset], [PU["all"][p].get(u, np.nan) for u in uset], uset) for p in PREDS} for s in ("D1", "D2")}
        OUT["paired"][name] = {s: {f"{a}-{b}": compare_vals([PU[s][a].get(u, np.nan) for u in uset], [PU[s][b].get(u, np.nan) for u in uset], uset) for a, b in PAIRED} for s in SUBSETS}
    # ---- within-site vs held-out GBT_ship on the feasible pairs (same pairs for both)
    for s in SUBSETS:
        el = elig[s]["within_site"]; Sg = E.score_one(SUB[s], "GBT_ship_A_fine", "A_fine", "tile", units16, elig_pairs=el); Sw = E.score_one(SUB[s], "within_site", "A_fine", "tile", units16, elig_pairs=el)
        OUT["within_vs_gbt"][s] = {name: dict(n_pairs=len(el), within=median_ci_vals([Sw["per_unit"][u].get("rho", np.nan) for u in uset]), gbt_same_pairs=median_ci_vals([Sg["per_unit"][u].get("rho", np.nan) for u in uset]),
                                             cmp=compare_vals([Sw["per_unit"][u].get("rho", np.nan) for u in uset], [Sg["per_unit"][u].get("rho", np.nan) for u in uset], uset), per_unit={u: dict(within=Sw["per_unit"][u].get("rho", np.nan), gbt=Sg["per_unit"][u].get("rho", np.nan)) for u in uset}) for name, uset in RB.UNIT_SETS.items()}
    # ---- reading guide (§4)
    for p in ("GBT_ship_A_fine", "ref_s_lr"):
        c = OUT["interior_vs_all"]["ten"]["D1"][p]; OUT["reading_guide"][p] = dict(paired_median=c["paired_median"], paired_ci95=c["paired_ci95"], material=bool(c["paired_median"] <= RB.MATERIAL_DELTA or c["paired_ci95"][1] < RB.MATERIAL_CI_HI))
    OUT["reading_guide"]["material_any"] = bool(any(OUT["reading_guide"][p]["material"] for p in ("GBT_ship_A_fine", "ref_s_lr")))
    # ---- EW0207 one-unit table
    ew_preds = [c for c in EW.columns if c in PREDS]; EWS = {}
    for s, tab in (("all", EW), ("D1", EW[EW.interior_D1 == True]), ("D2", EW[EW.interior_D2 == True])):
        EWS[s] = dict(n_tiles=int(len(tab)), rho={p: E.score_one(tab.assign(unit="lu16"), p, "A_fine", "tile", ["lu16"], elig_pairs=[RB.R8.EW_PAIR])["per_unit"]["lu16"].get("rho", np.nan) for p in ew_preds})
    OUT["EW0207"] = EWS
    RB.R8.dump(OUT, f"{RB.OUT_SCORES}/scores.json")
    for name in RB.UNIT_SETS:
        print(f"[{name}] " + "; ".join(f"{p} all {OUT['median'][name]['all'][p]['value']:.3f} D1 {OUT['median'][name]['D1'][p]['value']:.3f} (paired Δ {OUT['interior_vs_all'][name]['D1'][p].get('paired_median', np.nan):+.3f})" for p in ("ref_s_lr", "GBT_ship_A_fine", "GBT_ship_dropEV_A_fine", "V2_seedavg", "P-V1_seedavg")))
    print("reading guide:", OUT["reading_guide"]); print(f"done ({time.time()-t0:.0f} s)")
