"""R08 §2 — feature importance scored in the evaluation metric.
(1) Refit the frozen R05 GBT_ship A_fine / tile models (one per held-out unit; the bar's outer fit: capped training rows
    of the other 15 units, the configuration R05 selected for that unit, float32 features) and check that the refit
    predictions equal R05's pred_tile.parquet within 1e-9 m (stop otherwise).
(2) Fixed feature groups; every column of the frozen GBT_ship feature set assigned to exactly one group (mapping written).
(3) Permutation importance on the held-out predictions: for each unit and group, permute the group's columns jointly
    (one row permutation per pair, applied to all the group's columns), predict with the unit's held-out model, score
    through eval_v1_1 (within-pair rho, median over the unit's eligible pairs); 30 permutations from one
    numpy.random.default_rng(20261009) stream in the order units (units-v1 order) > groups > permutations > pairs (sorted).
    Importance = baseline unit rho - mean permuted unit rho.
(4) Drop-group refit: the frozen nested LOUO (bar_v1_1.nested_louo: full grid, inner criterion eval_v1_1, cap, seed)
    without the edge-and-validity group; scored against GBT_ship over all 16, the six and the ten."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r08_common as R8, r04_common as C4, bar_v1_1 as B, eval_v1_1 as E

GROUPS = {   # §2 fixed groups; '(nature)' marks columns the directive does not name literally, assigned by what they measure
    "roughness": ["s_lr_128", "s_lr_224", "s_lr_256", "std_ship_128", "std_ship_224", "C_128", "C_224", "C_hr_128", "C_hr_224", "curv_128", "curv_224", "curv_hr_128", "curv_hr_224", "curv_ship_std_128", "curv_ship_std_224"],
    "slope": ["slope_ship_mean_128", "slope_ship_mean_224", "slope_ship_p90_128", "slope_ship_p90_224", "slope_hr_128", "slope_hr_224", "slope_128", "slope_224"],
    "spectral": ["beta_128", "beta_224", "logA_128", "logA_224", "fit_r2_128", "fit_r2_224", "shat_fine_128", "shat_fine_224", "noise_excess_128", "noise_excess_224", "aniso_spec_128", "aniso_spec_224", "beta_clamped_128", "beta_clamped_224", "n_fit_bins_128", "n_fit_bins_224"],
    "anisotropy": ["aniso_128", "aniso_224", "aniso_hr_128", "aniso_hr_224"],
    "depth_metadata": ["depth", "lr_native", "k", "hr_res", "log_depth_over_lr", "mu_lr_128", "mu_lr_224", "n_cells_128", "n_cells_224"],
    "edge_validity": ["dist_edge_128", "dist_edge_224", "frac_valid_128", "frac_valid_224", "valid_frac_128", "valid_frac_224"],
    "ship_products": [f"{k}_{w}" for w in (128, 224) for k in ("rsd_med", "rsd_p90", "sd_med", "logcount_med", "xtrack_mean", "beam_mean", "rsd_over_sd_med")],
}
BY_NATURE = {"s_lr_224", "slope_128", "slope_224", "beta_clamped_128", "beta_clamped_224", "n_fit_bins_128", "n_fit_bins_224", "mu_lr_128", "mu_lr_224", "n_cells_128", "n_cells_224", "valid_frac_128", "valid_frac_224", "curv_hr_128", "curv_hr_224"}
GROUP_ORDER = list(GROUPS)
D = {}

def _refit(u):
    X, y, g, cm, cfg = D["X"], D["y"], D["g"], D["cm"], D["cfg"][u]; tr = cm & (g != u); te = np.where(g == u)[0]
    mdl = B.make(B.GRID[cfg], "reg").fit(X[tr], y[tr]); return u, te, mdl.predict(X[te]), mdl

def unit_rho(tab_u, u, col, elig):
    return E.score_one(tab_u, col, "A_fine", "tile", [u], elig_pairs=elig)["per_unit"][u].get("rho", np.nan)

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)); t0 = time.time()
    m = R8.start("importance_r08"); os.makedirs(R8.OUT_IMP, exist_ok=True); units = list(C4.UNITS)
    P, rules = R8.load_pred_tile(); EP = rules["eligible"]["A_fine"]
    tiles = R8.load_tiles(); blocks = pd.read_parquet(R8.R05_BLOCKS); feats = R8.load_features()
    Bt, Tt = B.assemble(tiles, blocks, feats); dev_T = Tt[Tt.unit.isin(units)].reset_index(drop=True)
    assert (dev_T.tile_id.values == P.tile_id.values).all(), "tile order differs from the R05 prediction table"
    # ---- group mapping covers B.ALL exactly once
    mapped = [c for g in GROUPS.values() for c in g]; assert len(mapped) == len(set(mapped)), "a column is in two groups"
    other = [c for c in B.ALL if c not in set(mapped)]; GROUPS_ALL = dict(GROUPS); 
    if other: GROUPS_ALL["other"] = other
    extra = [c for c in mapped if c not in B.ALL]; assert not extra, f"group columns not in the frozen feature set: {extra}"
    R8.dump(dict(groups=GROUPS_ALL, n_features=len(B.ALL), by_nature=sorted(BY_NATURE), other=other, feature_set="bar_v1_1.ALL (features_v1 columns + META)"), f"{R8.OUT_IMP}/groups.json")
    # ---- (1) refit check
    ok = np.isfinite(dev_T["A_fine"].values) & (dev_T["A_fine"].values > 0)
    X = dev_T.loc[ok, B.ALL].values.astype(np.float32); y = np.log10(dev_T.loc[ok, "A_fine"].values + B.EPS); g = dev_T.unit.values[ok]; prs = dev_T.pair.values[ok]
    chosen = json.load(open(R8.CHOSEN_TILE))["tile/A_fine/GBT_ship"]; cfg = {u: int(chosen[u]["cfg_idx"]) for u in units}
    D.update(X=X, y=y, g=g, cm=B.cap_mask(g), cfg=cfg)
    with Pool(min(workers, 16)) as pool: outs = pool.map(_refit, units)
    pred = np.full(len(X), np.nan); models = {}
    for u, te, p, mdl in outs: pred[te] = p; models[u] = mdl
    ref = P["GBT_ship_A_fine"].values[ok]; diff = np.abs(10 ** pred - ref); mx = float(np.nanmax(diff))
    chk = dict(n_rows=int(ok.sum()), max_abs_diff_m=mx, n_rows_diff_gt_1e9=int((diff > 1e-9).sum()), threshold_m=1e-9, match=bool(mx <= 1e-9),
               configs={u: dict(cfg_idx=cfg[u], **B.GRID[cfg[u]]) for u in units}, note="outer fit of bar_v1_1 (cap_mask rows of the 15 other units, float32 features, R05 chosen configuration per unit); prediction 10**pred (m)")
    R8.dump(chk, f"{R8.OUT_IMP}/refit_check.json"); print(f"refit check: max |diff| {mx:.3e} m on {ok.sum()} rows -> {'MATCH' if chk['match'] else 'MISMATCH'} ({time.time()-t0:.0f} s)", flush=True)
    if not chk["match"]: raise SystemExit("STOP (directive §2): refit predictions do not match R05 within 1e-9")
    # ---- (3) permutation importance, one rng stream, serial in the fixed order
    rng = np.random.default_rng(R8.SEED); col_idx = {gname: [B.ALL.index(c) for c in cols] for gname, cols in GROUPS_ALL.items()}
    tabP = P[["unit", "pair", "tile_id", "block_id", "A_fine"]].copy(); tabP["perm"] = np.nan
    rows = []; base = {}
    for u in units:
        iu = np.where(g == u)[0]; Xu = X[iu]; pr_u = prs[iu]; mdl = models[u]
        tab_u = tabP.iloc[np.where(P.unit.values == u)[0]].copy(); assert (tab_u.tile_id.values == dev_T.tile_id.values[ok][iu]).all()
        tab_u["perm"] = 10 ** mdl.predict(Xu); base[u] = unit_rho(tab_u, u, "perm", EP)
        assert abs(base[u] - E.score_one(P[P.unit == u], "GBT_ship_A_fine", "A_fine", "tile", [u], elig_pairs=EP)["per_unit"][u]["rho"]) < 1e-12
        pair_idx = {p: np.where(pr_u == p)[0] for p in sorted(set(pr_u))}
        for gname in GROUPS_ALL:
            ci = col_idx[gname]; vals = []
            for k in range(R8.N_PERM):
                Xp = Xu.copy()
                for p, idx in pair_idx.items():
                    perm = rng.permutation(len(idx)); Xp[np.ix_(idx, ci)] = Xu[np.ix_(idx[perm], ci)]
                tab_u["perm"] = 10 ** mdl.predict(Xp); vals.append(unit_rho(tab_u, u, "perm", EP))
            vals = np.array(vals, float)
            rows.append(dict(unit=u, group=gname, n_features=len(ci), baseline_rho=base[u], mean_permuted_rho=float(np.nanmean(vals)), sd_permuted_rho=float(np.nanstd(vals, ddof=1)), importance=float(base[u] - np.nanmean(vals)), n_perm=int(np.isfinite(vals).sum()), low_amplitude=(u in R8.LOW_AMP)))
        print(f"  {u}: baseline rho {base[u]:+.3f}; importance " + ", ".join(f"{r['group']} {r['importance']:+.3f}" for r in rows if r["unit"] == u) + f" ({time.time()-t0:.0f} s)", flush=True)
    PI = pd.DataFrame(rows); PI.to_csv(f"{R8.OUT_IMP}/permutation_importance.csv", index=False)
    summ = {}
    for gname in GROUPS_ALL:
        s = PI[PI.group == gname].set_index("unit").importance
        summ[gname] = dict(median_six=float(np.nanmedian(s[list(R8.LOW_AMP)])), median_ten=float(np.nanmedian(s[list(R8.OTHER)])), median_all16=float(np.nanmedian(s[units])), per_unit=s.to_dict())
    R8.dump(dict(groups=GROUPS_ALL, n_perm=R8.N_PERM, seed=R8.SEED, rng_order="units (units-v1 order) > groups > permutations > pairs (sorted), one default_rng(20261009) stream", baseline_rho=base, summary=summ, rows=rows), f"{R8.OUT_IMP}/permutation_importance.json")
    # ---- (4) drop-group refit (edge and validity), frozen nested LOUO
    drop = GROUPS_ALL["edge_validity"]; cols = [c for c in B.ALL if c not in drop]; t1 = time.time()
    Xd = dev_T.loc[ok, cols].values
    pred_d, best = B.nested_louo(Xd, y, g, "reg", units, workers, pr=prs, elig=EP)
    P["GBT_ship_dropEV_A_fine"] = np.nan; P.loc[ok, "GBT_ship_dropEV_A_fine"] = 10 ** pred_d
    P[["unit", "pair", "tile_id", "block_id", "A_fine", "GBT_ship_A_fine", "GBT_ship_dropEV_A_fine"]].to_parquet(f"{R8.OUT_IMP}/pred_tile_dropgroup.parquet")
    R8.dump(dict(dropped=drop, n_features=len(cols), configs=best, seconds=time.time() - t1), f"{R8.OUT_IMP}/chosen_configs_dropgroup.json")
    S = {c: E.score_one(P, c, "A_fine", "tile", units, elig_pairs=EP) for c in ("GBT_ship_dropEV_A_fine", "GBT_ship_A_fine")}
    cmp = {name: E.compare(P, "GBT_ship_dropEV_A_fine", "GBT_ship_A_fine", "A_fine", "tile", uu, metric="rho", elig_pairs=EP) for name, uu in (("all16", units), ("six", list(R8.LOW_AMP)), ("ten", list(R8.OTHER)))}
    ci = {name: E.median_ci(P, "GBT_ship_dropEV_A_fine", "A_fine", "tile", uu, metric="rho", elig_pairs=EP) for name, uu in (("all16", units), ("six", list(R8.LOW_AMP)), ("ten", list(R8.OTHER)))}
    ci_ref = {name: E.median_ci(P, "GBT_ship_A_fine", "A_fine", "tile", uu, metric="rho", elig_pairs=EP) for name, uu in (("all16", units), ("six", list(R8.LOW_AMP)), ("ten", list(R8.OTHER)))}
    per_unit = {u: dict(dropEV=S["GBT_ship_dropEV_A_fine"]["per_unit"][u].get("rho", np.nan), GBT_ship=S["GBT_ship_A_fine"]["per_unit"][u].get("rho", np.nan), low_amplitude=(u in R8.LOW_AMP)) for u in units}
    for u in units: per_unit[u]["diff"] = per_unit[u]["dropEV"] - per_unit[u]["GBT_ship"]
    R8.dump(dict(dropped=drop, per_unit=per_unit, median_ci_dropEV=ci, median_ci_GBT_ship=ci_ref, comparisons=cmp, seconds=time.time() - t1), f"{R8.OUT_IMP}/dropgroup_scores.json")
    print(f"drop-group refit: all16 {ci['all16']['value']:.3f} vs GBT_ship {ci_ref['all16']['value']:.3f}; diff " + "; ".join(f"{k} {v['diff_median']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] {v['units_a_better']}/{v['units_b_better']}" for k, v in cmp.items()) + f" ({time.time()-t1:.0f} s)", flush=True)
    print(f"done ({time.time()-t0:.0f} s)")
