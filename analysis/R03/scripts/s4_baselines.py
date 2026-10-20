"""R03 §4.3–4.4 — strongest free baseline: context-size dependence, ŝ_fine, expanded feature-bank GBT (nested LOUO,
Cal DIG capped to 200 training rows as ONE unit), permutation importance; T33-style tables on all 15 units and on the
7 native units with σ_m (mosaic); Δ tables with bootstrap CIs; CI width at N=7 vs N=15."""
import sys, os, json, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C, r03_metrics as M
OUT = f"{C.ROOT}/s4"; F = pd.read_parquet(f"{OUT}/features_all.parquet"); U15 = list(C.ALL15); U7 = list(C.NATIVE7); CTX = ["a", "b", "c", "d"]
pt = pd.read_parquet(f"{C.R02}/eval/pertile.parquet"); pt = pt[pt["mode"] == "mosaic"].set_index("key")
F["sigma_m"] = F.key.map(pt.sigma_m); F["T_ship_mosaic"] = F.key.map(pt.T_ship); F["log_depth_over_lr"] = np.log10(np.abs(F.depth) / F.lr_native)
META_A = ["slope_a_hrgrid", "dist_edge", "frac_valid", "depth", "lr_native"]; R3_A = ["C_a_hrgrid", "curv_a_hrgrid", "aniso_a_hrgrid"]      # exact R02 definitions (tile, HR grid)
def bank(ctx): return [f"{k}_{ctx}" for k in ("C", "curv", "aniso", "slope", "std_ship", "slope_ship_mean", "slope_ship_p90", "curv_ship_std", "beta", "logA", "fit_r2", "aniso_spec", "noise_excess", "shat_fine")] + ["s_lr", "dist_edge", "frac_valid", "depth", "lr_native", "log_depth_over_lr", f"ctx_{ctx}_valid_frac"]
TARGETS = ["T_ship", "T_res", "T_fine", "A_fine"]; CAP = {"CalDIG": 200}
def run_block(args):
    tgt, units_key = args; units = U15 if units_key == "all15" else U7
    d = F[F.unit.isin(units)].dropna(subset=[tgt] + R3_A + META_A).reset_index(drop=True); g = d.unit.values; y = d[tgt].values; P = {}; imp = {}; ctxinfo = {}
    P["roughness-3 (R02, tile)"] = M.nested_linear(d, R3_A, tgt, units, g); P["metadata GBT (R02, tile)"] = M.nested_gbt(d, META_A, tgt, units, g, train_cap=CAP)
    P["R02 free combined GBT (tile)"] = M.nested_gbt(d, R3_A + META_A, tgt, units, g, train_cap=CAP)
    for ctx in CTX:
        dd = d.copy(); cols = bank(ctx); ok = np.isfinite(dd[cols].values.astype(float)).all(1)
        ctxinfo[ctx] = dict(n_ok=int(ok.sum()), n=int(len(dd)), cells_median={u: float(dd[dd.unit == u][f"ctx_{ctx}_cells"].median()) for u in units}, valid_frac_median=float(dd[f"ctx_{ctx}_valid_frac"].median()))
        sh = dd[f"shat_fine_{ctx}"].values.astype(float); P[f"ŝ_fine raw ({ctx})"] = np.where(ok, sh, np.nan)
        cal = np.full(len(dd), np.nan); cal[ok] = M.nested_linear(dd[ok].reset_index(drop=True), [f"shat_fine_{ctx}"], tgt, units, g[ok], log_cols=(f"shat_fine_{ctx}",)); P[f"ŝ_fine calibrated ({ctx})"] = cal
        P[f"roughness-3 ({ctx})"] = np.where(ok, M.nested_linear(dd.fillna(0), [f"C_{ctx}", f"curv_{ctx}", f"aniso_{ctx}"], tgt, units, g, log_cols=(f"C_{ctx}",)), np.nan)
        pr = np.full(len(dd), np.nan); pr[ok], models = M.nested_gbt_models(dd[ok].reset_index(drop=True), cols, tgt, units, g[ok], train_cap=CAP); P[f"expanded bank GBT ({ctx})"] = pr
        imp[ctx] = M.permutation_importance(dd[ok].reset_index(drop=True), cols, tgt, units, g[ok], models)
    if units_key == "native7": P["σ_m (mosaic)"] = d.sigma_m.values.astype(float)
    S = M.score(P, y, g, units)
    best = max(CTX, key=lambda c: S[f"expanded bank GBT ({c})"]["within_r2"])
    comps = [(f"expanded bank GBT ({c})", "R02 free combined GBT (tile)") for c in CTX] + [(f"ŝ_fine raw ({c})", "roughness-3 (R02, tile)") for c in CTX] + [(f"ŝ_fine calibrated ({c})", "roughness-3 (R02, tile)") for c in CTX] + [(f"expanded bank GBT ({best})", "expanded bank GBT (a)")]
    if units_key == "native7": comps += [("σ_m (mosaic)", f"expanded bank GBT ({c})") for c in CTX] + [("σ_m (mosaic)", "R02 free combined GBT (tile)"), ("σ_m (mosaic)", "roughness-3 (R02, tile)")]
    import pickle; pickle.dump(dict(P=P, y=y, g=g, imp=imp, ctxinfo=ctxinfo, keys=d.key.values), open(f"{OUT}/preds_{tgt}_{units_key}.pkl", "wb"))
    S, Cm = M.compare(P, y, g, units, comps)
    return dict(target=tgt, units=units_key, n=int(len(d)), scores=S, comparisons=Cm, importance=imp, best_ctx=best, ctxinfo=ctxinfo, units_list=units)
if __name__ == "__main__":
    jobs = [(t, k) for t in TARGETS for k in ("all15", "native7")]
    with Pool(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))) as pool: results = pool.map(run_block, jobs)
    json.dump(results, open(f"{OUT}/s4_results.json", "w"), indent=1, default=float); md = []
    for r in results:
        U = r["units_list"]; md += [f"\n### §4.4 {r['target']} — {r['units']} ({r['n']} tiles, {len(U)} units); best context for the expanded GBT: ({r['best_ctx']})\n", M.md_pred_table(r["scores"], U), "\nDifferences (bootstrap over units, 2,000 resamples):\n", M.md_cmp_table(r["comparisons"])]
        md.append("\nPermutation importance (median rank over folds; top 8) per context: " + " | ".join(f"({c}) " + ", ".join(f"{k.replace('_'+c,'')}:{v['median_rank']:.0f}" for k, v in sorted(r["importance"][c].items(), key=lambda kv: kv[1]["median_rank"])[:8]) for c in CTX))
        md.append("\nContext windows (median cells per unit): " + "; ".join(f"({c}) " + ", ".join(f"{u} {r['ctxinfo'][c]['cells_median'][u]:.0f}" for u in U) for c in CTX))
    open(f"{OUT}/s4_tables.md", "w").write("\n".join(md)); print("wrote", OUT, flush=True)
