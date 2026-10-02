"""R05 §3 reporting tables (run after run_bar_v1_1): (2) v1 vs v1.1 side by side for every predictor on A_fine at tile and
block scale — R04 bar under eval_v1 (from R04 results), R04 predictions re-scored under eval_v1.1 on tiles_v1.1 (isolates
the aggregation change from the refit), and the R05 bar under eval_v1.1; (3) per-pair rho for the multi-pair units;
(4) R02 sigma_m under eval_v1.1 with its provenance label; (5) EW0207 is written by run_bar_v1_1."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common as C5, r04_common as C, eval_v1_1 as E, eval_v1 as E1
m = C5.start("report_tables_v1_1"); OUT = f"{C5.ROOT}/bar"; units = list(C.UNITS); md = []; R = {}
r04 = {sc: json.load(open(f"{C5.R04}/bar/bar_results_{sc}.json")) for sc in ("block", "tile")}
pred4 = {sc: pd.read_parquet(f"{C5.R04}/bar/pred_{sc}.parquet") for sc in ("block", "tile")}
pred5 = {sc: pd.read_parquet(f"{OUT}/pred_{sc}.parquet") for sc in ("block", "tile")}
t11 = pd.read_parquet(f"{C5.ROOT}/tiles/tiles_v1_1.parquet"); keep = set(t11.tile_id)
prov = json.load(open(f"{C5.ROOT}/sigma_m_provenance.json")); rules = json.load(open(f"{OUT}/pair_rules.json")); EP = rules["eligible"]["A_fine"]
preds = ["GBT_ship_A_fine", "GBT_noship_A_fine", "ref_C", "ref_s_lr", "ref_shipstd128", "ref_rsd128", "sigma_m_R02"]
# (2) side by side
md.append("## v1 vs v1.1 side by side — A_fine, median rho over units [95 % CI], every predictor\n")
md.append("| scale | predictor | R04 bar under eval_v1 (tiles_v1) | R04 predictions under eval_v1.1 (tiles_v1.1) | R05 refit under eval_v1.1 (tiles_v1.1) |\n|---|---|---|---|---|")
for sc in ("tile", "block"):
    p4 = pred4[sc][pred4[sc].tile_id.isin(keep)].reset_index(drop=True); p5 = pred5[sc]
    for p in preds:
        uu = [u for u in units if u in C.NATIVE7] if p == "sigma_m_R02" else units
        a = E1.median_ci(pred4[sc], p, "A_fine", sc, uu); b = E.median_ci(p4, p, "A_fine", sc, uu, elig_pairs=EP); c = E.median_ci(p5, p, "A_fine", sc, uu, elig_pairs=EP)
        R[f"{sc}/{p}"] = dict(v1=a, v1_1_on_r04_pred=b, v1_1_refit=c)
        f = lambda x: f"{x['value']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] (n={x['n_units']})"
        md.append(f"| {sc} | {p}{' (LOUO, CONFIRMED)' if p == 'sigma_m_R02' else ''} | {f(a)} | {f(b)} | {f(c)} |")
# (3) per-pair rho, multi-pair units, tile scale, v1.1 refit
md.append("\n## Per-pair Spearman rho (tile scale, A_fine, R05 refit) for the multi-pair units\n")
multi = [u for u in units if len(C.UNITS[u]) > 1]; cols = ["GBT_ship_A_fine", "GBT_noship_A_fine", "ref_C", "ref_s_lr"]
md.append("| unit | pair | k | n tiles | " + " | ".join(cols) + " | unit value (median over pairs): " + " / ".join(cols) + " |\n|---|---|---|---|" + "---|" * len(cols) + "---|")
S = {c: E.score_one(pred5["tile"], c, "A_fine", "tile", units, elig_pairs=EP) for c in cols}; ks = t11.groupby("pair").k.first()
for u in multi:
    for pr in C.UNITS[u]:
        vals = [S[c]["per_pair"][u].get(pr, {}) for c in cols]
        md.append(f"| {u} | {pr} | {int(ks.get(pr, 0))} | {vals[0].get('n', 0)} | " + " | ".join(f"{v.get('rho', np.nan):+.3f}" if v else "ineligible" for v in vals) + " | " + " / ".join(f"{S[c]['per_unit'][u].get('rho', np.nan):+.3f}" for c in cols) + " |")
R["per_pair_multi"] = {u: {pr: {c: S[c]["per_pair"][u].get(pr, {}) for c in cols} for pr in C.UNITS[u]} for u in multi}
# also v1 (unit-pooled) per unit for these predictors, for the "what changed" reading
md.append("\n| unit | " + " | ".join(f"{c}: v1 pooled rho -> v1.1 unit value" for c in cols) + " |\n|---|" + "---|" * len(cols))
for u in multi: md.append(f"| {u} | " + " | ".join(f"{r04['tile']['A_fine/tile']['scores'][c]['per_unit'][u]['rho']:+.3f} -> {S[c]['per_unit'][u].get('rho', np.nan):+.3f}" for c in cols) + " |")
# (4) sigma_m provenance
md.append("\n## R02 sigma_m provenance (§3.4)\n\n| unit | label | held pairs of the fold | n_train (expected under LOUO) | n_held |\n|---|---|---|---|---|")
for u, r in prov.items():
    if u.startswith("_"): continue
    e = r["evidence"]; md.append(f"| {u} | {r['label']} | {', '.join(e['perpatch_held_pairs'])} | {e['n_train']} ({e['n_train_expected_if_LOUO']}) | {e['n_held']} |")
json.dump(R, open(f"{OUT}/report_tables_v1_1.json", "w"), indent=1, default=float); open(f"{OUT}/report_tables_v1_1.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
