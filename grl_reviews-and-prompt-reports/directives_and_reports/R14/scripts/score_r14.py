"""R14 §5 — scoring with the frozen functions (phase4_score.score_pair, boot_ci, blocks_of, pooled_row; eval_v1_1.rho), each pair its
own unit, on the R10b scoring inputs (OAK-verified) with the ship side replaced level by level:
  levels    frozen (the R10b columns untouched: the §5.4 must-hold), native (the R14 re-computation of s_lr / features at the native grid;
            a second check), c8.0 c4.0 c3.0 c2.1 c1.7 (the coarsened grids; a pair is skipped at a level it does not reach);
  s_lr      the level's s_lr on every frozen tile (NaN where undefined); lr_native of the row = lr_target (cells_across follows);
  GBT_ship  the leave-one-unit-out fold model of the pair's unit (gbt_folds_r14.py; trained on native features of the other units) applied
            to the level's feature table (bar_v1_1.ALL columns), 10**pred; both variants (cfg18, frozen_fold_cfg) are scored and kept; the
            `GBT_ship` column used by score_pair is the variant named main_variant in <GBT>/record.json;
  primary   target, tile set, interior flag at D1, A_fine_int and blocks exactly as in R10 (only the ship side changes);
  secondary target rebuilt (targets_r14.py): A_fine, interior flag and A_fine_int at the level's band and D1', frozen B4 view rules;
  CI        per pair: block bootstrap over 4 x 4-tile blocks, 2,000 resamples, default_rng(20261009) per pair (phase4_score.boot_ci) for
            s_lr (inside score_pair) and for GBT_ship (boot_ci called on the same view); pooled: phase4_score.pooled_row, seed 20261013,
            over the pairs scored at the level (and, for reference, the frozen native row over the same pairs).
P-V1 is not scored (its rho is computed inside score_pair and dropped here). Writes <SCORES>/results.json and per_pair.csv."""
import json, os, sys, time
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_common as R
P4, PS, C4, R8 = R.P4, R.PS, R.C4, R.R8
import bar_v1_1 as B, eval_v1_1 as E
VARIANTS = ("cfg18", "frozen_fold_cfg"); LEVEL_KEYS = ["frozen", "native"] + [R.tag(c) for c in R.LEVELS]
EXPECT = dict(n_pass=9, n_pairs=14, mean=0.392, ci=(0.317, 0.430))          # §5.4 (R10 §6 as recomputed by R11 on the same inputs)

def strip(r): return {k: v for k, v in r.items() if "P-V1" not in k and k != "_view"}

def score_level(analysis, key, pairs, SI, PP, cmap, stats, models, main, TG, dev_min, frozen_units):
    rows, units_slr, units_gbt, counts, scored = [], [], {v: [] for v in VARIANTS}, dict(n_tiles=0, n_undefined=0, n_floor=0, view_n_tiles=0, view_n_undefined=0, view_n_floor=0), []
    for pid in pairs:
        X = SI[SI.pair == pid].copy().reset_index(drop=True); unit = C4.UNIT_OF[pid]; Pf = None
        if key != "frozen":
            p = f"{R.PRED}/{pid}/{key}.parquet"
            if not os.path.exists(p): rows.append(dict(pair=pid, unit=unit, analysis=analysis, level=key, skipped=True, note="lr_target <= lr_native")); continue
            Pf = pd.read_parquet(p).set_index("tile_id").reindex(X.tile_id.values); X["s_lr"] = Pf.s_lr.values; X["lr_native"] = Pf.lr_native.values
            for v in VARIANTS: X[f"GBT_ship__{v}"] = 10 ** models[v][unit].predict(Pf[B.ALL].values.astype(np.float32))
            X["GBT_ship"] = X[f"GBT_ship__{main}"].values; und = np.isnan(Pf.s_lr.values); flo = (Pf.s_floor_hit.values == True)
            counts["n_tiles"] += len(Pf); counts["n_undefined"] += int(und.sum()); counts["n_floor"] += int(flo.sum())
        else:
            for v in VARIANTS: X[f"GBT_ship__{v}"] = X["GBT_ship"].values
        if analysis == "secondary" and key != "frozen":
            Tg = TG[pid].reindex(X.tile_id.values); X["A_fine"] = Tg[f"A_fine_{key}"].values; X["interior"] = Tg[f"interior_{key}"].values.astype(bool); X["A_fine_int"] = Tg[f"A_fine_int_{key}"].values
        r = PS.score_pair(X, cmap.get(pid, {}), R.B_MIN, ["s_lr", "GBT_ship", "P-V1"], (stats.get(pid) if analysis == "primary" else None), dev_min)
        view, tab, tgt = R.view_table(X); assert view == r["view"], (pid, key, view, r["view"])
        for v in VARIANTS:
            if tab is not None:
                pr = tab[f"GBT_ship__{v}"].values.astype(float); y = tab[tgt].values.astype(float); blk = PS.blocks_of(tab); b = PS.boot_ci(pr, y, blk)
                r.update({f"rho_GBT__{v}": E.rho(pr, y), f"ci_lo_GBT__{v}": b["ci95"][0], f"ci_hi_GBT__{v}": b["ci95"][1], f"ci_estimable_GBT__{v}": b["estimable"], f"lower_gt0_GBT__{v}": bool(b["estimable"] and b["ci95"][0] > 0), f"boot_undefined_GBT__{v}": b["n_undefined"]})
                units_gbt[v].append(dict(pair=pid, pred=pr, target=y, blk=blk))
            else: r.update({f"rho_GBT__{v}": np.nan, f"ci_lo_GBT__{v}": np.nan, f"ci_hi_GBT__{v}": np.nan, f"ci_estimable_GBT__{v}": False, f"lower_gt0_GBT__{v}": False, f"boot_undefined_GBT__{v}": np.nan})
        if "_view" in r: units_slr.append(r["_view"]); scored.append(pid)
        if tab is not None and Pf is not None:
            s = Pf.reindex(tab.tile_id.values); counts["view_n_tiles"] += len(s); counts["view_n_undefined"] += int(np.isnan(s.s_lr.values).sum()); counts["view_n_floor"] += int((s.s_floor_hit.values == True).sum())
        r = strip(r); r.update(analysis=analysis, level=key, skipped=False, gbt_main_variant=main, lower_gt0_s_lr=bool(r["c1_condition"]), n_s_lr_undefined_pair=(int(np.isnan(Pf.s_lr.values).sum()) if Pf is not None else 0), n_s_lr_floor_pair=(int((Pf.s_floor_hit.values == True).sum()) if Pf is not None else 0), n_tiles_pair=int(len(X)))
        rows.append(r)
    pooled = {"s_lr": PS.pooled_row(units_slr)[0], **{f"GBT_ship__{v}": PS.pooled_row(units_gbt[v])[0] for v in VARIANTS}}
    ref = {k: PS.pooled_row([u for u in frozen_units[k] if u["pair"] in scored])[0] for k in frozen_units} if frozen_units else None   # the frozen native row over the same pairs
    sc = [r for r in rows if not r.get("skipped")]
    summ = dict(analysis=analysis, level=key, n_pairs_reached=len(sc), n_pairs_scored=len(scored), pairs_scored=scored, n_not_scorable=int(sum(r["view"] == "not_scorable" for r in sc)), views={r["pair"]: r["view"] for r in sc},
                n_lower_gt0_s_lr=int(sum(r["lower_gt0_s_lr"] for r in sc)), n_ci_estimable_s_lr=int(sum(bool(r["ci_estimable"]) for r in sc)), **{f"n_lower_gt0_GBT__{v}": int(sum(r[f"lower_gt0_GBT__{v}"] for r in sc)) for v in VARIANTS},
                pooled=pooled, pooled_frozen_same_pairs=ref, tile_counts=counts, frac_undefined=(counts["n_undefined"] / counts["n_tiles"] if counts["n_tiles"] else np.nan), frac_floor=(counts["n_floor"] / counts["n_tiles"] if counts["n_tiles"] else np.nan),
                frac_undefined_view=(counts["view_n_undefined"] / counts["view_n_tiles"] if counts["view_n_tiles"] else np.nan), frac_floor_view=(counts["view_n_floor"] / counts["view_n_tiles"] if counts["view_n_tiles"] else np.nan))
    return rows, summ, dict(s_lr=units_slr, **{f"GBT_ship__{v}": units_gbt[v] for v in VARIANTS})

if __name__ == "__main__":
    t0 = time.time(); m, rec0 = R.start("score_r14.py"); os.makedirs(R.SCORES, exist_ok=True); dev_min = P4.dev_min_cells()
    SI, PP, cmap, stats = R.load_r10b(); pairs = R.c1_pairs(PP); assert len(pairs) == 14
    grec = json.load(open(f"{R.GBT}/record.json")); main = grec["main_variant"]; models = {v: {u: joblib.load(grec["variants"][v]["folds"][u]["path"]) for u in grec["units"]} for v in VARIANTS}
    for v in VARIANTS:
        for u in grec["units"]: assert R.sha256_file(grec["variants"][v]["folds"][u]["path"]) == grec["variants"][v]["folds"][u]["sha256"]
    TG = {pid: pd.read_parquet(f"{R.TGT}/{pid}/targets.parquet").set_index("tile_id") for pid in pairs}
    results = dict(main_variant=main, main_variant_note=grec["main_variant_note"], gbt_record=f"{R.GBT}/record.json", levels=LEVEL_KEYS, expect=EXPECT, inputs=dict(inputs_D1=f"{R.OAK10B}/score_inputs/inputs_D1.parquet", sha256=R.sha256_file(f"{R.OAK10B}/score_inputs/inputs_D1.parquet")), analyses={}); allrows = []
    # must-hold first: the frozen level of the primary analysis
    rows, summ, units = score_level("primary", "frozen", pairs, SI, PP, cmap, stats, models, main, TG, dev_min, None); frozen_units = units
    p = summ["pooled"]["s_lr"]; got = dict(n_pass=summ["n_lower_gt0_s_lr"], n_pairs=summ["n_pairs_scored"], mean=round(p["mean_rho"], 3), ci=(round(p["ci95"][0], 3), round(p["ci95"][1], 3)))
    results["must_hold_5_4"] = dict(expected=EXPECT, got=got, passed=bool(got["n_pass"] == EXPECT["n_pass"] and got["n_pairs"] == EXPECT["n_pairs"] and abs(got["mean"] - EXPECT["mean"]) < 5e-4 and abs(got["ci"][0] - EXPECT["ci"][0]) < 5e-4 and abs(got["ci"][1] - EXPECT["ci"][1]) < 5e-4))
    print("must-hold §5.4:", json.dumps(results["must_hold_5_4"], default=float), flush=True)
    if not results["must_hold_5_4"]["passed"]: R.dump(results, f"{R.SCORES}/results.json"); raise SystemExit("STOP (directive §5.4): the native level does not reproduce R10 §6")
    results["analyses"]["primary"] = {"frozen": summ}; allrows += rows
    for analysis in ("primary", "secondary"):
        for key in LEVEL_KEYS:
            if analysis == "primary" and key == "frozen": continue
            t1 = time.time(); rows, summ, _ = score_level(analysis, key, pairs, SI, PP, cmap, stats, models, main, TG, dev_min, frozen_units); results["analyses"].setdefault(analysis, {})[key] = summ; allrows += rows
            ps, pg = summ["pooled"]["s_lr"], summ["pooled"][f"GBT_ship__{main}"]
            print(f"[{analysis:9s} {key:7s}] pairs scored {summ['n_pairs_scored']:2d} (not scorable {summ['n_not_scorable']}); s_lr lower>0 {summ['n_lower_gt0_s_lr']:2d}, pooled {ps['mean_rho']:+.3f} [{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}]; GBT({main}) lower>0 {summ[f'n_lower_gt0_GBT__{main}']:2d}, pooled {pg['mean_rho']:+.3f} [{pg['ci95'][0]:+.3f}, {pg['ci95'][1]:+.3f}]; s_lr undefined {summ['frac_undefined']:.3f} floor {summ['frac_floor']:.3f} ({time.time()-t1:.0f} s)" if ps else f"[{analysis} {key}] nothing scored", flush=True)
    PR = pd.DataFrame(allrows); PR.to_csv(f"{R.SCORES}/per_pair.csv", index=False); results["per_pair_csv"] = dict(path=f"{R.SCORES}/per_pair.csv", sha256=R.sha256_file(f"{R.SCORES}/per_pair.csv")); results.update(node=os.uname().nodename, seconds=time.time() - t0)
    R.dump(results, f"{R.SCORES}/results.json"); print(f"done ({time.time()-t0:.0f} s)", flush=True)
