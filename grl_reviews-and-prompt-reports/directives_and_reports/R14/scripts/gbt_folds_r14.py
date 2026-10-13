"""R14 §2.4 / §4.3 — GBT_ship leave-one-unit-out fold models on NATIVE features. No fold model was persisted (R05 kept predictions and
chosen configurations; R09 kept the all-16 model only), so the folds are refit with the frozen bar code (bar_v1_1.make, cap_mask seed
20261001, the frozen R04 feature table on tiles_v1.1 assembled by bar_v1_1.assemble exactly as run_bar_v1_1 / refit_gbt16_r09 did;
target log10(A_fine + 1e-9); every unit held out in turn, 16 folds), in two variants:
  cfg18            every fold with configuration 18 (the all-16 Phase 4 artifact's configuration), as §4.3 names it;
  frozen_fold_cfg  every fold with the configuration the frozen bar's inner loop chose for that fold (R05 chosen_configs_tile.json),
                   which is how the frozen bar was actually fit (none of the 16 folds chose cfg 18).
Check (§4.3): the native-level predictions of each variant against the frozen held-out GBT_ship predictions (R05 pred_tile, as carried
by the R07 table and the R10b scoring inputs) per tile, and the per-pair rho on the R10 view against R10b's rho_GBT_ship, within 0.01.
Persists the fold models with joblib (sha256 recorded) under <GBT>/<variant>/fold_<unit>.joblib and writes <GBT>/record.json."""
import json, os, sys, time
import numpy as np, pandas as pd, joblib, sklearn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_common as R
C4, R8 = R.C4, R.R8
import bar_v1_1 as B, eval_v1_1 as E
VARIANTS = ("cfg18", "frozen_fold_cfg")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1"); t0 = time.time(); m, rec0 = R.start("gbt_folds_r14.py"); units = list(C4.UNITS)
    P7, rules = R8.load_pred_tile(); tiles = R8.load_tiles(); blocks = pd.read_parquet(R8.R05_BLOCKS); feats = R8.load_features(); Bt, Tt = B.assemble(tiles, blocks, feats)
    dev_T = Tt[Tt.unit.isin(units)].reset_index(drop=True); assert (dev_T.tile_id.values == P7.tile_id.values).all()
    ok = np.isfinite(dev_T["A_fine"].values) & (dev_T["A_fine"].values > 0); X = dev_T.loc[ok, B.ALL].values.astype(np.float32); y = np.log10(dev_T.loc[ok, "A_fine"].values + B.EPS); g = dev_T.unit.values[ok]; cm = B.cap_mask(g); tid = dev_T.tile_id.values[ok]
    chosen = json.load(open(R8.CHOSEN_TILE))["tile/A_fine/GBT_ship"]; cfgs = {"cfg18": {u: R.CFG18 for u in units}, "frozen_fold_cfg": {u: int(chosen[u]["cfg_idx"]) for u in units}}
    SI, PP, cmap, stats = R.load_r10b(); pairs = R.c1_pairs(PP); frozen_pred = P7.set_index("tile_id")["GBT_ship_A_fine"]
    rec = dict(n_rows_total=int(ok.sum()), n_train_rows_capped=int(cm.sum()), cap=B.CAP, seed=B.SEED, features=B.ALL, n_features=len(B.ALL), sklearn_version=sklearn.__version__, target="log10(A_fine + 1e-9), tile scale", units=units, variants={})
    for v in VARIANTS:
        t1 = time.time(); d = f"{R.GBT}/{v}"; os.makedirs(d, exist_ok=True); pred = np.full(len(X), np.nan); files = {}
        for u in units:
            tr = cm & (g != u); te = g == u; mdl = B.make(B.GRID[cfgs[v][u]], "reg").fit(X[tr], y[tr]); p = f"{d}/fold_{u}.joblib"; joblib.dump(mdl, p); files[u] = dict(path=p, sha256=R.sha256_file(p), cfg_idx=cfgs[v][u], cfg=B.GRID[cfgs[v][u]], n_train=int(tr.sum()), n_test=int(te.sum()))
            if te.any(): pred[te] = 10 ** mdl.predict(X[te])
        S = pd.Series(pred, index=tid); fz = frozen_pred.reindex(tid).values; both = np.isfinite(pred) & np.isfinite(fz)
        tile_cmp = dict(n=int(both.sum()), max_abs_diff_m=float(np.max(np.abs(pred[both] - fz[both]))), max_rel_diff=float(np.max(np.abs(pred[both] - fz[both]) / np.abs(fz[both]))), identical_1e9=bool(np.max(np.abs(pred[both] - fz[both])) <= 1e-9))
        per_pair = {}
        for pid in pairs:
            Xp = SI[SI.pair == pid].copy(); view, tab, tgt = R.view_table(Xp)
            if tab is None: per_pair[pid] = dict(view=view); continue
            pr = S.reindex(tab.tile_id.values).values; rho = E.rho(pr, tab[tgt].values.astype(float)); rho_fz = float(PP.loc[pid, "rho_GBT_ship"]); rho_fz_here = E.rho(tab["GBT_ship"].values.astype(float), tab[tgt].values.astype(float))
            per_pair[pid] = dict(unit=C4.UNIT_OF[pid], view=view, n=int(len(tab)), rho_refit=rho, rho_frozen_r10b=rho_fz, rho_frozen_recomputed=rho_fz_here, abs_diff=abs(rho - rho_fz), within_0p01=bool(abs(rho - rho_fz) <= 0.01), cfg_idx_fold=cfgs[v][C4.UNIT_OF[pid]])
        pp = [r for r in per_pair.values() if "abs_diff" in r]
        rec["variants"][v] = dict(folds=files, tile_comparison_vs_frozen=tile_cmp, per_pair=per_pair, n_pairs=len(pp), n_within_0p01=int(sum(r["within_0p01"] for r in pp)), all_within_0p01=bool(all(r["within_0p01"] for r in pp)), max_abs_rho_diff=float(max(r["abs_diff"] for r in pp)), seconds=time.time() - t1)
        pd.DataFrame(dict(tile_id=tid, unit=g, pred=pred, frozen=fz)).to_parquet(f"{d}/native_predictions.parquet", index=False)
        print(f"[{v}] tiles: max |d| {tile_cmp['max_abs_diff_m']:.3e} m vs frozen; per-pair rho within 0.01 on {rec['variants'][v]['n_within_0p01']}/{len(pp)} (max |d rho| {rec['variants'][v]['max_abs_rho_diff']:.4f}) ({time.time()-t1:.0f} s)", flush=True)
        for pid, r in per_pair.items():
            if "abs_diff" in r: print(f"    {pid:40s} {r['view']:9s} cfg {r['cfg_idx_fold']:2d} rho refit {r['rho_refit']:+.4f} frozen {r['rho_frozen_r10b']:+.4f} |d| {r['abs_diff']:.4f} {'ok' if r['within_0p01'] else 'FAIL'}", flush=True)
    passing = [v for v in VARIANTS if rec["variants"][v]["all_within_0p01"]]
    rec["reproduction_bar"] = dict(rule="per-pair rho within 0.01 of R10b rho_GBT_ship on the R10 view, every C1-set pair", passing_variants=passing, directive_variant="cfg18", frozen_bar_fold_cfgs={u: int(chosen[u]["cfg_idx"]) for u in units}, n_frozen_folds_using_cfg18=int(sum(int(chosen[u]["cfg_idx"]) == R.CFG18 for u in units)))
    rec["main_variant"] = ("cfg18" if "cfg18" in passing else (passing[0] if passing else "cfg18")); rec["main_variant_note"] = ("cfg18 passes the §4.3 bar" if "cfg18" in passing else (f"cfg18 fails the §4.3 bar; {passing[0]} passes (HOLD)" if passing else "no variant passes the §4.3 bar (HOLD)"))
    rec.update(node=os.uname().nodename, seconds=time.time() - t0); R.dump(rec, f"{R.GBT}/record.json"); print(json.dumps(rec["reproduction_bar"], indent=1), rec["main_variant_note"], f"({time.time()-t0:.0f} s)")
