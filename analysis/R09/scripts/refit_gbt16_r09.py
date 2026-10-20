"""R09 §B2.2 — the GBT_ship all-16-unit model. R05 scored EW0207 with an all-16 model fitted in memory (configuration =
the most frequent per-fold choice) and did not persist it. Per the directive, refit ONCE with the frozen bar code on the
16 development units: configuration chosen by the frozen inner criterion (bar_v1_1 inner loop: leave-one-unit-out over
the 16 units for every grid configuration, unit value = eval_v1_1 within-pair rho median over the unit's eligible pairs,
criterion = median over the 16 units; ties -> lowest grid index, as nested_louo), then one fit on the capped rows of all 16
units (cap_mask, seed 20261001). Persisted with joblib; sha256 recorded. The model is also applied to the R05 EW0207 feature
rows and compared with R05's EW0207 predictions (equal iff the chosen configuration is R05's mode configuration)."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, joblib, sklearn
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r09_common as R9, r04_common as C4, bar_v1_1 as B, eval_v1_1 as E
D = {}

def task(args):
    ci, u = args; X, y, g, cm = D["X"], D["y"], D["g"], D["cm"]; tr = cm & (g != u); te = g == u
    if tr.sum() < 10 or te.sum() < 3: return ci, u, np.nan
    mdl = B.make(B.GRID[ci], "reg").fit(X[tr], y[tr]); return ci, u, B._score_unit(mdl.predict(X[te]), y[te], D["pr"][te], "reg", D["elig"], D["elig"])

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1"); workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)); t0 = time.time(); m = R9.start("refit_gbt16_r09"); units = list(C4.UNITS)
    P, rules = R9.R8.load_pred_tile(); EP = rules["eligible"]["A_fine"]
    tiles = R9.R8.load_tiles(); blocks = pd.read_parquet(R9.R8.R05_BLOCKS); feats = R9.R8.load_features(); Bt, Tt = B.assemble(tiles, blocks, feats)
    dev_T = Tt[Tt.unit.isin(units)].reset_index(drop=True); assert (dev_T.tile_id.values == P.tile_id.values).all()
    ok = np.isfinite(dev_T["A_fine"].values) & (dev_T["A_fine"].values > 0); X = dev_T.loc[ok, B.ALL].values.astype(np.float32); y = np.log10(dev_T.loc[ok, "A_fine"].values + B.EPS); g = dev_T.unit.values[ok]; pr = dev_T.pair.values[ok]
    D.update(X=X, y=y, g=g, cm=B.cap_mask(g), pr=pr, elig=set(EP))
    with Pool(workers) as pool: res = pool.map(task, [(ci, u) for ci in range(len(B.GRID)) for u in units], chunksize=1)
    sc = {}
    for ci, u, s in res: sc.setdefault(ci, {})[u] = s
    med = {ci: (float(np.nanmedian(list(v.values()))) if np.isfinite(list(v.values())).any() else np.nan) for ci, v in sc.items()}
    order = [ci for ci in range(len(B.GRID)) if np.isfinite(med[ci])]; best = max(order, key=lambda ci: med[ci])
    mdl = B.make(B.GRID[best], "reg").fit(X[D["cm"]], y[D["cm"]])
    os.makedirs(R9.OUT_ART, exist_ok=True); path = f"{R9.OUT_ART}/gbt_ship_all16_A_fine_tile.joblib"; joblib.dump(mdl, path); sha = R9.R8.sha256_file(path)
    # R05 EW0207 comparison
    chosen = json.load(open(R9.R8.CHOSEN_TILE))["tile/A_fine/GBT_ship"]; idx = [v["cfg_idx"] for v in chosen.values()]; r05_mode = max(set(idx), key=idx.count)
    EW = pd.read_parquet(R9.R8.PRED_TILE_EW_R05); rows = Tt[Tt.unit == "lu16"].reset_index(drop=True); assert (rows.tile_id.values == EW.tile_id.values).all()
    pe = 10 ** mdl.predict(rows[B.ALL].values); diff = float(np.max(np.abs(pe - EW.GBT_ship_A_fine.values)))
    rec = dict(artifact=path, sha256=sha, config=B.GRID[best], cfg_idx=best, inner_criterion_median_rho=med[best], inner_scores_by_cfg={str(ci): med[ci] for ci in range(len(B.GRID))}, inner_per_unit_best={u: sc[best][u] for u in units},
               n_train_rows=int(D["cm"].sum()), n_rows_total=int(ok.sum()), cap=B.CAP, seed=B.SEED, features=B.ALL, n_features=len(B.ALL), sklearn_version=sklearn.__version__, target="log10(A_fine + 1e-9), tile scale",
               r05_ew0207_model=dict(cfg_idx_mode=r05_mode, cfg=B.GRID[r05_mode], mode_count=idx.count(r05_mode), same_as_inner_choice=(r05_mode == best), max_abs_diff_vs_r05_ew0207_pred_m=diff), seconds=time.time() - t0)
    R9.R8.dump(rec, f"{R9.OUT_ART}/gbt_ship_all16_record.json"); joblib.dump(dict(model=mdl, record=rec), f"{R9.OUT_ART}/gbt_ship_all16_bundle.joblib")
    print(f"chosen cfg {best} {B.GRID[best]} inner median rho {med[best]:.3f}; R05 EW0207 mode cfg {r05_mode} (count {idx.count(r05_mode)}/16) same={r05_mode == best}; max |diff| vs R05 EW0207 preds {diff:.3e} m; sha256 {sha} ({time.time()-t0:.0f} s)")
