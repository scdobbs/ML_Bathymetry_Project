"""R08 §1 — within-site upper reference. A multi-feature GBT (HistGradientBoostingRegressor, fixed configuration, no tuning)
fit WITHIN each pair on the frozen GBT_ship feature set (bar_v1_1.ALL; columns that are all-NaN for the pair dropped) and
the log10 A_fine tile target, with spatially blocked 5-fold cross-validation (4 x 4-tile blocks in tile-grid coordinates,
blocks assigned to folds by a seeded permutation, 2-tile Chebyshev buffer removed from training around every test tile),
out-of-fold predictions for every tile. Feasibility: >= 80 tiles, >= 5 blocks, every fold keeps >= 40 training tiles after
the buffer. Scored through eval_v1_1 on the feasible pairs next to held-out GBT_ship, s_lr and V2 (seed-averaged)."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r08_common as R8, r04_common as C4, bar_v1_1 as B, eval_v1_1 as E, targets_v1 as T

D = {}

def folds_for_pair(gi, gj, seed=R8.SEED):
    """Tile-grid coords -> (block id per tile, fold per tile). Blocks of 4 x 4 tiles; sorted unique blocks permuted with
    default_rng(seed) and dealt round-robin into 5 folds (balanced; every fold non-empty when n_blocks >= 5)."""
    bid = np.c_[gi // R8.WS_BLOCK_TILES, gj // R8.WS_BLOCK_TILES]; keys = sorted({(int(a), int(b)) for a, b in bid})
    rng = np.random.default_rng(seed); perm = rng.permutation(len(keys)); fold_of = {keys[perm[i]]: i % R8.WS_FOLDS for i in range(len(keys))}
    return np.array([keys.index((int(a), int(b))) for a, b in bid]), np.array([fold_of[(int(a), int(b))] for a, b in bid]), len(keys)

def one_pair(pid):
    T_ = D["T"]; idx = np.where(T_.pair.values == pid)[0]; d = T_.iloc[idx]
    gi = (d.r0.values // T.STRIDE).astype(int); gj = (d.c0.values // T.STRIDE).astype(int)
    rec = dict(pair=pid, unit=C4.UNIT_OF[pid], n_tiles=int(len(d)), feasible=True, reasons=[])
    if len(d) < R8.WS_MIN_TILES: rec["reasons"].append(f"tiles {len(d)} < {R8.WS_MIN_TILES}")
    blk, fold, nb = folds_for_pair(gi, gj); rec["n_blocks"] = int(nb)
    if nb < R8.WS_MIN_BLOCKS: rec["reasons"].append(f"blocks {nb} < {R8.WS_MIN_BLOCKS}")
    y = np.log10(d.A_fine.values + B.EPS); cols = [c for c in B.ALL if np.isfinite(d[c].values.astype(float)).any()]; X = d[cols].values.astype(float)
    rec["n_features_used"] = len(cols); rec["features_dropped_all_nan"] = [c for c in B.ALL if c not in cols]
    oof = np.full(len(d), np.nan); fold_info = []
    for f in range(R8.WS_FOLDS):
        te = fold == f
        if te.sum() == 0: fold_info.append(dict(fold=f, n_test=0, n_train=0)); rec["reasons"].append(f"fold {f} has no test tile"); continue
        # Chebyshev distance (tile units) from every tile to the nearest test tile
        dist = np.min(np.maximum(np.abs(gi[:, None] - gi[te][None, :]), np.abs(gj[:, None] - gj[te][None, :])), axis=1)
        tr = (~te) & (dist > R8.WS_BUFFER_TILES)
        fold_info.append(dict(fold=f, n_test=int(te.sum()), n_train=int(tr.sum()), n_buffer_removed=int(((~te) & (dist <= R8.WS_BUFFER_TILES)).sum())))
        if tr.sum() < R8.WS_MIN_TRAIN: rec["reasons"].append(f"fold {f} keeps {int(tr.sum())} < {R8.WS_MIN_TRAIN} training tiles after the buffer")
    rec["folds"] = fold_info; rec["feasible"] = not rec["reasons"]
    if rec["feasible"]:
        for f in range(R8.WS_FOLDS):
            te = fold == f; dist = np.min(np.maximum(np.abs(gi[:, None] - gi[te][None, :]), np.abs(gj[:, None] - gj[te][None, :])), axis=1); tr = (~te) & (dist > R8.WS_BUFFER_TILES)
            assert tr.sum() < 10000   # sklearn's early_stopping='auto' would switch on above 10,000 rows; early_stopping=False below makes this explicit
            mdl = HistGradientBoostingRegressor(**R8.WS_HGB, early_stopping=False).fit(X[tr], y[tr]); oof[te] = mdl.predict(X[te])
        assert np.isfinite(oof).all()
    return rec, d.tile_id.values, 10 ** oof

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1"); workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)); t0 = time.time()
    m = R8.start("within_site_r08"); os.makedirs(R8.OUT_WITHIN, exist_ok=True); units = list(C4.UNITS)
    P, rules = R8.load_pred_tile(); EP = rules["eligible"]["A_fine"]
    tiles = R8.load_tiles(); blocks = pd.read_parquet(R8.R05_BLOCKS); feats = R8.load_features()
    Bt, Tt = B.assemble(tiles, blocks, feats); dev_T = Tt[Tt.unit.isin(units)].reset_index(drop=True)
    assert (dev_T.tile_id.values == P.tile_id.values).all()
    dev_T = dev_T.merge(tiles[["tile_id", "r0", "c0"]], on="tile_id", how="left"); assert dev_T.r0.notna().all()
    D["T"] = dev_T; pairs = list(C4.DEV_PAIRS)
    with Pool(min(workers, len(pairs))) as pool: outs = pool.map(one_pair, pairs)
    P["within_site"] = np.nan; recs = []
    for rec, tids, pred in outs:
        recs.append(rec)
        if rec["feasible"]: P.loc[P.tile_id.isin(tids), "within_site"] = pd.Series(pred, index=tids).reindex(P.tile_id[P.tile_id.isin(tids)]).values
    feas = [r["pair"] for r in recs if r["feasible"]]; infeas = [dict(pair=r["pair"], unit=r["unit"], n_tiles=r["n_tiles"], n_blocks=r.get("n_blocks"), reasons=r["reasons"]) for r in recs if not r["feasible"]]
    elig = [p for p in EP if p in set(feas)]; feas_units = [u for u in units if any(C4.UNIT_OF[p] == u for p in elig)]
    R8.dump(dict(feasible_pairs=feas, infeasible=infeas, eligible_and_feasible=elig, feasible_units=feas_units, units_without_feasible_pair=[u for u in units if u not in feas_units], records=recs,
                 config=dict(hgb=R8.WS_HGB, early_stopping=False, block_tiles=R8.WS_BLOCK_TILES, folds=R8.WS_FOLDS, buffer_tiles=R8.WS_BUFFER_TILES, min_tiles=R8.WS_MIN_TILES, min_blocks=R8.WS_MIN_BLOCKS, min_train=R8.WS_MIN_TRAIN, fold_assignment="sorted blocks permuted with default_rng(20261009) per pair, dealt round-robin into 5 folds")),
            f"{R8.OUT_WITHIN}/feasibility.json")
    P[["unit", "pair", "tile_id", "block_id", "A_fine", "within_site", "GBT_ship_A_fine", "ref_s_lr", "V2_seedavg"]].to_parquet(f"{R8.OUT_WITHIN}/pred_tile_within_site.parquet")
    print(f"feasible pairs {len(feas)}/38 ({len(infeas)} infeasible: {[(r['pair'], r['reasons']) for r in infeas]}); feasible units {len(feas_units)}/16 ({time.time()-t0:.0f} s)", flush=True)
    # ---- scoring on the same feasible pairs and tiles
    COLS = ["within_site", "GBT_ship_A_fine", "ref_s_lr", "V2_seedavg"]
    S = {c: E.score_one(P, c, "A_fine", "tile", units, elig_pairs=elig) for c in COLS}
    sets = dict(all=feas_units, six=[u for u in feas_units if u in R8.LOW_AMP], ten=[u for u in feas_units if u in R8.OTHER])
    CI = {name: {c: E.median_ci(P, c, "A_fine", "tile", uu, metric="rho", elig_pairs=elig) for c in COLS} for name, uu in sets.items()}
    CMP = {name: {c: E.compare(P, "within_site", c, "A_fine", "tile", uu, metric="rho", elig_pairs=elig) for c in COLS[1:]} for name, uu in sets.items()}
    per_unit = []
    for u in units:
        fp = [p for p in elig if C4.UNIT_OF[p] == u]; r = dict(unit=u, low_amplitude=(u in R8.LOW_AMP), n_pairs_total=len(C4.UNITS[u]), n_pairs_feasible=len(fp), n_tiles_used=int(P[P.pair.isin(fp)].shape[0]))
        for c in COLS: r[c] = S[c]["per_unit"][u].get("rho", np.nan)
        r["within_minus_GBT_ship"] = r["within_site"] - r["GBT_ship_A_fine"]; per_unit.append(r)
    pd.DataFrame(per_unit).to_csv(f"{R8.OUT_WITHIN}/per_unit.csv", index=False)
    R8.dump(dict(columns=COLS, unit_sets=sets, median_ci=CI, comparisons_within_minus=CMP, per_unit=per_unit, scores=S), f"{R8.OUT_WITHIN}/scores.json")
    for name in sets: print(f"  [{name}] " + "; ".join(f"{c} {CI[name][c]['value']:.3f} [{CI[name][c]['ci95'][0]:.3f}, {CI[name][c]['ci95'][1]:.3f}] n={CI[name][c]['n_units']}" for c in COLS) + f" | within - GBT_ship {CMP[name]['GBT_ship_A_fine']['diff_median']:+.3f} [{CMP[name]['GBT_ship_A_fine']['ci95'][0]:+.3f}, {CMP[name]['GBT_ship_A_fine']['ci95'][1]:+.3f}] {CMP[name]['GBT_ship_A_fine']['units_a_better']}/{CMP[name]['GBT_ship_A_fine']['units_b_better']}", flush=True)
    print(f"done ({time.time()-t0:.0f} s)")
