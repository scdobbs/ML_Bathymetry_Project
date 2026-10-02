"""bar_v1 — FROZEN bar (directive R04 §3.2-3.3; tag phase0-freeze): the strongest free (ship-only) predictor of the
frozen targets at block and tile scale.

Models (sklearn HistGradientBoosting; NaN handled natively; OMP threads = 1):
  GBT_ship   nested leave-one-unit-out over the development units with inner leave-one-unit-out tuning over the grid
             max_depth {2,3,4,6} x n_estimators {100,300,600} x learning_rate {0.03,0.1} x min_child_samples {20,100};
             inner criterion = median over inner held-out units of Spearman rho (regressors) / AUPRC (E_1m classifier);
             one model per target and scale; target = log10 value; E_1m = classifier P(E_1m > 0) + regressor on
             log10 E_1m where positive, combined as E_hat = P x 10^reg (hurdle expectation); all features.
  GBT_noship identical without the contract-v2.1 ship features.
  Reference rows (no fitting): roughness C alone (C_hr_128), s_lr alone (s_lr_256, the R02 tile statistic),
             ship std at 128 px alone (s_lr_128), median ship_rsd at 128 px alone (rsd_med_128; usable units only).
  R02 sigma_m (native 7 units, mean sigma over the block / tile) for reference only.
Training rows: each unit capped at CAP = 200 rows per fold by one seeded subsample (SEED), applied identically to all
units in every outer and inner fold; the held-out unit is always scored on all its rows."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r04_common as C, features_v1 as F, eval_v1 as E

SEED = 20261001; CAP = 200; EPS = 1e-9
GRID = [dict(max_depth=d, n_estimators=n, learning_rate=l, min_child_samples=m) for d in (2, 3, 4, 6) for n in (100, 300, 600) for l in (0.03, 0.1) for m in (20, 100)]
META = ["s_lr_256", "depth", "lr_native", "k", "hr_res", "log_depth_over_lr"]
CONTRACT = [f"{k}_{w}" for w in F.WINDOWS_PX for k in F.CONTRACT_KEYS]
ALL = F.feature_columns() + META
NOSHIP = [c for c in ALL if c not in CONTRACT]
REFS = {"ref_C": "C_hr_128", "ref_s_lr": "s_lr_256", "ref_shipstd128": "s_lr_128", "ref_rsd128": "rsd_med_128"}
D = {}

def make(cfg, kind):
    kw = dict(max_depth=cfg["max_depth"], max_iter=cfg["n_estimators"], learning_rate=cfg["learning_rate"], min_samples_leaf=cfg["min_child_samples"], early_stopping=False, random_state=SEED)
    return HistGradientBoostingClassifier(**kw) if kind == "clf" else HistGradientBoostingRegressor(**kw)

def cap_mask(g, seed=SEED, cap=CAP):
    """One seeded subsample of <= cap rows per unit (units in sorted order), reused in every fold."""
    rng = np.random.default_rng(seed); keep = np.zeros(len(g), bool)
    for u in sorted(np.unique(g)):
        idx = np.where(g == u)[0]
        if len(idx) > cap: idx = np.sort(rng.choice(idx, cap, replace=False))
        keep[idx] = True
    return keep

def _score(pred, y, kind):
    if kind == "clf":
        yb = (y > 0).astype(int)
        return float(average_precision_score(yb, pred)) if 0 < yb.sum() < len(yb) and len(yb) >= 5 else np.nan
    if len(y) < 3 or np.std(pred) == 0 or np.std(y) == 0: return np.nan
    return float(spearmanr(pred, y)[0])

def _predict(model, X, kind): return model.predict_proba(X)[:, 1] if kind == "clf" else model.predict(X)

def inner_task(args):
    u, ci = args; X, y, g, cm, kind, units, score_units = D["X"], D["y"], D["g"], D["cm"], D["kind"], D["units"], D["score_units"]; cfg = GRID[ci]; sc = []
    for v in units:
        if v == u or v not in score_units: continue
        tr = cm & (g != u) & (g != v); te = g == v
        if tr.sum() < 10 or te.sum() < 3: continue
        mdl = make(cfg, kind).fit(X[tr], y[tr]); sc.append(_score(_predict(mdl, X[te], kind), D["y_score"][te], kind))
    return u, ci, (float(np.nanmedian(sc)) if len(sc) and np.isfinite(sc).any() else np.nan)

def outer_task(args):
    u, ci = args; X, y, g, cm, kind = D["X"], D["y"], D["g"], D["cm"], D["kind"]; tr = cm & (g != u); te = np.where(g == u)[0]
    if len(te) == 0: return u, te, np.array([])
    mdl = make(GRID[ci], kind).fit(X[tr], y[tr]); return u, te, _predict(mdl, X[te], kind)

def nested_louo(X, y, g, kind, units, workers, y_score=None, score_units=None, grid_idx=None):
    """Returns (pred aligned with X, {unit: dict(cfg, inner_score)})."""
    D.clear(); D.update(X=X.astype(np.float32), y=y, g=g, cm=cap_mask(g), kind=kind, units=list(units), y_score=(y if y_score is None else y_score), score_units=set(score_units or units))
    gi = list(range(len(GRID))) if grid_idx is None else grid_idx
    with Pool(workers) as pool:
        inner = pool.map(inner_task, [(u, ci) for u in units for ci in gi], chunksize=1)
        best = {}
        for u in units:
            sc = {ci: s for uu, ci, s in inner if uu == u}; order = [ci for ci in gi if np.isfinite(sc.get(ci, np.nan))]
            bi = max(order, key=lambda ci: sc[ci]) if order else gi[0]; best[u] = dict(cfg=GRID[bi], cfg_idx=bi, inner_score=sc.get(bi, np.nan), inner_scores={str(ci): sc.get(ci, np.nan) for ci in gi})
        outs = pool.map(outer_task, [(u, best[u]["cfg_idx"]) for u in units])
    pred = np.full(len(X), np.nan)
    for u, te, p in outs:
        if len(te): pred[te] = p
    D.clear(); return pred, best

def assemble(tiles, blocks, feats):
    """Block table (kept blocks) and tile table with targets + features. sigma_m_R02 kept as a reference column."""
    tcols = ["unit", "pair", "tile_id", "A_fine", "T_fine", "E_1m", "n_joint", "frac_joint", "sigma_m_R02"]
    ft = feats[feats.block_id < 0].drop(columns=["unit", "pair"]); fb = feats[feats.block_id >= 0].drop(columns=["unit", "pair"])
    T = tiles[tcols].merge(ft, on="tile_id", how="inner"); T["block_id"] = -1
    B = blocks[blocks.kept][tcols + ["block_id"]].merge(fb, on=["tile_id", "block_id"], how="inner")
    return B.reset_index(drop=True), T.reset_index(drop=True)

def run_scale(tab, scale, units, workers, grid_idx=None, log=print):
    """Fit every model on one scale table; returns the prediction table and the chosen configs."""
    P = tab[["unit", "pair", "tile_id", "block_id", "A_fine", "T_fine", "E_1m", "sigma_m_R02"]].copy(); chosen = {}
    for k, c in REFS.items(): P[k] = tab[c].values.astype(float)
    g = tab.unit.values
    for tgt in ("A_fine", "T_fine"):
        ok = np.isfinite(tab[tgt].values) & (tab[tgt].values > 0)
        for name, cols in (("GBT_ship", ALL), ("GBT_noship", NOSHIP)):
            t0 = time.time(); X = tab.loc[ok, cols].values; y = np.log10(tab.loc[ok, tgt].values + EPS)
            pred, best = nested_louo(X, y, g[ok], "reg", units, workers, grid_idx=grid_idx)
            col = f"{name}_{tgt}"; P[col] = np.nan; P.loc[ok, col] = 10 ** pred; chosen[f"{scale}/{tgt}/{name}"] = best
            log(f"    [{scale}/{tgt}/{name}] n={ok.sum()} ({time.time()-t0:.0f} s) median inner rho={np.nanmedian([b['inner_score'] for b in best.values()]):.3f}")
    if scale == "block":
        det_units, det_exc, _ = E.detection_units(tab); ok = np.isfinite(tab["E_1m"].values); pos = ok & (tab["E_1m"].values > 0)
        for name, cols in (("GBT_ship", ALL), ("GBT_noship", NOSHIP)):
            t0 = time.time(); X = tab.loc[ok, cols].values; yb = (tab.loc[ok, "E_1m"].values > 0).astype(int)
            pc, bc = nested_louo(X, yb, g[ok], "clf", units, workers, score_units=det_units, grid_idx=grid_idx)
            Xp = tab.loc[pos, cols].values; yp = np.log10(tab.loc[pos, "E_1m"].values)
            pr, br = nested_louo(Xp, yp, g[pos], "reg", units, workers, score_units=det_units, grid_idx=grid_idx)
            # regressor prediction on ALL ok rows uses the per-unit models chosen on positives: refit once per unit on capped positives
            D.clear(); D.update(X=Xp.astype(np.float32), y=yp, g=g[pos], cm=cap_mask(g[pos]), kind="reg")
            reg_all = np.full(ok.sum(), np.nan); gok = g[ok]
            with Pool(workers) as pool: outs = pool.map(_refit_predict_all, [(u, br[u]["cfg_idx"], X[gok == u]) for u in units])
            for u, p in outs: reg_all[gok == u] = p
            D.clear()
            P[f"P_{name}_E_1m"] = np.nan; P.loc[ok, f"P_{name}_E_1m"] = pc; P[f"Epos_{name}_E_1m"] = np.nan; P.loc[pos, f"Epos_{name}_E_1m"] = 10 ** pr
            P[f"{name}_E_1m"] = np.nan; P.loc[ok, f"{name}_E_1m"] = pc * 10 ** reg_all
            chosen[f"{scale}/E_1m/{name}/clf"] = bc; chosen[f"{scale}/E_1m/{name}/reg"] = br
            log(f"    [{scale}/E_1m/{name}] n={ok.sum()} pos={pos.sum()} det_units={len(det_units)} excluded={det_exc} ({time.time()-t0:.0f} s)")
    return P, chosen

def _refit_predict_all(args):
    u, ci, Xu = args; X, y, g, cm = D["X"], D["y"], D["g"], D["cm"]; tr = cm & (g != u)
    if len(Xu) == 0: return u, np.array([])
    return u, make(GRID[ci], "reg").fit(X[tr], y[tr]).predict(Xu)

def evaluate_bar(P, scale, units):
    """All endpoints on one prediction table; comparisons GBT_ship vs GBT_noship / references / R02 sigma_m."""
    res = {}; preds = [c for c in P.columns if c.startswith(("GBT_", "ref_", "P_GBT")) or c == "sigma_m_R02"]
    tgts = [("A_fine", scale), ("T_fine", scale)] + ([("E_1m", "block")] if scale == "block" else [])
    det_units, det_exc, prev = E.detection_units(P) if scale == "block" else (units, {}, {})
    for tgt, sc in tgts:
        uu = [u for u in units if u in det_units] if tgt == "E_1m" else units
        cols = [c for c in preds if (tgt in c) or c.startswith("ref_") or c == "sigma_m_R02"]
        if tgt == "E_1m": cols = [c for c in cols if not any(t in c for t in ("A_fine", "T_fine"))]
        S = {c: E.score_one(P, c, tgt, sc, uu) for c in cols}
        cmp = []; a = f"GBT_ship_{tgt}"; b = f"GBT_noship_{tgt}"
        metrics = ["rho", "capture20"] + (["r2_loglog"] if tgt != "E_1m" else ["auprc"])
        for mtr in metrics:
            if mtr == "auprc": cmp.append(E.compare(P, f"P_GBT_ship_E_1m", f"P_GBT_noship_E_1m", tgt, sc, uu, metric="auprc"))
            else:
                cmp.append(E.compare(P, a, b, tgt, sc, uu, metric=mtr))
                for r in REFS: cmp.append(E.compare(P, a, r, tgt, sc, uu, metric=mtr))
                nat = [u for u in uu if u in C.NATIVE7]; cmp.append(E.compare(P, a, "sigma_m_R02", tgt, sc, nat, metric=mtr))
        ci = E.median_ci(P, a, tgt, sc, uu, metric="rho")
        res[f"{tgt}/{sc}"] = dict(units=uu, scores=S, comparisons=cmp, primary_ci=ci, detection_excluded=det_exc if tgt == "E_1m" else None, e1m_prevalence=prev if tgt == "E_1m" else None)
    return res

def md_block(res, key, units):
    r = res[key]; L = [f"\n### {key} ({len(r['units'])} units)\n", E.md_scores(r["scores"], r["units"]), "", f"GBT_ship median rho = {r['primary_ci']['value']:.3f} [95% CI {r['primary_ci']['ci95'][0]:.3f}, {r['primary_ci']['ci95'][1]:.3f}] (bootstrap over units)", "", E.md_compare(r["comparisons"])]
    if r.get("detection_excluded") is not None: L.append(f"\nUnits excluded from detection metrics (prevalence of E_1m > 0 at block scale outside [5 %, 95 %]): {r['detection_excluded']}; prevalence per unit: " + ", ".join(f"{u} {p:.2f}" for u, p in sorted(r["e1m_prevalence"].items())))
    return "\n".join(L)
