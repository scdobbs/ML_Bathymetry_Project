"""R03 metric machinery = R02 evaluate.py's (within-unit log-log R2, per-unit Spearman, top-20% capture, nested-LOUO
linear / GBT with task5's gbt_fit_predict, bootstrap-over-units CIs), factored so §2, §3 and §4 use one implementation."""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common  # lockbox guard
from scipy.stats import spearmanr
sys.path.insert(0, "/scratch/users/scdobbs/dir44"); import task5_metadata as t5
EPS = 1e-9

def within_r2(x, y, g):
    x = np.log10(np.asarray(x, float) + EPS); y = np.log10(np.asarray(y, float) + EPS); g = np.asarray(g)
    xd = x - pd.Series(x).groupby(g).transform("mean").values; yd = y - pd.Series(y).groupby(g).transform("mean").values
    b = np.sum(xd * yd) / max(np.sum(xd * xd), EPS); return float(1 - np.sum((yd - b * xd) ** 2) / max(np.sum(yd ** 2), EPS))

def capture20(pred, err, g):
    out = {}
    for u in np.unique(g):
        s = g == u; p = np.asarray(pred)[s]; e = np.asarray(err)[s] ** 2
        k = max(1, int(round(0.2 * s.sum()))); top = np.argsort(p)[::-1][:k]; out[str(u)] = float(e[top].sum() / (e.sum() + EPS))
    return out

def per_unit_rho(pred, err, g):
    return {str(u): float(spearmanr(np.asarray(pred)[g == u], np.asarray(err)[g == u])[0]) for u in np.unique(g)}

def per_unit_r2(pred, err, g):
    """log-log r^2 inside each unit (single-unit within_r2)."""
    return {str(u): within_r2(np.asarray(pred)[g == u], np.asarray(err)[g == u], np.asarray(g)[g == u]) for u in np.unique(g)}

def _logcols(X, cols, log_cols):
    X = X.copy()
    for i, c in enumerate(cols):
        if c in log_cols: X[:, i] = np.log10(np.abs(X[:, i]) + EPS)
    return X

def nested_linear(d, cols, target, units, g, log_cols=("C", "s_lr", "sigma_m", "sigma_norm", "shat_fine", "A_spec", "depth", "std_ship")):
    """LOUO linear on standardized features (log for the listed cols), log target. Prediction in target units."""
    y = np.log10(d[target].values + EPS); X = _logcols(d[cols].values.astype(float), cols, log_cols); pred = np.full(len(d), np.nan)
    for u in units:
        tr = g != u; te = g == u
        mu = X[tr].mean(0); sd = X[tr].std(0) + EPS; A = np.c_[(X - mu) / sd, np.ones(len(X))]
        beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None); pred[te] = 10 ** (A[te] @ beta)
    return pred

def nested_gbt(d, cols, target, units, g, train_cap=None, seed=0):
    """LOUO GBT (task5 protocol: 150 trees, lr 0.06, depth 3), log target. train_cap: {unit: n} caps a unit's rows in
    the TRAINING set only (D2: Cal DIG as one unit), uniform subsample."""
    rng = np.random.default_rng(seed); pred = np.full(len(d), np.nan); Xall = d[cols].values.astype(float); yall = np.log10(d[target].values + EPS)
    for u in units:
        tr = np.where(g != u)[0]; te = np.where(g == u)[0]
        if train_cap:
            keep = []
            for v in np.unique(g[tr]):
                idx = tr[g[tr] == v]
                if v in train_cap and len(idx) > train_cap[v]: idx = np.sort(rng.choice(idx, train_cap[v], replace=False))
                keep.append(idx)
            tr = np.concatenate(keep)
        pred[te] = 10 ** t5.gbt_fit_predict(Xall[tr], yall[tr], Xall[te])
    return pred

def score(P, y, g, units):
    res = {}
    for name, p in P.items():
        ok = np.isfinite(p)
        r = dict(within_r2=within_r2(p[ok], y[ok], g[ok]), rho=per_unit_rho(p[ok], y[ok], g[ok]), capture20=capture20(p[ok], y[ok], g[ok]), r2_unit=per_unit_r2(p[ok], y[ok], g[ok]))
        r["rho_median"] = float(np.median(list(r["rho"].values()))); r["capture20_median"] = float(np.median(list(r["capture20"].values()))); r["n"] = int(ok.sum())
        res[name] = r
    return res

def compare(P, y, g, units, comps, boot=2000, seed=0):
    """Bootstrap over units (resample units with replacement, concatenate copies) of the difference in within R2,
    median rho and median capture between predictor a and b."""
    rng = np.random.default_rng(seed); ulist = list(units)
    def stat(a, b, idx):
        pa, pb, yy, gg = [], [], [], []
        for n, i in enumerate(idx):
            s = (g == ulist[i]) & np.isfinite(P[a]) & np.isfinite(P[b]); pa.append(P[a][s]); pb.append(P[b][s]); yy.append(y[s]); gg.append(np.full(s.sum(), n))
        pa, pb, yy, gg = map(np.concatenate, (pa, pb, yy, gg))
        return (within_r2(pa, yy, gg) - within_r2(pb, yy, gg), np.median(list(per_unit_rho(pa, yy, gg).values())) - np.median(list(per_unit_rho(pb, yy, gg).values())),
                np.median(list(capture20(pa, yy, gg).values())) - np.median(list(capture20(pb, yy, gg).values())))
    S = score(P, y, g, units); out = {}
    for a, b in comps:
        obs = stat(a, b, list(range(len(ulist)))); bs = np.array([stat(a, b, rng.integers(0, len(ulist), len(ulist))) for _ in range(boot)])
        out[f"{a} - {b}"] = dict(within_r2=dict(obs=obs[0], ci95=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))]),
                                 rho_median=dict(obs=obs[1], ci95=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))]),
                                 capture20_median=dict(obs=obs[2], ci95=[float(np.percentile(bs[:, 2], 2.5)), float(np.percentile(bs[:, 2], 97.5))]),
                                 units_a_better_rho=int(sum(S[a]["rho"][u] > S[b]["rho"][u] for u in map(str, ulist) if u in S[a]["rho"] and u in S[b]["rho"])), units_compared=int(sum(1 for u in map(str, ulist) if u in S[a]["rho"] and u in S[b]["rho"])),
                                 units_a_better_capture=int(sum(S[a]["capture20"][u] > S[b]["capture20"][u] for u in map(str, ulist) if u in S[a]["capture20"] and u in S[b]["capture20"])), n_units=len(ulist))
    return S, out

def md_pred_table(S, units, names=None):
    names = names or {k: k for k in S}
    L = ["| predictor | within-unit R² | ρ median | ρ per unit " + " / ".join(units) + " | capture20 median | capture per unit |", "|---|---|---|---|---|---|"]
    for k, r in S.items():
        L.append(f"| {names.get(k, k)} | {r['within_r2']:.3f} | {r['rho_median']:.3f} | " + " / ".join(f"{r['rho'].get(u, float('nan')):.2f}" for u in units) + f" | {r['capture20_median']:.3f} | " + " / ".join(f"{r['capture20'].get(u, float('nan')):.2f}" for u in units) + " |")
    return "\n".join(L)

def md_cmp_table(Cm):
    L = ["| comparison | Δ within R² [95% CI] | Δ ρ median [CI] | Δ capture median [CI] | units a better (ρ / capture) of N |", "|---|---|---|---|---|"]
    g = lambda d: f"{d['obs']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    for k, c in Cm.items(): L.append(f"| {k} | {g(c['within_r2'])} | {g(c['rho_median'])} | {g(c['capture20_median'])} | {c['units_a_better_rho']} / {c['units_a_better_capture']} of {c['n_units']} |")
    return "\n".join(L)

# ---- GBT with retained trees (task5 protocol) for permutation importance ----
def gbt_fit(Xtr, ytr, n_est=150, lr=0.06, depth=3):
    F0 = float(ytr.mean()); F = np.full(len(ytr), F0); trees = []; feats = list(range(Xtr.shape[1])); idx_all = np.arange(len(ytr))
    for _ in range(n_est):
        t = t5._fit_tree(Xtr, ytr - F, idx_all, depth, feats); F += lr * t5._pred_tree(t, Xtr); trees.append(t)
    return F0, trees, lr

def gbt_predict(model, X):
    F0, trees, lr = model; p = np.full(len(X), F0)
    for t in trees: p += lr * t5._pred_tree(t, X)
    return p

def nested_gbt_models(d, cols, target, units, g, train_cap=None, seed=0):
    """Like nested_gbt but returns (pred, {unit: model}) so features can be permuted on the held-out unit."""
    rng = np.random.default_rng(seed); pred = np.full(len(d), np.nan); Xall = d[cols].values.astype(float); yall = np.log10(d[target].values + EPS); models = {}
    for u in units:
        tr = np.where(g != u)[0]; te = np.where(g == u)[0]
        if train_cap:
            keep = []
            for v in np.unique(g[tr]):
                idx = tr[g[tr] == v]
                if v in train_cap and len(idx) > train_cap[v]: idx = np.sort(rng.choice(idx, train_cap[v], replace=False))
                keep.append(idx)
            tr = np.concatenate(keep)
        models[u] = gbt_fit(Xall[tr], yall[tr]); pred[te] = 10 ** gbt_predict(models[u], Xall[te])
    return pred, models

def permutation_importance(d, cols, target, units, g, models, seed=0):
    """Per held-out unit: drop in within-unit log-log r2 when each feature is permuted within the unit; ranks per fold; median rank."""
    rng = np.random.default_rng(seed); Xall = d[cols].values.astype(float); y = d[target].values; ranks = {c: [] for c in cols}; drops = {c: [] for c in cols}
    for u in units:
        te = np.where(g == u)[0]; base = within_r2(10 ** gbt_predict(models[u], Xall[te]), y[te], g[te]); dr = []
        for j, c in enumerate(cols):
            Xp = Xall[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j]); dr.append(base - within_r2(10 ** gbt_predict(models[u], Xp), y[te], g[te]))
        order = np.argsort(-np.array(dr))
        for r, j in enumerate(order): ranks[cols[j]].append(r + 1); drops[cols[j]].append(dr[j])
    return {c: dict(median_rank=float(np.median(ranks[c])), median_drop=float(np.median(drops[c]))) for c in cols}
