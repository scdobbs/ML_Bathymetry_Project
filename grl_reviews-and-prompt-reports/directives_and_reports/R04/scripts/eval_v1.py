"""eval_v1 — FROZEN evaluation (directive R04 §2; tag phase0-freeze).

Input: a table with columns unit, pair, tile_id, block_id (-1 for tile rows), the target column(s) and predictor columns.
Primary endpoint: A_fine at block scale, within-unit Spearman rho, median over the development units.
Secondary: A_fine at tile scale; T_fine at block and tile scale; E_1m at block scale (within-unit AUPRC for E_1m > 0 and
within-unit Spearman rho on E_1m).  Metric set for every endpoint: within-unit Spearman rho, within-unit log-log R^2
(A_fine, T_fine only), top-20 % capture; each per unit, median over units, pooled.
Differences: bootstrap over units (2,000 resamples, seed 20261001, 95 % percentile CI), count of units where the first
predictor is better, exact one-sided sign-test p-value of that count (reported, not a gate).
Weighting: unit-equal (median over units) primary; pooled secondary. The unit set must equal the frozen list (or a
versioned extension); rows for sealed pairs are refused. EW0207 is scored through the same code as a one-unit table."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.metrics import average_precision_score
import r04_common as C

EPS = 1e-9; N_BOOT = 2000; SEED = 20261001; TOP_FRAC = 0.2
TARGETS = {"A_fine": ("block", "tile"), "T_fine": ("block", "tile"), "E_1m": ("block",)}
LOGLOG_TARGETS = ("A_fine", "T_fine")

def assert_units(table, allow_ew0207=False):
    units = set(table.unit.unique()); frozen = set(C.UNITS)
    if allow_ew0207: frozen |= set(C.EW0207)
    extra = units - frozen
    if extra: raise AssertionError(f"eval_v1: unit set {sorted(units)} is not the frozen list {C.UNITS_VERSION} (extra {sorted(extra)})")
    bad = set(table.pair.unique()) & set(C.SEALED)
    if bad: raise PermissionError(f"eval_v1: rows for sealed pairs {sorted(bad)}")
    return sorted(units)

def _ok(p, y): return np.isfinite(p) & np.isfinite(y)

def rho(p, y):
    ok = _ok(p, y)
    if ok.sum() < 3 or np.std(p[ok]) == 0 or np.std(y[ok]) == 0: return np.nan
    return float(spearmanr(p[ok], y[ok])[0])

def loglog_r2(p, y):
    """R^2 of log10 y on log10 p (slope and intercept fitted) within one unit."""
    ok = _ok(p, y) & (p > 0) & (y > 0)
    if ok.sum() < 3: return np.nan
    x = np.log10(p[ok] + EPS); z = np.log10(y[ok] + EPS); xd = x - x.mean(); zd = z - z.mean()
    if np.sum(xd ** 2) == 0 or np.sum(zd ** 2) == 0: return np.nan
    b = np.sum(xd * zd) / np.sum(xd ** 2); return float(1 - np.sum((zd - b * xd) ** 2) / np.sum(zd ** 2))

def within_r2_pooled(p, y, g):
    """R03 within_r2: unit-demeaned log-log, one slope pooled over all rows."""
    ok = _ok(p, y) & (p > 0) & (y > 0); x = np.log10(p[ok] + EPS); z = np.log10(y[ok] + EPS); gg = np.asarray(g)[ok]
    if ok.sum() < 3: return np.nan
    xd = x - pd.Series(x).groupby(gg).transform("mean").values; zd = z - pd.Series(z).groupby(gg).transform("mean").values
    if np.sum(xd ** 2) == 0 or np.sum(zd ** 2) == 0: return np.nan
    b = np.sum(xd * zd) / np.sum(xd ** 2); return float(1 - np.sum((zd - b * xd) ** 2) / np.sum(zd ** 2))

def capture20(p, y):
    """Share of the target's squared value carried by the top 20 % predicted rows."""
    ok = _ok(p, y); p = p[ok]; e = y[ok] ** 2
    if len(p) < 5 or e.sum() == 0: return np.nan
    k = max(1, int(round(TOP_FRAC * len(p)))); top = np.argsort(p)[::-1][:k]; return float(e[top].sum() / e.sum())

def auprc(p, y):
    ok = _ok(p, y); yb = (y[ok] > 0).astype(int)
    if ok.sum() < 5 or yb.sum() == 0 or yb.sum() == len(yb): return np.nan
    return float(average_precision_score(yb, p[ok]))

def detection_units(table, target="E_1m", lo=0.05, hi=0.95):
    """§1.2: units whose block-scale prevalence of E_1m > 0 is in [5 %, 95 %]; returns (kept, excluded_with_prevalence)."""
    b = table[table.block_id >= 0]; prev = b.groupby("unit")[target].apply(lambda s: float(np.mean(s.dropna() > 0)))
    kept = sorted(prev[(prev >= lo) & (prev <= hi)].index); exc = {u: float(prev[u]) for u in prev.index if u not in kept}
    return kept, exc, prev.to_dict()

def score_one(table, pred, target, scale, units=None):
    """Per-unit, median-over-units and pooled metrics for one predictor column on one target at one scale."""
    d = table[table.block_id >= 0] if scale == "block" else table[table.block_id < 0]
    units = units or sorted(d.unit.unique()); per = {}
    for u in units:
        s = d[d.unit == u]; p = s[pred].values.astype(float); y = s[target].values.astype(float)
        r = dict(n=int(_ok(p, y).sum()), rho=rho(p, y), capture20=capture20(p, y))
        if target in LOGLOG_TARGETS: r["r2_loglog"] = loglog_r2(p, y)
        if target == "E_1m": r["auprc"] = auprc(p, y)
        per[u] = r
    keys = [k for k in ("rho", "r2_loglog", "capture20", "auprc") if any(k in v for v in per.values())]
    med = {k: float(np.nanmedian([per[u][k] for u in units if k in per[u]])) for k in keys}
    nunits = {k: int(np.sum(np.isfinite([per[u][k] for u in units if k in per[u]]))) for k in keys}
    p = d[pred].values.astype(float); y = d[target].values.astype(float); g = d.unit.values
    pooled = dict(n=int(_ok(p, y).sum()), rho=rho(p, y), capture20=capture20(p, y))
    if target in LOGLOG_TARGETS: pooled["r2_loglog_within"] = within_r2_pooled(p, y, g)
    if target == "E_1m": pooled["auprc"] = auprc(p, y)
    return dict(pred=pred, target=target, scale=scale, per_unit=per, median=med, n_units_scored=nunits, pooled=pooled)

def compare(table, pred_a, pred_b, target, scale, units=None, metric="rho", n_boot=N_BOOT, seed=SEED):
    """Difference a - b in the median-over-units metric, bootstrap over units (resample units with replacement, recompute
    the median), 95 % percentile CI, units where a is better, exact one-sided sign test (ties dropped)."""
    Sa = score_one(table, pred_a, target, scale, units); Sb = score_one(table, pred_b, target, scale, units)
    units = units or sorted(set(Sa["per_unit"]) & set(Sb["per_unit"]))
    va = np.array([Sa["per_unit"][u].get(metric, np.nan) for u in units]); vb = np.array([Sb["per_unit"][u].get(metric, np.nan) for u in units])
    ok = np.isfinite(va) & np.isfinite(vb); va, vb, uu = va[ok], vb[ok], [u for u, o in zip(units, ok) if o]
    obs = float(np.median(va) - np.median(vb)) if len(va) else np.nan
    rng = np.random.default_rng(seed); n = len(va)
    bs = np.array([np.median(va[i]) - np.median(vb[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]) if n else np.array([np.nan])
    better = int(np.sum(va > vb)); worse = int(np.sum(va < vb)); ties = int(np.sum(va == vb))
    pval = float(binomtest(better, better + worse, 0.5, alternative="greater").pvalue) if better + worse > 0 else np.nan
    return dict(a=pred_a, b=pred_b, target=target, scale=scale, metric=metric, diff_median=obs, ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                units_a_better=better, units_b_better=worse, ties=ties, n_units=n, sign_test_p_one_sided=pval, per_unit_diff={u: float(x - y) for u, x, y in zip(uu, va, vb)})

def median_ci(table, pred, target, scale, units=None, metric="rho", n_boot=N_BOOT, seed=SEED):
    """Bootstrap-over-units 95 % percentile CI of the median-over-units metric for one predictor."""
    S = score_one(table, pred, target, scale, units); units = units or sorted(S["per_unit"])
    v = np.array([S["per_unit"][u].get(metric, np.nan) for u in units]); v = v[np.isfinite(v)]; rng = np.random.default_rng(seed); n = len(v)
    bs = np.array([np.median(v[rng.integers(0, n, n)]) for _ in range(n_boot)]) if n else np.array([np.nan])
    return dict(pred=pred, target=target, scale=scale, metric=metric, value=float(np.median(v)) if n else np.nan, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_units=int(n))

def evaluate(table, preds, allow_ew0207=False, units=None):
    """Score every predictor on every endpoint. Returns {(target, scale): {pred: score}} plus the detection-unit rule."""
    units_all = assert_units(table, allow_ew0207); units = units or units_all
    det_units, det_excluded, prevalence = detection_units(table)
    out = {"units": units, "detection_units": det_units, "detection_excluded": det_excluded, "e1m_prevalence_block": prevalence, "scores": {}}
    for tgt, scales in TARGETS.items():
        for sc in scales:
            uu = [u for u in units if u in det_units] if tgt == "E_1m" else units
            out["scores"][f"{tgt}/{sc}"] = {p: score_one(table, p, tgt, sc, uu) for p in preds if p in table.columns}
    return out

def md_scores(scores, units, metric_keys=("rho", "r2_loglog", "capture20", "auprc")):
    S = scores; keys = [k for k in metric_keys if any(k in s["median"] for s in S.values())]
    L = ["| predictor | n units | " + " | ".join(f"median {k}" for k in keys) + " | pooled rho | pooled within-R² / AUPRC | rho per unit " + " / ".join(units) + " |", "|---|---|" + "---|" * len(keys) + "---|---|---|"]
    for p, s in S.items():
        pr = s["pooled"]; pw = pr.get("r2_loglog_within", pr.get("auprc", np.nan))
        L.append(f"| {p} | {s['n_units_scored'].get('rho', 0)} | " + " | ".join(f"{s['median'].get(k, np.nan):.3f}" for k in keys) + f" | {pr['rho']:.3f} | {pw:.3f} | " + " / ".join(f"{s['per_unit'].get(u, {}).get('rho', np.nan):+.2f}" for u in units) + " |")
    return "\n".join(L)

def md_compare(cmps):
    L = ["| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |", "|---|---|---|---|---|---|"]
    for c in cmps: L.append(f"| {c['a']} − {c['b']} | {c['target']}/{c['scale']} | {c['metric']} | {c['diff_median']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['units_a_better']} / {c['units_b_better']} / {c['ties']} of {c['n_units']} | {c['sign_test_p_one_sided']:.3f} |")
    return "\n".join(L)
