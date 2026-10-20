"""eval_v1_1 — FROZEN evaluation, version 1.1 (directive R05 §2.2; tag phase0-freeze). A copy of R04's eval_v1 with
exactly these changes:
  1. primary endpoint = A_fine at TILE scale (within-pair Spearman rho, aggregated as in 3);
  2. secondary = A_fine block; T_fine tile and block; E_1m block (AUPRC for E_1m > 0, Spearman rho on E_1m);
  3. every metric is computed WITHIN EACH PAIR (log-log R^2 demeaned within the pair); the unit value is the median over
     the unit's eligible pairs; the endpoint is the median of unit values; the bootstrap resamples units (unit values);
     units-better count and sign test compare unit values;
  4. pair eligibility: >= MIN_TILES_PER_PAIR (10) tiles in tiles_v1.1 with finite target and finite prediction; the same
     pair set is used at tile and block scale; a unit with no eligible pair is dropped from that endpoint;
  5. detection rule per PAIR: a pair enters the E_1m metrics only if its block-scale prevalence of E_1m > 0 is within
     [5 %, 95 %]; E_1m threshold 1.0 m unchanged;
  6. pooled metrics kept as in v1 (over all rows), secondary.
Unchanged: bootstrap over units (2,000 resamples, seed 20261001, 95 % percentile CI), units-better count, exact one-sided
sign test (reported, not gated), assert_units / sealed-pair refusal, EW0207 as a separate table."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.metrics import average_precision_score
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common  # sets the R04/R05 script paths
import r04_common as C

EPS = 1e-9; N_BOOT = 2000; SEED = 20261001; TOP_FRAC = 0.2
MIN_TILES_PER_PAIR = 10; DETECTION_BOUNDS = (0.05, 0.95)
TARGETS = {"A_fine": ("tile", "block"), "T_fine": ("tile", "block"), "E_1m": ("block",)}
PRIMARY = ("A_fine", "tile")
LOGLOG_TARGETS = ("A_fine", "T_fine")

def assert_units(table, allow_ew0207=False):
    units = set(table.unit.unique()); frozen = set(C.UNITS)
    if allow_ew0207: frozen |= set(C.EW0207)
    extra = units - frozen
    if extra: raise AssertionError(f"eval_v1_1: unit set {sorted(units)} is not the frozen list {C.UNITS_VERSION} (extra {sorted(extra)})")
    bad = set(table.pair.unique()) & set(C.SEALED)
    if bad: raise PermissionError(f"eval_v1_1: rows for sealed pairs {sorted(bad)}")
    return sorted(units)

def _ok(p, y): return np.isfinite(p) & np.isfinite(y)

def rho(p, y):
    ok = _ok(p, y)
    if ok.sum() < 3 or np.std(p[ok]) == 0 or np.std(y[ok]) == 0: return np.nan
    return float(spearmanr(p[ok], y[ok])[0])

def loglog_r2(p, y):
    """R^2 of log10 y on log10 p (slope and intercept fitted) within one pair."""
    ok = _ok(p, y) & (p > 0) & (y > 0)
    if ok.sum() < 3: return np.nan
    x = np.log10(p[ok] + EPS); z = np.log10(y[ok] + EPS); xd = x - x.mean(); zd = z - z.mean()
    if np.sum(xd ** 2) == 0 or np.sum(zd ** 2) == 0: return np.nan
    b = np.sum(xd * zd) / np.sum(xd ** 2); return float(1 - np.sum((zd - b * xd) ** 2) / np.sum(zd ** 2))

def within_r2_pooled(p, y, g):
    ok = _ok(p, y) & (p > 0) & (y > 0); x = np.log10(p[ok] + EPS); z = np.log10(y[ok] + EPS); gg = np.asarray(g)[ok]
    if ok.sum() < 3: return np.nan
    xd = x - pd.Series(x).groupby(gg).transform("mean").values; zd = z - pd.Series(z).groupby(gg).transform("mean").values
    if np.sum(xd ** 2) == 0 or np.sum(zd ** 2) == 0: return np.nan
    b = np.sum(xd * zd) / np.sum(xd ** 2); return float(1 - np.sum((zd - b * xd) ** 2) / np.sum(zd ** 2))

def capture20(p, y):
    ok = _ok(p, y); p = p[ok]; e = y[ok] ** 2
    if len(p) < 5 or e.sum() == 0: return np.nan
    k = max(1, int(round(TOP_FRAC * len(p)))); top = np.argsort(p)[::-1][:k]; return float(e[top].sum() / e.sum())

def auprc(p, y):
    ok = _ok(p, y); yb = (y[ok] > 0).astype(int)
    if ok.sum() < 5 or yb.sum() == 0 or yb.sum() == len(yb): return np.nan
    return float(average_precision_score(yb, p[ok]))

def eligible_pairs(table, pred, target, min_tiles=MIN_TILES_PER_PAIR):
    """§2.2.4: pairs with >= min_tiles TILE rows having finite target and finite prediction. For a block-only predictor
    (no finite tile value anywhere) the tile count uses the target alone and the pair must have >= 1 finite block value."""
    t = table[table.block_id < 0]; b = table[table.block_id >= 0]
    has_tile_pred = np.isfinite(t[pred].values).any() if pred in t.columns else False
    out = {}
    for pr, s in t.groupby("pair"):
        n = int((_ok(s[pred].values.astype(float), s[target].values.astype(float)) if has_tile_pred else np.isfinite(s[target].values.astype(float))).sum())
        okb = has_tile_pred or (pred in b.columns and np.isfinite(b[b.pair == pr][pred].values.astype(float)).any())
        out[pr] = dict(n_tiles_ok=n, eligible=bool(n >= min_tiles and okb))
    return out

def detection_pairs(table, target="E_1m", lo=DETECTION_BOUNDS[0], hi=DETECTION_BOUNDS[1]):
    """§2.2.5: pairs whose block-scale prevalence of E_1m > 0 lies within [lo, hi]."""
    b = table[table.block_id >= 0]; prev = b.groupby("pair")[target].apply(lambda s: float(np.mean(s.dropna() > 0)))
    return {pr: dict(prevalence=float(v), detection=bool(lo <= v <= hi)) for pr, v in prev.items()}

def score_one(table, pred, target, scale, units=None, elig_pairs=None, det_pairs=None):
    """Per-pair -> unit (median over eligible pairs) -> median over units; plus pooled (all rows).
    elig_pairs / det_pairs: the pair sets fixed from the TILE table (§2.2.4) and the block table (§2.2.5); when a table
    holds one scale only (as the bar's prediction tables do) they must be passed in; otherwise they are computed here."""
    elig = ({p: dict(eligible=(p in set(elig_pairs))) for p in table.pair.unique()} if elig_pairs is not None else eligible_pairs(table, pred, target))
    det = (({p: dict(detection=(p in set(det_pairs))) for p in table.pair.unique()} if det_pairs is not None else detection_pairs(table)) if target == "E_1m" else None)
    d = table[table.block_id >= 0] if scale == "block" else table[table.block_id < 0]
    units = units or sorted(d.unit.unique()); per_pair = {}; per_unit = {}
    for u in units:
        du = d[d.unit == u]; vals = {}
        for pr, s in du.groupby("pair"):
            if not elig.get(pr, {}).get("eligible", False): continue
            if det is not None and not det.get(pr, {}).get("detection", False): continue
            p = s[pred].values.astype(float); y = s[target].values.astype(float)
            r = dict(n=int(_ok(p, y).sum()), rho=rho(p, y), capture20=capture20(p, y))
            if target in LOGLOG_TARGETS: r["r2_loglog"] = loglog_r2(p, y)
            if target == "E_1m": r["auprc"] = auprc(p, y)
            vals[pr] = r
        per_pair[u] = vals
        keys = sorted({k for v in vals.values() for k in v if k != "n"})
        per_unit[u] = {k: float(np.nanmedian([v[k] for v in vals.values()])) if any(np.isfinite(v.get(k, np.nan)) for v in vals.values()) else np.nan for k in keys}
        per_unit[u]["n_pairs"] = len(vals); per_unit[u]["n"] = int(sum(v["n"] for v in vals.values()))
    keys = [k for k in ("rho", "r2_loglog", "capture20", "auprc") if any(k in per_unit[u] for u in units)]
    med = {k: float(np.nanmedian([per_unit[u].get(k, np.nan) for u in units])) if any(np.isfinite(per_unit[u].get(k, np.nan)) for u in units) else np.nan for k in keys}
    nunits = {k: int(np.sum([np.isfinite(per_unit[u].get(k, np.nan)) for u in units])) for k in keys}
    p = d[pred].values.astype(float); y = d[target].values.astype(float); g = d.unit.values
    pooled = dict(n=int(_ok(p, y).sum()), rho=rho(p, y), capture20=capture20(p, y))
    if target in LOGLOG_TARGETS: pooled["r2_loglog_within"] = within_r2_pooled(p, y, g)
    if target == "E_1m": pooled["auprc"] = auprc(p, y)
    return dict(pred=pred, target=target, scale=scale, per_pair=per_pair, per_unit=per_unit, median=med, n_units_scored=nunits, pooled=pooled,
                ineligible_pairs=sorted(pr for pr, v in elig.items() if not v["eligible"]), detection_excluded=(sorted(pr for pr, v in det.items() if not v["detection"]) if det else None),
                units_dropped=sorted(u for u in units if per_unit[u]["n_pairs"] == 0))

def compare(table, pred_a, pred_b, target, scale, units=None, metric="rho", n_boot=N_BOOT, seed=SEED, elig_pairs=None, det_pairs=None):
    Sa = score_one(table, pred_a, target, scale, units, elig_pairs, det_pairs); Sb = score_one(table, pred_b, target, scale, units, elig_pairs, det_pairs)
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

def median_ci(table, pred, target, scale, units=None, metric="rho", n_boot=N_BOOT, seed=SEED, elig_pairs=None, det_pairs=None):
    S = score_one(table, pred, target, scale, units, elig_pairs, det_pairs); units = units or sorted(S["per_unit"])
    v = np.array([S["per_unit"][u].get(metric, np.nan) for u in units]); v = v[np.isfinite(v)]; rng = np.random.default_rng(seed); n = len(v)
    bs = np.array([np.median(v[rng.integers(0, n, n)]) for _ in range(n_boot)]) if n else np.array([np.nan])
    return dict(pred=pred, target=target, scale=scale, metric=metric, value=float(np.median(v)) if n else np.nan, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_units=int(n))

def evaluate(table, preds, allow_ew0207=False, units=None, elig_pairs=None, det_pairs=None):
    """elig_pairs: {target: [pairs]} or None (computed from the table's tile rows); det_pairs: [pairs] or None."""
    units_all = assert_units(table, allow_ew0207); units = units or units_all
    out = {"units": units, "detection_pairs": det_pairs if det_pairs is not None else detection_pairs(table), "scores": {}}
    for tgt, scales in TARGETS.items():
        for sc in scales: out["scores"][f"{tgt}/{sc}"] = {p: score_one(table, p, tgt, sc, units, (elig_pairs or {}).get(tgt), det_pairs) for p in preds if p in table.columns}
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

# ---------------- self-test (§2.2.7) ----------------
def selftest(verbose=True):
    """Synthetic table with a multi-pair unit whose pairs differ in level; a predictor carrying only the pair level must
    score near zero under v1.1 and positive under v1 (unit-pooled ranking)."""
    import eval_v1 as E1
    rng = np.random.default_rng(1); rows = []; pairs_of = {u: ps for u, ps in C.UNITS.items()}
    for u, ps in pairs_of.items():
        for j, pr in enumerate(ps[:3]):
            level = 10 ** (0.6 * j)                      # pairs of a unit differ in level
            for i in range(40):
                z = rng.normal(); y = level * 10 ** (0.3 * z + rng.normal(0, 0.15))
                rows.append(dict(unit=u, pair=pr, tile_id=f"{pr}_{i}", block_id=-1, A_fine=y, T_fine=y, E_1m=0.3 if z > 0 else 0.0, level_only=level * 10 ** rng.normal(0, 0.05), informative=y * 10 ** rng.normal(0, 0.2), noise=10 ** rng.normal(0, 0.3)))
                for bi in range(16): rows.append({**rows[-1], "block_id": bi, "tile_id": f"{pr}_{i}", "A_fine": y * 10 ** rng.normal(0, 0.1), "E_1m": max(0.0, 0.3 * z + rng.normal(0, 0.1))})
    t = pd.DataFrame(rows); multi = [u for u, ps in pairs_of.items() if len(ps) > 1]
    v11 = {p: score_one(t, p, "A_fine", "tile", list(C.UNITS)) for p in ("level_only", "informative", "noise")}
    v1 = {p: E1.score_one(t, p, "A_fine", "tile", list(C.UNITS)) for p in ("level_only", "informative", "noise")}
    res = {p: dict(v1_1_median_rho=v11[p]["median"]["rho"], v1_median_rho=v1[p]["median"]["rho"], v1_1_multi_units={u: v11[p]["per_unit"][u]["rho"] for u in multi}, v1_multi_units={u: v1[p]["per_unit"][u]["rho"] for u in multi}) for p in v11}
    assert abs(res["level_only"]["v1_1_median_rho"]) < 0.15, res["level_only"]
    assert all(res["level_only"]["v1_multi_units"][u] > 0.5 for u in multi), res["level_only"]
    assert res["informative"]["v1_1_median_rho"] > 0.6 and abs(res["noise"]["v1_1_median_rho"]) < 0.15
    try: assert_units(t.assign(unit="zz")); raise AssertionError("unit assertion did not fire")
    except AssertionError as e:
        if "did not fire" in str(e): raise
    try: assert_units(t.assign(pair=C.SEALED[0])); raise AssertionError("sealed refusal did not fire")
    except PermissionError: pass
    c = compare(t, "informative", "noise", "A_fine", "tile"); assert c["units_a_better"] == 16 and c["ci95"][0] > 0
    el = eligible_pairs(t.assign(informative=np.where(t.pair == pairs_of["lu05"][0], np.nan, t.informative)), "informative", "A_fine"); assert not el[pairs_of["lu05"][0]]["eligible"]
    if verbose:
        print("eval_v1_1 self-test PASS")
        for p, r in res.items(): print(f"  {p:12s} v1.1 median rho {r['v1_1_median_rho']:+.3f} | v1 median rho {r['v1_median_rho']:+.3f} | multi-pair units v1.1 " + ", ".join(f"{u} {v:+.2f}" for u, v in r["v1_1_multi_units"].items()) + " | v1 " + ", ".join(f"{u} {v:+.2f}" for u, v in r["v1_multi_units"].items()))
    return res

if __name__ == "__main__":
    import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common
    r = selftest(); os.makedirs(f"{r05_common.ROOT}/eval", exist_ok=True); json.dump(r, open(f"{r05_common.ROOT}/eval/eval_v1_1_selftest.json", "w"), indent=1)
