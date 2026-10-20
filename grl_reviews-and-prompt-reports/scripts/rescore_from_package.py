#!/usr/bin/env python3
"""Rescore every pair of the development set and both blind tests from the packaged tile tables (data record
10.5281/zenodo.23515276) with the frozen scoring rule, and check each rho against the frozen value.
Usage: python rescore_from_package.py <folder with tiles.parquet and pairs.parquet>
Rule (phase4_score.py score_pair): drop excluded_r17 tiles and tiles without a finite A_fine; interior view
(Spearman vs A_fine on interior tiles) if >= MIN_TILES tiles in >= bmin 4x4-tile blocks, else fallback view
(vs A_fine_int on tiles with finite A_fine_int) under the same counts, else not scorable (rho on the larger subset).
Bootstrap intervals and pooled statistics are the frozen scorers' job; this script reproduces the point values only."""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

MIN_TILES = 10
folder = sys.argv[1]
tiles = pd.read_parquet(f"{folder}/tiles.parquet")
pairs = pd.read_parquet(f"{folder}/pairs.parquet")
pairs = pairs[pairs.in_corpus & pairs.rho_s_lr.notna()]


def rho(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 3 or np.std(p[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(p[ok], y[ok])[0])


def n_blocks(d):
    return len(set(zip(d.r0.values // 128 // 4, d.c0.values // 128 // 4)))


rows, bad = [], 0
for r in pairs.itertuples():
    t = tiles[(tiles.pair == r.pair_id) & (~tiles.excluded_r17.astype(bool))]
    t = t[np.isfinite(t.A_fine.values)]
    bmin = int(r.bmin) if pd.notna(r.bmin) else 8
    vi, vf = t[t.interior.astype(bool)], t[np.isfinite(t.A_fine_int.values)]
    if len(vi) >= MIN_TILES and n_blocks(vi) >= bmin:
        view, tab, tgt = "interior", vi, "A_fine"
    elif len(vf) >= MIN_TILES and n_blocks(vf) >= bmin:
        view, tab, tgt = "fallback", vf, "A_fine_int"
    else:
        view = "not_scorable"
        tab = vi if len(vi) >= MIN_TILES else (vf if len(vf) >= MIN_TILES else None)
        tgt = "A_fine" if len(vi) >= MIN_TILES else "A_fine_int"
    out = {"pair": r.pair_id, "group": r.group, "view": view, "frozen_view": r.view, "n_scored": 0 if tab is None else len(tab)}
    for cand, frozen in (("s_lr", r.rho_s_lr), ("GBT_ship", r.rho_GBT_ship), ("P-V1", r.rho_P_V1)):
        val = rho(tab[cand].values.astype(float), tab[tgt].values.astype(float)) if tab is not None else float("nan")
        match = (np.isfinite(val) and pd.notna(frozen) and abs(val - frozen) < 5e-4) or (not np.isfinite(val) and pd.isna(frozen))
        out[f"rho_{cand}"], out[f"frozen_{cand}"], out[f"match_{cand}"] = val, frozen, bool(match)
        bad += int(not match)
    rows.append(out)
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n{len(res)} scored pairs; mismatches at 3 decimals: {bad}; views agree with frozen: {int((res.view == res.frozen_view).sum())}/{len(res)}")
sys.exit(1 if bad else 0)
