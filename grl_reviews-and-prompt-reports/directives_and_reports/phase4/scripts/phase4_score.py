"""Phase 4 scoring (pre-registration v2, B4 and B5), run ONCE in phase4 mode. Inputs: a tile table (tile_id, pair, unit, role,
r0, c0, A_fine, A_fine_int, interior, s_lr, GBT_ship, P-V1 [, s_lr_W64]), the joined blind-calls table (pair, relief,
criteria, dominant, ab_whole_survey), B_min, and (optional) the build's amplitude statistics to assert against.
Per pair (its own unit):
  view  = interior tiles vs A_fine if >= 10 interior tiles in >= B_min non-empty 4 x 4-tile blocks; else the fallback (tiles with
          a defined A_fine_int, vs A_fine_int) under the same counts; else not scorable (point rho reported without CI where a view has >= 10 tiles);
  rho   = eval_v1_1.rho (within-pair Spearman) per candidate on the view;
  CI    = bootstrap over the non-empty blocks of the view (same number of blocks, with replacement), 2,000 resamples, a fresh
          default_rng(20261009) per pair, 95 % percentile; resamples with undefined rho counted and excluded; > 5 % -> CI not estimable, outside C1;
  amplitude = median of the scored target over the scored tiles of the view;
  secondary = all-tile rho vs A_fine; A_fine_int rho where the primary view is in use (each with >= 10 tiles);
  status = artifact-dominated (dominant = artifact) -> outside C1; low-amplitude artifact-affected (ab_whole_survey = yes and amplitude
           < 0.04 m) -> outside C1, in the second C1 statement; else in C1 (flagged if any criterion).
C1 per role: confirmed / partially confirmed / not confirmed / not testable, stated with and without the low-amplitude pairs;
C2 differences per pair with the sign-test caveat; C3 = the lockbox role. Writes <out>/results_<tag>.json and per_pair_<tag>.csv."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase4_common as P4
import eval_v1_1 as E

def blocks_of(t): return [(int(r0) // 128 // 4, int(c0) // 128 // 4) for r0, c0 in zip(t.r0.values, t.c0.values)]

def boot_ci(pred, target, blk, n_boot=P4.N_BOOT, seed=P4.BOOT_SEED):
    keys = sorted(set(blk)); idx = {k: np.where(np.array([b == k for b in blk]))[0] for k in keys}; nb = len(keys); rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, nb, nb); ii = np.concatenate([idx[keys[j]] for j in pick]); vals.append(E.rho(pred[ii], target[ii]))
    vals = np.array(vals, float); n_undef = int(np.isnan(vals).sum()); frac = n_undef / n_boot; ok = vals[np.isfinite(vals)]
    if frac > P4.UNDEF_MAX_FRAC or len(ok) == 0: return dict(ci95=[np.nan, np.nan], estimable=False, n_undefined=n_undef, frac_undefined=frac, n_blocks=nb)
    return dict(ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))], estimable=True, n_undefined=n_undef, frac_undefined=frac, n_blocks=nb)

def score_pair(t, call, bmin, cands, stats=None):
    t = t[np.isfinite(t.A_fine.values)]; r = dict(pair=t.pair.iloc[0], unit=t.unit.iloc[0], role=t.role.iloc[0], n_tiles_all=int(len(t)), bmin=bmin)
    vi = t[t.interior.values.astype(bool)]; vf = t[np.isfinite(t.A_fine_int.values)]
    r["n_interior"], r["blocks_interior"] = len(vi), len(set(blocks_of(vi))); r["n_fallback"], r["blocks_fallback"] = len(vf), len(set(blocks_of(vf)))
    if r["n_interior"] >= P4.MIN_TILES and r["blocks_interior"] >= bmin: view, tab, tgt = "interior", vi, "A_fine"
    elif r["n_fallback"] >= P4.MIN_TILES and r["blocks_fallback"] >= bmin: view, tab, tgt = "fallback", vf, "A_fine_int"
    else: view, tab, tgt = "not_scorable", (vi if r["n_interior"] >= P4.MIN_TILES else (vf if r["n_fallback"] >= P4.MIN_TILES else None)), ("A_fine" if r["n_interior"] >= P4.MIN_TILES else "A_fine_int")
    r["view"] = view; r["target"] = tgt if tab is not None else None; r["n_scored"] = int(len(tab)) if tab is not None else 0
    r["amplitude_m"] = float(np.nanmedian(tab[tgt].values)) if tab is not None and len(tab) else np.nan
    if stats is not None and view in ("interior", "fallback"):
        ref = stats["views"][view]["amplitude_m"]; assert abs(ref - r["amplitude_m"]) <= 1e-9, (r["pair"], view, ref, r["amplitude_m"]); r["amplitude_from_build"] = ref
    for c in cands:
        r[f"rho_{c}"] = E.rho(tab[c].values.astype(float), tab[tgt].values.astype(float)) if tab is not None else np.nan
        r[f"rho_all_{c}"] = E.rho(t[c].values.astype(float), t.A_fine.values.astype(float)) if len(t) >= P4.MIN_TILES else np.nan
        r[f"rho_int_{c}"] = E.rho(vf[c].values.astype(float), vf.A_fine_int.values.astype(float)) if (view == "interior" and len(vf) >= P4.MIN_TILES) else np.nan
    if view in ("interior", "fallback"):
        b = boot_ci(tab["s_lr"].values.astype(float), tab[tgt].values.astype(float), blocks_of(tab)); r.update(ci_lo_s_lr=b["ci95"][0], ci_hi_s_lr=b["ci95"][1], ci_estimable=b["estimable"], boot_undefined=b["n_undefined"], boot_frac_undefined=b["frac_undefined"])
        r["c1_condition"] = bool(b["estimable"] and b["ci95"][0] > 0)
    else: r.update(ci_lo_s_lr=np.nan, ci_hi_s_lr=np.nan, ci_estimable=False, boot_undefined=np.nan, boot_frac_undefined=np.nan, c1_condition=False, note="too few blocks, not in C1" if tab is not None else "not scorable")
    r["d_GBT_minus_s_lr"] = r["rho_GBT_ship"] - r["rho_s_lr"]; r["d_PV1_minus_s_lr"] = r["rho_P-V1"] - r["rho_s_lr"]; r["d_PV1_minus_GBT"] = r["rho_P-V1"] - r["rho_GBT_ship"]
    if "s_lr_W64" in cands: r["d_W64_minus_s_lr"] = r["rho_s_lr_W64"] - r["rho_s_lr"]
    crit = str(call.get("criteria", "") or ""); dom = str(call.get("dominant", "")); ab = str(call.get("ab_whole_survey", "")).lower()
    r.update(call_relief=call.get("relief"), call_criteria=crit, call_dominant=dom, call_ab_whole_survey=ab)
    if view == "not_scorable": r["status"] = "not scorable"
    elif dom == "artifact": r["status"] = "artifact-dominated"
    elif ab == "yes" and np.isfinite(r["amplitude_m"]) and r["amplitude_m"] < P4.LOWAMP_M: r["status"] = "low-amplitude artifact-affected"
    elif not r["ci_estimable"]: r["status"] = "CI not estimable"
    else: r["status"] = "in C1" + (" (flagged)" if crit not in ("", "none") else "")
    return r

def c1_statement(rows):
    inset = [r for r in rows if r["status"].startswith("in C1")]; low = [r for r in rows if r["status"] == "low-amplitude artifact-affected"]
    def read(rs):
        if not rs: return "not testable (C1 set empty)"
        n = sum(r["c1_condition"] for r in rs); return "confirmed" if n == len(rs) else ("not confirmed" if n == 0 else "partially confirmed") + f" ({n} of {len(rs)})"
    return dict(c1_set=[r["pair"] for r in inset], c1_set_size=len(inset), reading=read(inset), with_low_amplitude=dict(set=[r["pair"] for r in inset + low], reading=read(inset + low)))

if __name__ == "__main__":
    ap = P4.argparser("Phase 4 scoring"); ap.add_argument("--inputs", required=True); ap.add_argument("--calls", required=True, help="joined calls table (pair, relief, criteria, dominant, ab_whole_survey)")
    ap.add_argument("--stats", default=None, help="JSON {pair: pair_stats} from the build (amplitude assert)"); ap.add_argument("--bmin", type=int, default=P4.B_MIN_DEFAULT); ap.add_argument("--tag", default="D1"); ap.add_argument("--out", default=None); a = ap.parse_args()
    pairs = P4.split_pairs(a.pairs); out = a.out or f"{P4.ROOT[a.mode]}/scores"; results = f"{out}/results_{a.tag}.json"
    m, root, rec = P4.start("phase4_score.py", a.mode, pairs, calls_path=a.calls, results_path=results if a.mode == "phase4" else None); t0 = time.time(); os.makedirs(out, exist_ok=True)
    T = pd.read_parquet(a.inputs); T = T[T.pair.isin(pairs)]; calls = pd.read_csv(a.calls).fillna(""); cmap = {r["pair"]: r for r in calls.to_dict("records")}
    stats = json.load(open(a.stats)) if a.stats else {}; cands = [c for c in ("s_lr", "GBT_ship", "P-V1", "s_lr_W64") if c in T.columns]
    rows = [score_pair(T[T.pair == pid], cmap.get(pid, {}), a.bmin, cands, stats.get(pid)) for pid in pairs]
    PR = pd.DataFrame(rows); PR.to_csv(f"{out}/per_pair_{a.tag}.csv", index=False)
    roles = sorted(set(PR.role)); C1 = {role: c1_statement([r for r in rows if r["role"] == role]) for role in roles}
    C2 = {role: {k: dict(values={r["pair"]: r[k] for r in rows if r["role"] == role and r["status"].startswith("in C1")}, caveat="no superiority claim: with at most 3 units, even 3 of 3 gives a one-sided sign test of p = 0.125") for k in ("d_GBT_minus_s_lr", "d_PV1_minus_s_lr", "d_PV1_minus_GBT") + (("d_W64_minus_s_lr",) if "s_lr_W64" in cands else ())} for role in roles}
    res = dict(mode=a.mode, tag=a.tag, bmin=a.bmin, candidates=cands, inputs=a.inputs, inputs_sha256=P4.sha256_file(a.inputs), calls=a.calls, calls_sha256=P4.sha256_file(a.calls), rules=dict(min_tiles=P4.MIN_TILES, n_boot=P4.N_BOOT, seed=P4.BOOT_SEED, undefined_max_frac=P4.UNDEF_MAX_FRAC, lowamp_m=P4.LOWAMP_M),
               per_pair=rows, C1=C1, C2=C2, C3="lockbox role, reported separately with its history label" if "lockbox" in roles else None, seconds=time.time() - t0)
    json.dump(res, open(results, "w"), indent=1, default=float); print(f"scored {len(rows)} pairs -> {results}; C1 per role: " + "; ".join(f"{k}: {v['reading']} (set {v['c1_set_size']})" for k, v in C1.items()), flush=True)
