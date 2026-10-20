"""Phase 4 scoring (pre-registration v2, B4 and B5), run ONCE in phase4 mode. Inputs: a tile table (tile_id, pair, unit, role,
r0, c0, hr_res, lr_native, A_fine, A_fine_int, interior, s_lr, GBT_ship, P-V1), the joined blind-calls table (pair, relief,
criteria, dominant, ab_whole_survey), B_min, and (optional) the build's pair statistics (amplitude assert; no-k status).
Per pair (its own unit):
  view  = interior tiles vs A_fine if >= 10 interior tiles in >= B_min non-empty 4 x 4-tile blocks; else the fallback (tiles with
          a defined A_fine_int, vs A_fine_int) under the same counts; else not scorable (point rho reported without CI where a view has >= 10 tiles);
  rho   = eval_v1_1.rho (within-pair Spearman) per candidate on the view;
  CI    = bootstrap over the non-empty blocks of the view (same number of blocks, with replacement), 2,000 resamples, a fresh
          default_rng(20261009) per pair, 95 % percentile; resamples with undefined rho counted and excluded; > 5 % -> CI not estimable, outside C1;
  amplitude = median of the scored target over the scored tiles of the view;
  cells_across = 256 x hr_res / lr_native, flagged thinner_than_development below the development minimum (3.88; no rule attached);
  secondary = all-tile rho vs A_fine; A_fine_int rho where the primary view is in use (each with >= 10 tiles);
  status = not scorable (no k) [B4 rule 0]; artifact-dominated (dominant = artifact); low-amplitude artifact-affected (ab_whole_survey = yes
           and amplitude < 0.04 m); else in C1 (flagged if any criterion).
Pooled row (§A1, secondary summary of C1): mean over units of s_lr's rho with a unit-stratified block bootstrap (2,000 resamples, one
default_rng(20261013) per statement, units in ascending pair-id order; a resample with any undefined unit rho is dropped and counted,
> 5 % -> not estimable), on the C1 set, on the C1 set plus the low-amplitude units, and on the scorable lockbox pairs.
C1 per role: confirmed / partially confirmed / not confirmed / not testable, stated with and without the low-amplitude pairs;
C2 differences per pair with the sign-test caveat; C3 = the lockbox role. Writes <out>/results_<tag>.json and per_pair_<tag>.csv.
k provenance (R11 §A3): every pair row carries k and k_sweep_status from its manifest / supplement row and, from the acquisition
step's run record (phase4 mode: the record accepted by the §A2.2 lock; rehearsal: --acq_record), the per-k verdicts, the source of
k (sweep | June record) and whether the record's k agrees. --rows (rehearsal only) replaces the listed pairs' manifest rows by the
rows of a harness supplement (R11 §A2.3); it is refused in phase4 mode."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase4_common as P4
import eval_v1_1 as E

def blocks_of(t): return [(int(r0) // 128 // 4, int(c0) // 128 // 4) for r0, c0 in zip(t.r0.values, t.c0.values)]

def block_index(blk):
    keys = sorted(set(blk)); arr = np.array([keys.index(b) for b in blk]); return keys, [np.where(arr == i)[0] for i in range(len(keys))]

def resample_rhos(pred, target, blk, rng, n_boot=P4.N_BOOT):
    """n_boot block-bootstrap rho values of one unit (same number of blocks, with replacement) drawn from rng; NaN where undefined."""
    keys, idx = block_index(blk); nb = len(keys); vals = np.empty(n_boot)
    for r in range(n_boot):
        pick = rng.integers(0, nb, nb); ii = np.concatenate([idx[j] for j in pick]); vals[r] = E.rho(pred[ii], target[ii])
    return vals

def boot_ci(pred, target, blk, n_boot=P4.N_BOOT, seed=P4.BOOT_SEED):
    vals = resample_rhos(pred, target, blk, np.random.default_rng(seed), n_boot); n_undef = int(np.isnan(vals).sum()); frac = n_undef / n_boot; ok = vals[np.isfinite(vals)]; nb = len(set(blk))
    if frac > P4.UNDEF_MAX_FRAC or len(ok) == 0: return dict(ci95=[np.nan, np.nan], estimable=False, n_undefined=n_undef, frac_undefined=frac, n_blocks=nb)
    return dict(ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))], estimable=True, n_undefined=n_undef, frac_undefined=frac, n_blocks=nb)

def pooled_row(units, seed=P4.POOL_SEED, n_boot=P4.N_BOOT, resamples=None):
    """§A1. units: list of dict(pair, pred, target, blk) on each unit's view in use. One default_rng(seed) per statement; units in
    ascending pair-id order; resample r of the pooled statistic = mean over units of their resample-r rho. Returns the row and the
    per-unit resample arrays (so that a caller may reuse them, as R10b §B5 does); `resamples` may supply them (keyed by pair)."""
    units = sorted(units, key=lambda u: u["pair"])
    if not units: return None, {}
    rng = np.random.default_rng(seed); R = {}
    for u in units: R[u["pair"]] = resamples[u["pair"]] if (resamples is not None and u["pair"] in resamples) else resample_rhos(u["pred"], u["target"], u["blk"], rng, n_boot)
    M = np.stack([R[u["pair"]] for u in units], 0); point = {u["pair"]: E.rho(u["pred"], u["target"]) for u in units}
    pooled = M.mean(0); anynan = np.isnan(M).any(0); n_drop = int(anynan.sum()); ok = pooled[~anynan]
    row = dict(n_units=len(units), pairs=[u["pair"] for u in units], mean_rho=float(np.mean(list(point.values()))), per_unit_rho=point, n_dropped=n_drop, frac_dropped=n_drop / n_boot, seed=seed, n_boot=n_boot)
    if n_drop / n_boot > P4.UNDEF_MAX_FRAC or len(ok) == 0: row.update(ci95=[np.nan, np.nan], estimable=False, lower_gt0=False)
    else: lo, hi = float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)); row.update(ci95=[lo, hi], estimable=True, lower_gt0=bool(lo > 0))
    return row, R

def score_pair(t, call, bmin, cands, stats=None, dev_min=P4.DEV_MIN_CELLS):
    r = dict(pair=t.pair.iloc[0], unit=t.unit.iloc[0], role=t.role.iloc[0], bmin=bmin)
    if stats is not None and stats.get("status") == P4.NO_K:
        r.update(view="not_scorable", status=P4.NO_K, note="no recoverable k: no fine band, no figure page, outside C1 (B4 rule 0)", n_tiles_all=0, n_scored=0); return r
    t = t[np.isfinite(t.A_fine.values)]; r["n_tiles_all"] = int(len(t))
    r["hr_res"] = float(t.hr_res.iloc[0]); r["lr_native"] = float(t.lr_native.iloc[0]); r["cells_across"] = 256 * r["hr_res"] / r["lr_native"]; r["thinner_than_development"] = bool(r["cells_across"] < dev_min)
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
        r["c1_condition"] = bool(b["estimable"] and b["ci95"][0] > 0); r["_view"] = dict(pair=r["pair"], pred=tab["s_lr"].values.astype(float), target=tab[tgt].values.astype(float), blk=blocks_of(tab))
    else: r.update(ci_lo_s_lr=np.nan, ci_hi_s_lr=np.nan, ci_estimable=False, boot_undefined=np.nan, boot_frac_undefined=np.nan, c1_condition=False, note="too few blocks, not in C1" if tab is not None else "not scorable")
    r["d_GBT_minus_s_lr"] = r["rho_GBT_ship"] - r["rho_s_lr"]; r["d_PV1_minus_s_lr"] = r["rho_P-V1"] - r["rho_s_lr"]; r["d_PV1_minus_GBT"] = r["rho_P-V1"] - r["rho_GBT_ship"]
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

def pooled_statements(rows):
    """The pooled row on the C1 set, on the C1 set + low-amplitude units, and (lockbox role) on the scorable pairs; each with a fresh generator."""
    inset = [r["_view"] for r in rows if r["status"].startswith("in C1")]; low = [r["_view"] for r in rows if r["status"] == "low-amplitude artifact-affected"]
    out = {"C1_set": pooled_row(inset)[0], "C1_set_plus_low_amplitude": pooled_row(inset + low)[0]}
    if rows and rows[0]["role"] == "lockbox": out["scorable_lockbox"] = pooled_row([r["_view"] for r in rows if "_view" in r])[0]
    return out

if __name__ == "__main__":
    ap = P4.argparser("Phase 4 scoring"); ap.add_argument("--inputs", required=True); ap.add_argument("--calls", required=True, help="joined calls table (pair, relief, criteria, dominant, ab_whole_survey)")
    ap.add_argument("--stats", default=None, help="JSON {pair: pair_stats} from the build (amplitude assert; no-k status)"); ap.add_argument("--bmin", type=int, default=P4.B_MIN_DEFAULT); ap.add_argument("--tag", default="D1"); ap.add_argument("--out", default=None)
    ap.add_argument("--acq_record", default=None, help="rehearsal only: an acquisition step1 run record (per_pair: per-k verdicts, k_source) for the k provenance columns"); ap.add_argument("--rows", default=None, help="rehearsal only: parquet whose rows replace the listed pairs' manifest rows (harness supplement)"); a = ap.parse_args()
    pairs = P4.split_pairs(a.pairs); out = a.out or f"{P4.ROOT[a.mode]}/scores"; results = f"{out}/results_{a.tag}.json"
    m, root, rec = P4.start("phase4_score.py", a.mode, pairs, calls_path=a.calls, results_path=results if a.mode == "phase4" else None); t0 = time.time(); os.makedirs(out, exist_ok=True); dev_min = P4.dev_min_cells()
    if a.mode == "phase4" and (a.acq_record or a.rows): raise P4.Refusal("--acq_record and --rows are rehearsal-only (phase4 mode takes the acquisition run record from the §A2.2 lock and the rows from the supplement)")
    acq = rec.get("acq_run_record") if a.mode == "phase4" else (dict(path=a.acq_record, sha256=P4.sha256_file(a.acq_record), per_pair=json.load(open(a.acq_record)).get("per_pair") or {}) if a.acq_record else None)
    if a.rows: m, _ = P4.harness_rows(m, a.rows, pairs)
    KP = {pid: P4.k_provenance(pid, P4.C4.pair_row(m, pid), acq) for pid in pairs}
    T = pd.read_parquet(a.inputs); T = T[T.pair.isin(pairs)]; calls = pd.read_csv(a.calls).fillna(""); cmap = {r["pair"]: r for r in calls.to_dict("records")}
    stats = json.load(open(a.stats)) if a.stats else {}; cands = [c for c in P4.CANDIDATES if c in T.columns]; assert "s_lr_W64" not in T.columns, "s_lr_W64 is not scored at Phase 4 (ruling H1, R10b)"
    rows = []
    for pid in pairs:
        st = stats.get(pid); tp = T[T.pair == pid]
        if (st and st.get("status") == P4.NO_K) or len(tp) == 0: rows.append(dict(pair=pid, unit=P4.unit_of(pid, m), role=(tp.role.iloc[0] if len(tp) else st.get("role", "?")), bmin=a.bmin, view="not_scorable", status=P4.NO_K, note="no recoverable k (B4 rule 0)", n_tiles_all=0, n_scored=0)); continue
        rows.append(score_pair(tp, cmap.get(pid, {}), a.bmin, cands, st, dev_min))
    for r in rows: k = KP[r["pair"]]; r.update(k=k["k"], k_sweep_status=k["k_sweep_status"], k_source=k["k_source_class"], k_source_text=k["k_source"], per_k_verdicts=k["per_k_text"], k_agrees_with_acq_record=k["k_agrees_with_acq_record"])
    PR = pd.DataFrame([{k: v for k, v in r.items() if k != "_view"} for r in rows]); PR.to_csv(f"{out}/per_pair_{a.tag}.csv", index=False)
    roles = sorted(set(PR.role)); C1 = {role: c1_statement([r for r in rows if r["role"] == role]) for role in roles}; POOL = {role: pooled_statements([r for r in rows if r["role"] == role]) for role in roles}
    C2 = {role: {k: dict(values={r["pair"]: r[k] for r in rows if r["role"] == role and r["status"].startswith("in C1")}, caveat="no superiority claim: with at most 3 units, even 3 of 3 gives a one-sided sign test of p = 0.125") for k in ("d_GBT_minus_s_lr", "d_PV1_minus_s_lr", "d_PV1_minus_GBT")} for role in roles}
    res = dict(mode=a.mode, tag=a.tag, bmin=a.bmin, candidates=cands, dev_min_cells=dev_min, inputs=a.inputs, k_provenance=KP, acq_run_record=({k: v for k, v in acq.items() if k != "per_pair"} if acq else None), harness_rows=(dict(path=a.rows, sha256=P4.sha256_file(a.rows)) if a.rows else None), inputs_sha256=P4.sha256_file(a.inputs), calls=a.calls, calls_sha256=P4.sha256_file(a.calls), rules=dict(min_tiles=P4.MIN_TILES, n_boot=P4.N_BOOT, seed=P4.BOOT_SEED, pool_seed=P4.POOL_SEED, undefined_max_frac=P4.UNDEF_MAX_FRAC, lowamp_m=P4.LOWAMP_M),
               per_pair=[{k: v for k, v in r.items() if k != "_view"} for r in rows], C1=C1, C1_pooled=POOL, C2=C2, C3="lockbox role: C1, the C1 summary and C2 reported separately with the history label" if "lockbox" in roles else None, seconds=time.time() - t0)
    json.dump(res, open(results, "w"), indent=1, default=float); print(f"scored {len(rows)} pairs -> {results}; C1 per role: " + "; ".join(f"{k}: {v['reading']} (set {v['c1_set_size']})" for k, v in C1.items()) + "; pooled: " + "; ".join(f"{k} C1 set mean {v['C1_set']['mean_rho']:+.3f} [{v['C1_set']['ci95'][0]:+.3f}, {v['C1_set']['ci95'][1]:+.3f}]" for k, v in POOL.items() if v["C1_set"]), flush=True)
