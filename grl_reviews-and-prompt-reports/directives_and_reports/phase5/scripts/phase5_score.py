"""Phase 5 scoring (pre-registration v3, step 5), run ONCE in phase5 mode: a parameterised copy of phase4_score.py's driver.
IMPORTED unchanged from the pinned phase4_score.py (R16 §B8): score_pair (views, B_min, the within-pair rho, the s_lr block
bootstrap CI with seed 20261009, the amplitude, the artifact rule and statuses), boot_ci / resample_rhos / block_index,
pooled_row (unit-stratified, seed 20261013) and c1_statement (the v2.1 per-site label). Changes from Phase 4:
  §B5  --calls is Steve's committed file keyed by Q-code (columns code, relief, criteria, dominant, ab_whole_survey, note; its
       sha256 in the ledger: the calls lock). It is joined to the pairs here, through <root>/figures/key/figset_A_key.json,
       with every value asserted unchanged; no joined file is written (the code -> pair map goes into the results file).
  §B1  a not-testable site (eligible = False) is reported "not testable (<eligibility_reason>)" from its supplement row and
       is outside C1; nothing of it is read.
  §B6  C1 primary (spec §3.4) = the pooled row over the C1-set sites: "confirmed" if its 95 % CI lower bound is above zero,
       "not confirmed" otherwise, "not testable" if fewer than 3 sites are in the C1 set. The v2.1 per-site reading
       (c1_statement: label and count with lower bound above zero) is written beside it as the secondary reading.
  §B7  C2 (spec §3.5, descriptive): per site with a scorable view and pooled over the C1 set, rho of GBT_ship and P-V1 and
       their differences from s_lr, each with a 95 % percentile CI from the SAME bootstrap resamples as s_lr (per site: the
       block picks of default_rng(20261009); pooled: the picks of default_rng(20261013), units in ascending pair-id order;
       resamples with an undefined value excluded and counted, > 5 % -> not estimable). The s_lr resamples are asserted
       identical to the imported resample_rhos / pooled_row ones. No superiority claim.
Writes <out>/results_<tag>.json and per_pair_<tag>.csv; --rows / --acq_record are rehearsal-only, as in Phase 4."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase5_common as P5
import phase4_score as PS4                                       # the pinned Phase 4 scoring functions
import eval_v1_1 as E
P4 = P5.P4
CALL_COLS = ["code", "relief", "criteria", "dominant", "ab_whole_survey", "note"]
C2_QUANT = ("s_lr", "GBT_ship", "P-V1", "d_GBT_minus_s_lr", "d_PV1_minus_s_lr")
MIN_C1_SITES = 3

def join_calls(calls_path, key_path):
    """§B5: Steve's code-keyed file -> {pair: call record}, parsed exactly as phase4_score (read_csv(...).fillna("")), joined through
    the figures key; asserts the codes match the key one to one and every value of Steve's file is carried unchanged."""
    key = json.load(open(key_path)); code2pair = {o["code"]: o["pair"] for o in key["order"]}
    raw = pd.read_csv(calls_path).fillna(""); txt = pd.read_csv(calls_path, dtype=str, keep_default_na=False)
    assert list(txt.columns) == CALL_COLS, f"calls file columns {list(txt.columns)} != {CALL_COLS}"
    assert txt.code.is_unique and set(txt.code) == set(code2pair), (sorted(txt.code), sorted(code2pair))
    recs = raw.to_dict("records"); cmap = {}
    for i, r in enumerate(recs):
        for c in CALL_COLS:
            v, s = r[c], txt.loc[i, c]; assert (v == "" and s == "") or str(v) == s or _num_equal(v, s), ("calls value changed by parsing", r["code"], c, v, s)
        pair = code2pair[r["code"]]; assert pair not in cmap, pair; cmap[pair] = dict(r)
    assert all(cmap[code2pair[r["code"]]] == r for r in recs), "joined calls differ from the file"
    return cmap, dict(code_to_pair=code2pair, key_path=key_path, key_sha256=P5.sha256_file(key_path), n_calls=len(recs), values_unchanged=True)

def _num_equal(v, s):
    try: return float(v) == float(s)
    except Exception: return False

def view_table(t, r):
    """The rows score_pair scored for this pair (its view), rebuilt from the same rule and asserted equal to score_pair's s_lr / target arrays."""
    t = t[np.isfinite(t.A_fine.values)]
    if r["view"] == "interior": tab, tgt = t[t.interior.values.astype(bool)], "A_fine"
    elif r["view"] == "fallback": tab, tgt = t[np.isfinite(t.A_fine_int.values)], "A_fine_int"
    else: return None, None
    v = r["_view"]; assert len(tab) == r["n_scored"] and np.array_equal(tab["s_lr"].values.astype(float), v["pred"]) and np.array_equal(tab[tgt].values.astype(float), v["target"], equal_nan=True), r["pair"]
    return tab, tgt

def picks(blk, rng, n_boot=P4.N_BOOT):
    """The block-bootstrap row indices of resample_rhos (same rng consumption: rng.integers(0, nb, nb) per resample)."""
    keys, idx = PS4.block_index(blk); nb = len(keys)
    return [np.concatenate([idx[j] for j in rng.integers(0, nb, nb)]) for _ in range(n_boot)]

def quant_resamples(tab, tgt, pk):
    y = tab[tgt].values.astype(float); X = {c: tab[c].values.astype(float) for c in ("s_lr", "GBT_ship", "P-V1")}
    R = {c: np.array([E.rho(X[c][ii], y[ii]) for ii in pk]) for c in X}
    R["d_GBT_minus_s_lr"] = R["GBT_ship"] - R["s_lr"]; R["d_PV1_minus_s_lr"] = R["P-V1"] - R["s_lr"]; return R

def ci_of(vals, n_boot=P4.N_BOOT):
    n_undef = int(np.isnan(vals).sum()); frac = n_undef / n_boot; ok = vals[np.isfinite(vals)]
    if frac > P4.UNDEF_MAX_FRAC or len(ok) == 0: return dict(ci95=[np.nan, np.nan], estimable=False, n_undefined=n_undef, frac_undefined=frac)
    return dict(ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))], estimable=True, n_undefined=n_undef, frac_undefined=frac)

def c2_site(t, r):
    """§B7 per site: point values (score_pair's) and CIs from the per-site resamples of default_rng(BOOT_SEED)."""
    tab, tgt = view_table(t, r)
    if tab is None: return None
    blk = PS4.blocks_of(tab); R = quant_resamples(tab, tgt, picks(blk, np.random.default_rng(P4.BOOT_SEED)))
    ref = PS4.resample_rhos(tab["s_lr"].values.astype(float), tab[tgt].values.astype(float), blk, np.random.default_rng(P4.BOOT_SEED))
    assert np.array_equal(R["s_lr"], ref, equal_nan=True), f"{r['pair']}: s_lr resamples differ from resample_rhos"
    pt = dict(s_lr=r["rho_s_lr"], GBT_ship=r["rho_GBT_ship"], **{"P-V1": r["rho_P-V1"]}, d_GBT_minus_s_lr=r["d_GBT_minus_s_lr"], d_PV1_minus_s_lr=r["d_PV1_minus_s_lr"])
    out = {q: dict(point=float(pt[q]), **ci_of(R[q])) for q in C2_QUANT}
    assert out["s_lr"]["estimable"] == r["ci_estimable"] and (not r["ci_estimable"] or out["s_lr"]["ci95"] == [r["ci_lo_s_lr"], r["ci_hi_s_lr"]]), f"{r['pair']}: s_lr CI differs from score_pair"
    return dict(pair=r["pair"], status=r["status"], view=r["view"], n_blocks=len(set(blk)), **out)

def c2_pooled(T, rows, pool_row, pool_R):
    """§B7 pooled over the C1 set: per resample the mean over units of each quantity (one default_rng(POOL_SEED), units in ascending
    pair-id order, the same picks as pooled_row); resamples with any undefined unit value dropped and counted."""
    inset = sorted([r for r in rows if r["status"].startswith("in C1")], key=lambda r: r["pair"])
    if not inset: return None
    rng = np.random.default_rng(P4.POOL_SEED); per = {}
    for r in inset:
        tab, tgt = view_table(T[T.pair == r["pair"]], r); R = quant_resamples(tab, tgt, picks(PS4.blocks_of(tab), rng)); per[r["pair"]] = R
        assert np.array_equal(R["s_lr"], pool_R[r["pair"]], equal_nan=True), f"{r['pair']}: pooled s_lr resamples differ from pooled_row"
    out = dict(n_units=len(inset), pairs=[r["pair"] for r in inset], seed=P4.POOL_SEED, n_boot=P4.N_BOOT)
    for q in C2_QUANT:
        M = np.stack([per[r["pair"]][q] for r in inset], 0); anynan = np.isnan(M).any(0); n_drop = int(anynan.sum()); ok = M.mean(0)[~anynan]
        key = {"s_lr": "rho_s_lr", "GBT_ship": "rho_GBT_ship", "P-V1": "rho_P-V1"}.get(q, q); point = float(np.mean([r[key] for r in inset]))
        est = not (n_drop / P4.N_BOOT > P4.UNDEF_MAX_FRAC or len(ok) == 0)
        out[q] = dict(point=point, ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if est else [np.nan, np.nan], estimable=est, n_dropped=n_drop, frac_dropped=n_drop / P4.N_BOOT)
    assert abs(out["s_lr"]["point"] - pool_row["mean_rho"]) < 1e-12 and (not pool_row["estimable"] or out["s_lr"]["ci95"] == pool_row["ci95"]), "pooled s_lr differs from pooled_row"
    return out

def c1_primary(pool_row, n_set):
    """§B6 / spec §3.4."""
    if n_set < MIN_C1_SITES: return dict(reading=f"not testable (fewer than {MIN_C1_SITES} sites in the C1 set: {n_set})", n_sites=n_set, pooled=pool_row)
    return dict(reading="confirmed" if (pool_row and pool_row.get("estimable") and pool_row["ci95"][0] > 0) else "not confirmed", n_sites=n_set, pooled=pool_row,
                rule="confirmed if the 95 % CI lower bound of the pooled mean s_lr rho over the C1-set sites is above zero")

if __name__ == "__main__":
    ap = P5.argparser("Phase 5 scoring"); ap.add_argument("--inputs", required=True); ap.add_argument("--calls", required=True, help="Steve's committed calls file keyed by Q-code (joined to pairs here through the figures key)")
    ap.add_argument("--stats", default=None, help="JSON {pair: pair_stats} from the build (amplitude assert; no-k status)"); ap.add_argument("--bmin", type=int, default=P4.B_MIN_DEFAULT); ap.add_argument("--tag", default="D1"); ap.add_argument("--out", default=None)
    ap.add_argument("--acq_record", default=None, help="rehearsal only: an acquisition step1 run record (per_pair) for the k provenance columns"); ap.add_argument("--rows", default=None, help="rehearsal only: parquet whose rows replace the listed pairs' manifest rows (harness supplement)"); a = ap.parse_args()
    pairs = P5.split_pairs(a.pairs); out = a.out or f"{P5.ROOT[a.mode]}/scores"; results = f"{out}/results_{a.tag}.json"
    m, root, rec = P5.start("phase5_score.py", a.mode, pairs, calls_path=a.calls, results_path=results if a.mode == "phase5" else None); t0 = time.time(); os.makedirs(out, exist_ok=True); dev_min = P4.dev_min_cells()
    if a.mode == "phase5" and (a.acq_record or a.rows): raise P5.Refusal("--acq_record and --rows are rehearsal-only (phase5 mode takes the acquisition run record from its lock and the rows from the supplement)")
    acq = rec.get("acq_run_record") if a.mode == "phase5" else (dict(path=a.acq_record, sha256=P5.sha256_file(a.acq_record), per_pair=json.load(open(a.acq_record)).get("per_pair") or {}) if a.acq_record else None)
    if a.rows: m, _ = P5.harness_rows(m, a.rows, pairs)
    KP = {pid: P5.k_provenance(pid, P5.C4.pair_row(m, pid), acq) for pid in pairs}
    RS = {pid: P5.row_status(P5.C4.pair_row(m, pid)) for pid in pairs}; NT = {p: s for p, s in RS.items() if s.startswith(P5.NOT_TESTABLE)}
    cmap, calls_rec = join_calls(a.calls, f"{root}/figures/key/figset_A_key.json")
    T = pd.read_parquet(a.inputs); T = T[T.pair.isin(pairs)]; assert not set(NT) & set(T.pair), "a not-testable site has scoring inputs"
    stats = json.load(open(a.stats)) if a.stats else {}; cands = [c for c in P4.CANDIDATES if c in T.columns]; assert "s_lr_W64" not in T.columns, "s_lr_W64 is not scored (ruling H1, R10b)"
    role0 = T.role.iloc[0] if len(T) else "phase5"; rows = []
    for pid in pairs:
        if pid in NT: rows.append(dict(pair=pid, unit=P5.unit_of(pid, m), role=role0, bmin=a.bmin, view="not_scorable", status=NT[pid], note="failed eligibility at step 1 (spec §3.3); outside C1", n_tiles_all=0, n_scored=0)); continue
        st = stats.get(pid); tp = T[T.pair == pid]
        if (st and st.get("status") == P4.NO_K) or len(tp) == 0: rows.append(dict(pair=pid, unit=P5.unit_of(pid, m), role=(tp.role.iloc[0] if len(tp) else st.get("role", role0)), bmin=a.bmin, view="not_scorable", status=P4.NO_K, note="no recoverable k (B4 rule 0)", n_tiles_all=0, n_scored=0)); continue
        rows.append(PS4.score_pair(tp, cmap.get(pid, {}), a.bmin, cands, st, dev_min))
    for r in rows: k = KP[r["pair"]]; r.update(k=k["k"], k_sweep_status=k["k_sweep_status"], k_source=k["k_source_class"], k_source_text=k["k_source"], per_k_verdicts=k["per_k_text"], k_agrees_with_acq_record=k["k_agrees_with_acq_record"])
    # C2 per site (every site with a scorable view), before the private view arrays are dropped
    C2_site = {r["pair"]: c2_site(T[T.pair == r["pair"]], r) for r in rows if r.get("view") in ("interior", "fallback")}
    inset = [r for r in rows if r["status"].startswith("in C1")]
    pool_row, pool_R = PS4.pooled_row([r["_view"] for r in inset]); pool_low = PS4.pooled_row([r["_view"] for r in rows if r["status"].startswith("in C1") or r["status"] == "low-amplitude artifact-affected"])[0]
    C1P = c1_primary(pool_row, len(inset)); C1S = PS4.c1_statement(rows); C2P = c2_pooled(T, rows, pool_row, pool_R) if inset else None
    PR = pd.DataFrame([{k: v for k, v in r.items() if k != "_view"} for r in rows]); PR.to_csv(f"{out}/per_pair_{a.tag}.csv", index=False)
    res = dict(mode=a.mode, tag=a.tag, bmin=a.bmin, candidates=cands, dev_min_cells=dev_min, inputs=a.inputs, k_provenance=KP, acq_run_record=({k: v for k, v in acq.items() if k != "per_pair"} if acq else None), harness_rows=(dict(path=a.rows, sha256=P5.sha256_file(a.rows)) if a.rows else None),
               inputs_sha256=P5.sha256_file(a.inputs), calls=a.calls, calls_sha256=P5.sha256_file(a.calls), calls_join=calls_rec, not_testable=NT,
               rules=dict(min_tiles=P4.MIN_TILES, n_boot=P4.N_BOOT, seed=P4.BOOT_SEED, pool_seed=P4.POOL_SEED, undefined_max_frac=P4.UNDEF_MAX_FRAC, lowamp_m=P4.LOWAMP_M, min_c1_sites=MIN_C1_SITES),
               per_pair=[{k: v for k, v in r.items() if k != "_view"} for r in rows], C1_primary=C1P, C1_per_site_secondary=C1S, C1_pooled_with_low_amplitude=pool_low,
               C2=dict(per_site=C2_site, pooled_C1_set=C2P, caveat="descriptive; no superiority claim (spec §3.5)"), seconds=time.time() - t0)
    json.dump(res, open(results, "w"), indent=1, default=float)
    pr = C1P["pooled"]; print(f"scored {len(rows)} sites ({len(NT)} not testable) -> {results}; C1 primary: {C1P['reading']} (C1 set {C1P['n_sites']})" + (f", pooled {pr['mean_rho']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]" if pr else "") + f"; per-site secondary: {C1S['reading']}", flush=True)
