"""R16 must-hold 3.9 — the refusal suite for Phase 5.
Part 1: R12c's 39 tests (R12c/scripts/refusal_tests_r12c.py) ported to phase5_common / the Phase 5 scripts with the REAL v3
(R16/PREREG_phase5_v3.md) wherever a test used the real pre-registration (the dummy sign-off holds the v3 sha256; test 2b uses a
copy of v3 with the GBT hash altered and a sign-off holding that copy's sha256):
  1 / 2a / 2b x the four scripts; 3 no calls file; 4 calls sha256 not in the ledger; 5 results file exists; 6 pair not admitted;
  7 positive control (every lock passes on dummies; the safety net holds: no allow-list applied, every token in place);
  9 no supplement; 10 supplement sha256 not in the ledger; 11a wrong pair id; 11b missing pair id; 12 EW0207 control refuses
  altered references; 13 no-k (build / figures / score); 14 guard after; 27a-e acquisition record (no record, other freeze sha256,
  other supplement sha256, the real PHASE5 directory as it stands, freeze record on disk with another sha256), 27f the ACQ-R09
  rehearsal record accepted; 28 a usable = false row -> product features masked; 30a-e the allow-list on the REAL Phase 5 names
  (string checks only); 29 guard after every test.
Part 2: new tests for must-hold items 3.2 to 3.6 (31-35) and the supplement column / type / eligibility locks (36-37).
The dummy eight-row supplements are built from the ACQ-R09 rehearsal supplement's rows and arrow schema (70 columns) with the
eight Phase 5 pair ids; nothing sealed is opened (every Phase 5 path in these tests is a string or is refused by the guard).
Every dummy lives in a temporary directory deleted at the end; nothing is committed. Writes R16/tables/refusal_tests.json."""
import json, os, shutil, sys, tempfile, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import r16_common as R
P5 = R.P5; P4 = P5.P4
import phase5_build as P5B, phase5_score as P5S, phase4_score as PS4, phase4_predict as PP4, bar_v1_1 as B
OUT = "/scratch/users/scdobbs/grl_review/R16/refusal_tests"; os.makedirs(OUT, exist_ok=True); V3 = P5.PREREG_V3
def run(name, fn, expect, exc=P4.Refusal):
    try: fn(); return dict(test=name, passed=False, exception=None, note="NO REFUSAL")
    except exc as e: return dict(test=name, passed=(expect in str(e)), exception=f"{type(e).__name__}: {e}"[:300])
    except Exception as e: return dict(test=name, passed=False, exception=f"{type(e).__name__}: {e}"[:300])
def acq_record(path, freeze_sha, sup_sha, status="complete", mode="phase5"):
    json.dump(dict(directive="R16 dummy", mode=mode, status=status, locks=dict(lock2_freeze_record=dict(freeze_record_sha256=freeze_sha)), freeze_record_sha256=freeze_sha, supplement_sha256=sup_sha, per_pair={}), open(path, "w"))
def write_sup(df, path, schema): pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), path); return P5.sha256_file(path)
if __name__ == "__main__":
    assert os.path.exists(V3), V3; v3_sha = P5.sha256_file(V3); m0 = P5.C4.load_manifest(); P5.R9.install_guard_r09(m0); P5.install_guard_p5()
    n, probes, _, _ = P5.R9.guard_selftest_r09(m0); n5, p5 = P5.guard_selftest_p5(); assert n == probes == 702 and n5 == p5; n_tokens = len(P5.R9.SEALED_TOKENS); n_p5tok = len(P5.P5_TOKENS)
    tmp = tempfile.mkdtemp(prefix="r16_refusal_", dir=OUT); site = [P5.SEALED8[0]]
    good_signoff = f"{tmp}/SIGNOFF_dummy.md"; open(good_signoff, "w").write(f"DUMMY sign-off for the refusal tests (never committed): {v3_sha}\n"); bad_signoff = f"{tmp}/SIGNOFF_bad.md"; open(bad_signoff, "w").write("1" * 64 + "\n")
    bad_art_prereg = f"{tmp}/PREREG_badart.md"; open(bad_art_prereg, "w").write(open(V3).read().replace(P4.GBT_SHA, "2" * 64)); assert P5.sha256_file(bad_art_prereg) != v3_sha; bad_art_signoff = f"{tmp}/SIGNOFF_badart.md"; open(bad_art_signoff, "w").write(P5.sha256_file(bad_art_prereg) + "\n")
    calls = f"{tmp}/calls.csv"; open(calls, "w").write("code,relief,criteria,dominant,ab_whole_survey,note\nQ01,yes,none,relief,na,\n"); calls_sha = P5.sha256_file(calls)
    # dummy supplements: the ACQ-R09 rehearsal rows (70 columns, its arrow schema) relabelled with the eight Phase 5 ids; sites 7 and 8 not testable
    schema = pq.read_schema(R.REH_SUP); S = pd.read_parquet(R.REH_SUP); base = S[S.eligible.astype(bool)].reset_index(drop=True); rows = []
    for i, pid in enumerate(P5.SEALED8):
        r = base.iloc[i % len(base)].copy(); r["pair_id"] = pid; r["leakage_unit"] = f"p5u{i+1:02d}"
        if i >= 6: r["coreg_pass"] = False; r["eligible"] = False; r["eligibility_reason"] = "1 co-registration (R16 refusal-test dummy)"
        rows.append(r)
    G = pd.DataFrame(rows).reset_index(drop=True); NT_IDS = list(P5.SEALED8[6:]); EL_IDS = list(P5.SEALED8[:6])
    sup_good = f"{tmp}/pairs_phase5_sealed.parquet"; sup_sha = write_sup(G, sup_good, schema)
    bad = G.copy(); bad.loc[0, "pair_id"] = "AR26__MGDS_31838"; sup_badid = f"{tmp}/sup_badid.parquet"; sup_badid_sha = write_sup(bad, sup_badid, schema)
    sup_missing = f"{tmp}/sup_missing.parquet"; sup_missing_sha = write_sup(G.iloc[:7], sup_missing, schema)
    sup_63 = f"{tmp}/sup_63.parquet"; G.drop(columns=["notes"]).to_parquet(sup_63, index=False); sup_63_sha = P5.sha256_file(sup_63)
    ge = G.copy(); ge["extra_col"] = 1; sup_extra = f"{tmp}/sup_extra.parquet"; ge.to_parquet(sup_extra, index=False); sup_extra_sha = P5.sha256_file(sup_extra)
    sup_type = f"{tmp}/sup_type.parquet"; G.to_parquet(sup_type, index=False); sup_type_sha = P5.sha256_file(sup_type)     # pandas round trip: all-null string columns written as null
    gb = G.copy(); gb.loc[0, "coreg_pass"] = False; sup_badelig = f"{tmp}/sup_badelig.parquet"; sup_badelig_sha = write_sup(gb, sup_badelig, schema)
    ALLSHA = [calls_sha, sup_sha, sup_badid_sha, sup_missing_sha, sup_63_sha, sup_extra_sha, sup_type_sha, sup_badelig_sha]
    ledger_with = f"{tmp}/ledger_with.md"; open(ledger_with, "w").write("| " + " | ".join(ALLSHA) + " |\n"); ledger_nosup = f"{tmp}/ledger_nosup.md"; open(ledger_nosup, "w").write(f"| calls {calls_sha} |\n"); ledger_nocalls = f"{tmp}/ledger_nocalls.md"; open(ledger_nocalls, "w").write(f"| supplement {sup_sha} |\n")
    results = f"{tmp}/results_D1.json"; open(results, "w").write("{}")
    def acqdir(name, fz, ss):
        d = f"{tmp}/{name}"; os.makedirs(d); acq_record(f"{d}/step1_run_phase5_dummy.json", fz, ss); return d
    acq_ok = acqdir("acq_ok", P5.ACQ_FREEZE_SHA, sup_sha); acq_empty = f"{tmp}/acq_empty"; os.makedirs(acq_empty); acq_badfz = acqdir("acq_badfz", "3" * 64, sup_sha); acq_badsup = acqdir("acq_badsup", P5.ACQ_FREEZE_SHA, "4" * 64)
    for nm, ss in (("acq_badid", sup_badid_sha), ("acq_missing", sup_missing_sha), ("acq_63", sup_63_sha), ("acq_extra", sup_extra_sha), ("acq_type", sup_type_sha), ("acq_badelig", sup_badelig_sha)): globals()[nm] = acqdir(nm, P5.ACQ_FREEZE_SHA, ss)
    L = dict(signoff=good_signoff, prereg_path=V3, calls_path=calls, ledger=ledger_with, supplement=sup_good, manifest=m0, acq_records_dir=acq_ok); T = []
    # ---------------- Part 1: R12c's 39 tests, ported ----------------
    for sc in P5.SCRIPTS[1:]:
        T.append(run(f"1 {sc}: no sign-off file", lambda sc=sc: P5.enforce_locks("phase5", sc, site, **{**L, "signoff": f"{tmp}/absent.md"}), "LOCK sign-off"))
        T.append(run(f"2a {sc}: wrong pre-registration sha256 (real v3, sign-off holding another sha256)", lambda sc=sc: P5.enforce_locks("phase5", sc, site, **{**L, "signoff": bad_signoff}), "LOCK pre-registration"))
        T.append(run(f"2b {sc}: wrong artifact hash (copy of v3 with the GBT hash altered)", lambda sc=sc: P5.enforce_locks("phase5", sc, site, **{**L, "signoff": bad_art_signoff, "prereg_path": bad_art_prereg}), "LOCK artifact"))
    T.append(run("3 phase5_predict.py: no calls file", lambda: P5.enforce_locks("phase5", "phase5_predict.py", site, **{**L, "calls_path": f"{tmp}/absent_calls.csv"}), "LOCK calls file"))
    T.append(run("4 phase5_score.py: calls sha256 not in the ledger (ledger holds the supplement sha256 only)", lambda: P5.enforce_locks("phase5", "phase5_score.py", site, **{**L, "ledger": ledger_nocalls, "results_path": f"{tmp}/absent.json"}), "LOCK calls file"))
    T.append(run("5 phase5_score.py: results file already exists", lambda: P5.enforce_locks("phase5", "phase5_score.py", site, **{**L, "results_path": results}), "LOCK single run"))
    T.append(run("6 any script: pair not admitted (development pair in phase5 mode)", lambda: P5.enforce_locks("phase5", "phase5_build.py", ["AR26__MGDS_31838"], **L), "LOCK pair list"))
    try:
        r = P5.enforce_locks("phase5", "phase5_build.py", list(P5.SEALED8), **L); v = r["verify"]
        ok = (set(r["manifest"].pair_id) == set(m0.pair_id) | set(P5.SEALED8) and r["supplement"]["n_rows"] == 8 and r["signoff_sha256"] == v3_sha and v["prereg"]["sha256"] == v3_sha and r["acq_run_record"]["status"] == "complete"
              and all(v[sc]["sha256"] == v[sc]["expected"] for sc in P5.SCRIPTS + P4.SCRIPTS) and sorted(r["admitted"]) == sorted(EL_IDS) and sorted(r["not_testable"]) == sorted(NT_IDS))
        net = (P5._admitted == () and len(P5.R9.SEALED_TOKENS) == n_tokens and len(P5.P5_TOKENS) == n_p5tok)
        T.append(dict(test="7 positive control: real v3 + dummy sign-off holding its sha256 + dummy supplement / ledger / acquisition record -> every lock passes (10 script hashes, 6 eligible / 2 not testable); stopped on the safety net (no allow-list applied, every token in place)", passed=bool(ok and net), exception=f"locks passed={ok}; safety net={net}"))
    except Exception as e: T.append(dict(test="7 positive control", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    T.append(run("9 phase5_build.py: no supplement file", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "supplement": f"{tmp}/absent_sup.parquet"}), "LOCK supplement (phase5)"))
    T.append(run("10 supplement sha256 not in the ledger", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "ledger": ledger_nosup}), "LOCK supplement (phase5): sha256"))
    T.append(run("11a supplement with a wrong pair id", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "supplement": sup_badid, "acq_records_dir": acq_badid}), "LOCK supplement (phase5): pair ids"))
    T.append(run("11b supplement with a missing pair id", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "supplement": sup_missing, "acq_records_dir": acq_missing}), "LOCK supplement (phase5): pair ids"))
    import joblib; gbt = joblib.load(P4.GBT_ARTIFACT); models = PP4.load_models(); ew = P4.R8.EW_PAIR
    alt_dir = f"{tmp}/oak_r10"; os.makedirs(f"{alt_dir}/rehearsal/predictions", exist_ok=True)
    for sub in ("inputs", "build"): os.symlink(f"{P4.R10_OAK}/rehearsal/{sub}", f"{alt_dir}/rehearsal/{sub}")
    Rr = pd.read_parquet(f"{P4.R10_OAK}/rehearsal/predictions/{ew}.parquet"); Rr["P-V1"] = Rr["P-V1"] * 10 ** 1e-3; Rr.to_parquet(f"{alt_dir}/rehearsal/predictions/{ew}.parquet")
    real_verify = P4.R8.oak_verify
    def fake_verify(sub, rel, path):
        if rel == f"predictions/{ew}.parquet": return dict(sha256="dummy")
        return real_verify(sub, rel, path)
    P4.R8.oak_verify = fake_verify; P4.R10_OAK = alt_dir
    T.append(run("12 phase5_predict.py (phase4_predict.environment_control, imported): EW0207 reference predictions altered by 1e-3 in log10", lambda: PP4.environment_control(gbt, models, 8), "LOCK environment control"))
    P4.R8.oak_verify = real_verify; P4.R10_OAK = f"{P4.R8.OAK_ARCHIVE}/R10"
    mk = m0.copy(); i = mk.index[mk.pair_id == "KM0923__MGDS_31429"][0]; mk.at[i, "max_recoverable_k"] = np.nan; root13 = f"{tmp}/root13"
    try:
        st = P5B.build_pair(mk, "KM0923__MGDS_31429", root13, 2, log=lambda *a, **k: None); built = sorted(os.listdir(f"{root13}/build/KM0923__MGDS_31429")); fields = os.path.exists(f"{root13}/fields/KM0923__MGDS_31429"); inputs = os.path.exists(f"{root13}/inputs/KM0923__MGDS_31429")
        key_skip = json.load(open(f"{root13}/build/KM0923__MGDS_31429/pair_stats.json")).get("status") == P4.NO_K
        T.append(dict(test="13 build: KM0923 row with k emptied (rehearsal, scratch root)", passed=(st["status"] == P4.NO_K and built == ["pair_stats.json"] and not fields and not inputs), exception=f"status={st['status']}; files built={built}; fields={fields}; inputs={inputs}"))
        T.append(dict(test="13 figures: no page for the no-k pair (pair_stats status read by phase5_figures)", passed=key_skip, exception=f"skip={key_skip}"))
        rows13 = [dict(pair="KM0923__MGDS_31429", unit="lu07", role="phase5", bmin=8, view="not_scorable", status=P4.NO_K, note="no recoverable k (B4 rule 0)", n_tiles_all=0, n_scored=0)]
        c1 = PS4.c1_statement(rows13); T.append(dict(test="13 score: no-k pair reported 'not scorable (no k)', outside C1 (secondary not testable; primary not testable)", passed=(c1["reading"].startswith("not testable") and P5S.c1_primary(None, 0)["reading"].startswith("not testable")), exception=f"secondary: {c1['reading']}; primary: {P5S.c1_primary(None, 0)['reading']}"))
    except Exception as e: T.append(dict(test="13 no-k behaviour", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    n2, pr2, _, _ = P5.R9.guard_selftest_r09(m0); n52, p52 = P5.guard_selftest_p5(); T.append(dict(test="14 guard after the ported lock tests: every Phase 4 and Phase 5 probe still blocked", passed=(n2 == pr2 == 702 and n52 == p52), exception=f"{n2}/{pr2}; phase5 {n52}/{p52}"))
    T.append(run("27a phase5_build.py: no acquisition run record (empty records directory)", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "acq_records_dir": acq_empty}), "LOCK acquisition run record (phase5): no step1_run_phase5"))
    T.append(run("27b phase5_build.py: acquisition run record citing a different freeze record sha256", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "acq_records_dir": acq_badfz}), "LOCK acquisition run record (phase5): no record"))
    T.append(run("27c phase5_build.py: acquisition run record citing a different supplement sha256", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "acq_records_dir": acq_badsup}), "LOCK acquisition run record (phase5): no record"))
    T.append(run("27d phase5_build.py: the acquisition repo's PHASE5 directory as it stands (freeze record only, no run record)", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "acq_records_dir": P5.ACQ_RECORDS_DIR}), "LOCK acquisition run record (phase5): no step1_run_phase5"))
    T.append(run("27e phase5_build.py: freeze record on disk with another sha256", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "acq_freeze_record": bad_art_prereg}), "LOCK acquisition run record (phase5): freeze record"))
    try:
        a27 = P5.load_acq_run_record(R.REH_SUP_SHA, records_dir=P5.ACQ_REHEARSAL_DIR, record_mode="rehearsal"); ok27 = (os.path.basename(a27["path"]) == "step1_run_rehearsal_r09.json" and a27["sha256"].startswith(R.REH_RECORD_SHA8) and a27["freeze_record_sha256"] == P5.ACQ_FREEZE_SHA and len(a27["per_pair"]) == 4)
        T.append(dict(test="27f (3.2) the ACQ-R09 rehearsal record (step1_run_rehearsal_r09.json) accepted against the real freeze record and the supplement it cites (record_mode rehearsal)", passed=bool(ok27), exception=f"accepted {os.path.basename(a27['path'])} {a27['sha256'][:12]}; per_pair {sorted(a27['per_pair'])}"))
    except Exception as e: T.append(dict(test="27f rehearsal record accepted", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    try:
        hs = pd.read_parquet("/scratch/users/scdobbs/grl_review/R11/rehearsal/harness/harness_supplement.parquet"); h = hs[hs.pair_id == "TN399__MGDS_33068"].copy(); h["ship_products_usable"] = False; h["ship_products_usable_reason"] = "refusal test 28: flag set false (harness)"
        mh = pd.concat([m0[m0.pair_id != "TN399__MGDS_33068"], h], ignore_index=True); root28 = f"{tmp}/root28"; st = P5B.build_pair(mh, "TN399__MGDS_33068", root28, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), log=lambda *a, **k: None)
        f = pd.read_parquet(f"{root28}/build/TN399__MGDS_33068/features.parquet"); allnan = bool(np.isnan(f[B.CONTRACT].values.astype(float)).all()); frozen = P4.R8.load_features(); fr = frozen[frozen.pair == "TN399__MGDS_33068"]; frozen_used = bool(np.isfinite(fr[B.CONTRACT].values.astype(float)).any())
        T.append(dict(test="28 supplement row with ship_products_usable = false -> product features masked (TN399 harness row, phase5_build.build_pair)", passed=bool(allnan and st["product_features"] == "masked" and st["ship_products_usable"] is False and frozen_used), exception=f"contract columns all NaN={allnan} ({len(f)} rows); product_features={st['product_features']}; frozen table finite={frozen_used}"))
    except Exception as e: T.append(dict(test="28 product features masked", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    # 30: the allow-list on the REAL Phase 5 names (string checks; nothing opened)
    try:
        hp = {pid: f"{P5.ACQ_OUT_ROOT}/harmonized/{pid}/hr.tif" for pid in P5.SEALED8}; pre = {p: P5.R9._is_sealed(q) for p, q in hp.items()}
        P5.apply_allowlist(P5.SEALED8); post = {p: P5.R9._is_sealed(q) for p, q in hp.items()}
        fake1 = P5.R9._is_sealed(f"{P5.ACQ_OUT_ROOT}/harmonized/{R.FAKE_ID}/hr.tif"); fake2 = {s: P5.R9._is_sealed(s) for s in (f"{P5.ACQ_OUT_ROOT}/harmonized/hr.tif", f"{P5.ACQ_OUT_ROOT}/harmonized", P5.ACQ_OUT_ROOT, P5.SEALED_CANDIDATES_V3)}
        T.append(dict(test="30a the eight real Phase 5 HR paths sealed BEFORE the allow-list", passed=all(pre.values()), exception=str(pre)[:300]))
        T.append(dict(test="30b the eight real Phase 5 HR paths admitted AFTER apply_allowlist(SEALED8)", passed=not any(post.values()), exception=str(post)[:300]))
        T.append(dict(test="30c phase5/harmonized/FAKE__MGDS_00000/hr.tif refused after the allow-list", passed=bool(fake1), exception=f"is_sealed={fake1}"))
        T.append(dict(test="30d the bare roots (phase5/harmonized/hr.tif, phase5/harmonized, phase5, sealed_candidates_v3) refused after the allow-list", passed=all(fake2.values()), exception=str(fake2)[:300]))
        P5._admitted = (); a0 = P5.SEALED8[0]; P5.apply_allowlist([a0]); c0, _, h0 = a0.partition("__")
        fam = {"sealed_candidates_v3/<hr id> of the admitted site": f"{P5.SEALED_CANDIDATES_V3}/{h0}/f.grd.gz", "raw_lr_swath/<cruise> of the admitted site": f"{P5.C4.DATA}/raw_lr_swath/{c0}/a.mb58", "another site's HR": hp[P5.SEALED8[1]],
               "MGDS_31811 (unused product)": f"{P5.SEALED_CANDIDATES_V3}/MGDS_31811", "Phase 4 RC2511 HR": os.path.join(P5.C4.DATA, "harmonized_confirmatory/nu01b/RC2511__MGDS_32239/hr.tif"), "CNN output of another site": f"{P5.ROOT['phase5']}/build/{P5.SEALED8[2]}/pair_stats.json"}
        f30 = {k: P5.R9._is_sealed(v) for k, v in fam.items()}; adm = not P5.R9._is_sealed(hp[a0]) and not P5.R9._is_sealed(f"{P5.ROOT['phase5']}/build/{a0}/pair_stats.json")
        T.append(dict(test="30e only one site admitted: its acquisition and CNN-output paths allowed; its sealed_candidates_v3 / raw swath dirs, another site, MGDS_31811 and a Phase 4 path refused", passed=bool(all(f30.values()) and adm), exception=f"admitted ok={adm}; {f30}"[:300]))
        P5._admitted = ()
    except Exception as e: T.append(dict(test="30 allow-list cases", passed=False, exception=f"{type(e).__name__}: {e}"[:300])); P5._admitted = ()
    # ---------------- Part 2: must-hold items 3.2 to 3.6 and the supplement locks ----------------
    T.append(run("31b (3.2) acquisition lock: a wrong freeze sha256 expected (rehearsal record directory)", lambda: P5.load_acq_run_record(R.REH_SUP_SHA, records_dir=P5.ACQ_REHEARSAL_DIR, record_mode="rehearsal", freeze_sha="0" * 64), "LOCK acquisition run record (phase5): freeze record"))
    T.append(run("31c (3.2) acquisition lock: a wrong supplement sha256 (rehearsal record directory)", lambda: P5.load_acq_run_record("5" * 64, records_dir=P5.ACQ_REHEARSAL_DIR, record_mode="rehearsal"), "LOCK acquisition run record (phase5): no record"))
    T.append(run("31d (3.2) acquisition lock: the rehearsal record is refused in phase5 record mode (mode test)", lambda: P5.load_acq_run_record(R.REH_SUP_SHA, records_dir=P5.ACQ_REHEARSAL_DIR, record_mode="phase5"), "no step1_run_phase5"))
    try:
        P5.apply_allowlist(P5.SEALED8[:7]); r32 = {"8th site (not admitted)": P5.R9._is_sealed(f"{P5.ACQ_OUT_ROOT}/harmonized/{P5.SEALED8[7]}/hr.tif"), "bare phase5 root": P5.R9._is_sealed(P5.ACQ_OUT_ROOT), "bare sealed_candidates_v3": P5.R9._is_sealed(P5.SEALED_CANDIDATES_V3)}
        T.append(dict(test="32a (3.3) seven sites admitted: the eighth site's path and the bare real roots stay refused", passed=all(r32.values()), exception=str(r32))); P5._admitted = ()
    except Exception as e: T.append(dict(test="32a", passed=False, exception=str(e)[:300])); P5._admitted = ()
    T.append(run("32b (3.3) allow-list refuses an id that is not a site id (phase5)", lambda: P5.apply_allowlist(["AR26__MGDS_31838"]), "LOCK allow-list (phase5)"))
    try:
        mg, sr = P5.load_supplement(m0, sup_good, ledger_with); st33 = {pid: P5.row_status(P5.C4.pair_row(mg, pid)) for pid in NT_IDS}
        T.append(dict(test="33a (3.4) load_supplement: eligible = False rows are 'not testable (<reason>)' and outside the admitted list", passed=bool(sorted(sr["not_testable"]) == sorted(NT_IDS) and not set(NT_IDS) & set(sr["eligible"]) and all(v.startswith("not testable (1 co-registration") for v in sr["not_testable"].values())), exception=str(sr["not_testable"])[:300]))
        T.append(dict(test="33b (3.4) row_status of an eligible = False row = 'not testable (<eligibility_reason>)' (checked before paths / k)", passed=all(v.startswith("not testable (") for v in st33.values()), exception=str(st33)[:300]))
        root33 = f"{tmp}/root33"; os.makedirs(root33); s33 = P5B.build_pair(mg, NT_IDS[0], root33, 2, log=lambda *a, **k: None)
        T.append(dict(test="33c (3.4) phase5_build.build_pair on a not-testable row: status only, nothing written", passed=bool(s33["status"].startswith("not testable") and os.listdir(root33) == []), exception=f"{s33['status']}; root33 entries {os.listdir(root33)}"))
        P5.apply_allowlist(sr["eligible"]); nt_sealed = P5.R9._is_sealed(f"{P5.ACQ_OUT_ROOT}/harmonized/{NT_IDS[0]}/hr.tif"); P5._admitted = ()
        T.append(dict(test="33d (3.4) after apply_allowlist(eligible) the not-testable site's paths stay refused", passed=bool(nt_sealed), exception=f"is_sealed={nt_sealed}"))
    except Exception as e: T.append(dict(test="33 not-testable cases", passed=False, exception=f"{type(e).__name__}: {e}"[:300])); P5._admitted = ()
    pr_pos = dict(mean_rho=0.3, ci95=[0.1, 0.5], estimable=True); pr_neg = dict(mean_rho=0.1, ci95=[-0.1, 0.3], estimable=True); pr_ne = dict(mean_rho=0.1, ci95=[np.nan, np.nan], estimable=False)
    T.append(dict(test="34a (3.5) C1 primary with 2 C1 sites -> not testable", passed=P5S.c1_primary(pr_pos, 2)["reading"].startswith("not testable"), exception=P5S.c1_primary(pr_pos, 2)["reading"]))
    T.append(dict(test="34b (3.5) C1 primary with 3 sites and CI lower bound > 0 -> confirmed", passed=P5S.c1_primary(pr_pos, 3)["reading"] == "confirmed", exception=P5S.c1_primary(pr_pos, 3)["reading"]))
    T.append(dict(test="34c (3.5) C1 primary with 3 sites and CI lower bound <= 0 / CI not estimable -> not confirmed", passed=(P5S.c1_primary(pr_neg, 3)["reading"] == "not confirmed" and P5S.c1_primary(pr_ne, 8)["reading"] == "not confirmed"), exception=f"{P5S.c1_primary(pr_neg, 3)['reading']}; {P5S.c1_primary(pr_ne, 8)['reading']}"))
    try:
        rng = np.random.default_rng(7); n = 120; blk = [(i // 12, i % 3) for i in range(n)]; y = rng.normal(size=n); tab = pd.DataFrame(dict(s_lr=y + rng.normal(size=n), GBT_ship=y + rng.normal(size=n), **{"P-V1": rng.normal(size=n)}, A_fine=y))
        Rq = P5S.quant_resamples(tab, "A_fine", P5S.picks(blk, np.random.default_rng(P4.BOOT_SEED))); ref = PS4.resample_rhos(tab.s_lr.values, tab.A_fine.values, blk, np.random.default_rng(P4.BOOT_SEED))
        same = np.array_equal(Rq["s_lr"], ref, equal_nan=True) and np.allclose(Rq["d_GBT_minus_s_lr"], Rq["GBT_ship"] - Rq["s_lr"])
        T.append(dict(test="34d (3.5) C2 resamples: the s_lr resamples of phase5_score.picks equal phase4_score.resample_rhos (same seed, same draws); differences computed per resample", passed=bool(same), exception=f"identical={same}"))
        units = [dict(pair=f"P{k}", pred=tab.s_lr.values, target=tab.A_fine.values, blk=blk) for k in range(3)]; row, Rp = PS4.pooled_row(units); rng2 = np.random.default_rng(P4.POOL_SEED)
        okp = all(np.array_equal(P5S.quant_resamples(tab, "A_fine", P5S.picks(blk, rng2))["s_lr"], Rp[f"P{k}"], equal_nan=True) for k in range(3))
        T.append(dict(test="34e (3.5) C2 pooled resamples: per-unit s_lr draws of one default_rng(20261013) equal pooled_row's", passed=bool(okp), exception=f"identical={okp}"))
    except Exception as e: T.append(dict(test="34d/e C2 resamples", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    kp = f"{tmp}/key.json"; json.dump(dict(order=[dict(code="Q01", pair="A__MGDS_1"), dict(code="Q02", pair="B__MGDS_2")]), open(kp, "w"))
    c_ok = f"{tmp}/c_ok.csv"; open(c_ok, "w").write("code,relief,criteria,dominant,ab_whole_survey,note\nQ02,weak,a;c,artifact,yes,stripes 1\nQ01,yes,none,relief,na,\n")
    try:
        cm, cr = P5S.join_calls(c_ok, kp); raw = pd.read_csv(c_ok).fillna("")
        ok35 = (cr["code_to_pair"] == {"Q01": "A__MGDS_1", "Q02": "B__MGDS_2"} and cm["B__MGDS_2"] == raw.iloc[0].to_dict() and cm["A__MGDS_1"] == raw.iloc[1].to_dict())
        T.append(dict(test="35a (3.6) join_calls: code -> pair through the key; every value of the file carried unchanged (phase4_score parsing)", passed=bool(ok35), exception=str(cm)[:300]))
    except Exception as e: T.append(dict(test="35a join", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    for nm, body, why in (("35b", "code,relief,criteria,dominant,ab_whole_survey,note\nQ01,yes,none,relief,na,\nQ03,yes,none,relief,na,\n", "a code not in the key"), ("35c", "code,relief,criteria,dominant,note\nQ01,yes,none,relief,\nQ02,yes,none,relief,\n", "a missing column"),
                          ("35d", "code,relief,criteria,dominant,ab_whole_survey,note\nQ01,yes,none,relief,na,\nQ01,yes,none,relief,na,\n", "a duplicated code")):
        p = f"{tmp}/{nm}.csv"; open(p, "w").write(body); T.append(run(f"{nm} (3.6) join_calls refuses {why}", lambda p=p: P5S.join_calls(p, kp), "", exc=AssertionError))
    T.append(run("36 supplement whose eligible disagrees with items 1-3 (coreg_pass False, eligible True)", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "supplement": sup_badelig, "acq_records_dir": acq_badelig}), "but spec §3.3 items 1-3 give"))
    T.append(run("37a supplement with 63 of the 64 Phase 4 columns", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "supplement": sup_63, "acq_records_dir": acq_63}), "LOCK supplement (phase5): the first"))
    T.append(run("37b supplement with a seventh extra column", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "supplement": sup_extra, "acq_records_dir": acq_extra}), "LOCK supplement (phase5): extra columns"))
    T.append(run("37c supplement with changed arrow types (pandas round trip: all-null string columns written as null)", lambda: P5.enforce_locks("phase5", "phase5_build.py", site, **{**L, "supplement": sup_type, "acq_records_dir": acq_type}), "LOCK supplement (phase5): arrow types"))
    n3, pr3, _, _ = P5.R9.guard_selftest_r09(m0); n53, p53 = P5.guard_selftest_p5(); T.append(dict(test="29 guard after every test: every Phase 4 and Phase 5 probe still blocked, empty allow-list", passed=(n3 == pr3 == 702 and n53 == p53 and P5._admitted == ()), exception=f"{n3}/{pr3}; phase5 {n53}/{p53}; allow-list {P5._admitted}"))
    shutil.rmtree(tmp); assert not os.path.exists(tmp)
    rec = dict(n_tests=len(T), n_passed=sum(t["passed"] for t in T), n_ported=len([t for t in T if t["test"].split()[0][:2] not in ("31", "32", "33", "34", "35", "36", "37")]), tests=T, prereg_v3=dict(path=V3, sha256=v3_sha), guard=dict(r09=[n3, pr3], phase5=[n53, p53]),
               dummy_signoff_deleted=not os.path.exists(good_signoff), time=time.strftime("%Y-%m-%dT%H:%M:%S"), node=os.uname().nodename)
    os.makedirs(f"{R.DR}/R16/tables", exist_ok=True); json.dump(rec, open(f"{R.DR}/R16/tables/refusal_tests.json", "w"), indent=1, default=str)
    for t in T: print(("PASS" if t["passed"] else "FAIL"), t["test"], "|", (t["exception"] or "")[:110])
    print(f"{rec['n_passed']}/{rec['n_tests']} passed ({rec['n_ported']} ported from R12c)")
