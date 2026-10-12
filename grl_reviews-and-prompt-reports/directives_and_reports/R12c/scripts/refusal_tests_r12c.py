"""R12c must-hold 5 — DERIVED from R11/scripts/refusal_tests_r11.py by derive_refusal_tests_r12c.py (diff in R12c/tables/refusal_tests_derivation.diff): R12c output dir; test 28 harness pinned to the R11 root; 27f (real acquisition directory accepted); 30a-e (B2). The R11 docstring follows.
R11 §D2 — the 26 R10b refusal tests re-run against the final scripts with the REAL PREREG_phase4_v2.md in place of the dummy
pre-registration wherever a test used one (the dummy sign-off holds the real v2 sha256; test 2b uses a copy of v2 with the GBT
hash altered and a sign-off holding that copy's sha256), plus
  27a no acquisition run record (empty records dir), 27b a record citing a different freeze record, 27c a record citing a different
      supplement sha256, 27d the acquisition repo's own ACQ-R06b directory as it stands (holds only the stopped refusal-test record):
      phase4_build.py refuses in each case;
  28  a supplement row with ship_products_usable = false (harness: the TN399__MGDS_33068 acquisition rehearsal row with the flag set
      false, built in rehearsal mode to a scratch root): the 14 contract (product) columns are NaN on every feature row, pair_stats says
      product_features = masked;
  29  the guard still blocks 702 / 702 afterwards.
The positive control (7) uses a DUMMY sign-off holding the real v2 sha256 and a dummy acquisition record citing the dummy supplement;
it must stop on the safety net (enforce_locks returns; the allow-list is never applied: _allowed stays empty and every sealed token
remains in the guard). Every dummy lives in a temporary directory deleted at the end; nothing is committed."""
import json, os, shutil, sys, tempfile, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, phase4_build as PB, phase4_score as PS, phase4_predict as PP, bar_v1_1 as B
OUT = "/scratch/users/scdobbs/grl_review/R12c/refusal_tests"; os.makedirs(OUT, exist_ok=True); V2 = P4.PREREG_V2
def run(name, fn, expect):
    try: fn(); return dict(test=name, passed=False, exception=None, note="NO REFUSAL")
    except P4.Refusal as e: return dict(test=name, passed=(expect in str(e)), exception=f"Refusal: {e}"[:300])
    except Exception as e: return dict(test=name, passed=False, exception=f"{type(e).__name__}: {e}"[:300])
def acq_record(path, freeze_sha, sup_sha, status="complete", mode="phase4"):
    json.dump(dict(directive="ACQ-R06b", mode=mode, status=status, locks=dict(lock2_freeze_record=dict(freeze_record_sha256=freeze_sha)), supplement_sha256=sup_sha, per_pair={}), open(path, "w"))
if __name__ == "__main__":
    assert os.path.exists(V2), V2; v2_sha = P4.sha256_file(V2); m0 = P4.C4.load_manifest(); P4.R9.install_guard_r09(m0); n, probes, _, _ = P4.R9.guard_selftest_r09(m0); assert n == probes == 702; n_tokens = len(P4.R9.SEALED_TOKENS)
    tmp = tempfile.mkdtemp(prefix="r12c_refusal_", dir=OUT); sealed = [P4.CONFIRMATORY[0]]
    good_signoff = f"{tmp}/SIGNOFF_dummy.md"; open(good_signoff, "w").write(f"DUMMY sign-off for the refusal tests (never committed): {v2_sha}\n"); bad_signoff = f"{tmp}/SIGNOFF_bad.md"; open(bad_signoff, "w").write("1" * 64 + "\n")
    bad_art_prereg = f"{tmp}/PREREG_badart.md"; open(bad_art_prereg, "w").write(open(V2).read().replace(P4.GBT_SHA, "2" * 64)); assert P4.sha256_file(bad_art_prereg) != v2_sha; bad_art_signoff = f"{tmp}/SIGNOFF_badart.md"; open(bad_art_signoff, "w").write(P4.sha256_file(bad_art_prereg) + "\n")
    calls = f"{tmp}/calls.csv"; open(calls, "w").write("code,relief,criteria,dominant,ab_whole_survey,note\nQ01,yes,none,relief,na,\n"); calls_sha = P4.sha256_file(calls)
    sup_good = f"{tmp}/pairs_phase4_sealed.parquet"; m0[m0.pair_id.isin(P4.SEALED5)].reset_index(drop=True).to_parquet(sup_good); sup_sha = P4.sha256_file(sup_good)
    sup_badid = f"{tmp}/pairs_phase4_sealed_badid.parquet"; bad = m0[m0.pair_id.isin(P4.SEALED5)].reset_index(drop=True).copy(); bad.loc[0, "pair_id"] = "AR26__MGDS_31838"; bad.to_parquet(sup_badid); sup_badid_sha = P4.sha256_file(sup_badid)
    sup_missing = f"{tmp}/pairs_phase4_sealed_missing.parquet"; m0[m0.pair_id.isin(P4.SEALED5[:4])].reset_index(drop=True).to_parquet(sup_missing); sup_missing_sha = P4.sha256_file(sup_missing)
    ledger_with = f"{tmp}/ledger_with.md"; open(ledger_with, "w").write(f"| calls {calls_sha} | supplement {sup_sha} | {sup_badid_sha} | {sup_missing_sha} |\n"); ledger_nosup = f"{tmp}/ledger_nosup.md"; open(ledger_nosup, "w").write(f"| calls {calls_sha} |\n"); ledger_nocalls = f"{tmp}/ledger_nocalls.md"; open(ledger_nocalls, "w").write(f"| supplement {sup_sha} |\n")
    results = f"{tmp}/results_D1.json"; open(results, "w").write("{}")
    acq_ok = f"{tmp}/acq_ok"; os.makedirs(acq_ok); acq_record(f"{acq_ok}/step1_run_phase4_run.json", P4.ACQ_FREEZE_SHA, sup_sha)
    acq_empty = f"{tmp}/acq_empty"; os.makedirs(acq_empty); acq_badfz = f"{tmp}/acq_badfz"; os.makedirs(acq_badfz); acq_record(f"{acq_badfz}/step1_run_phase4_run.json", "3" * 64, sup_sha); acq_badsup = f"{tmp}/acq_badsup"; os.makedirs(acq_badsup); acq_record(f"{acq_badsup}/step1_run_phase4_run.json", P4.ACQ_FREEZE_SHA, "4" * 64)
    L = dict(signoff=good_signoff, prereg_path=V2, calls_path=calls, ledger=ledger_with, supplement=sup_good, manifest=m0, acq_records_dir=acq_ok); T = []
    for sc in P4.SCRIPTS[1:]:
        T.append(run(f"1 {sc}: no sign-off file", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, **{**L, "signoff": f"{tmp}/absent.md"}), "LOCK sign-off"))
        T.append(run(f"2a {sc}: wrong pre-registration sha256 (real v2, sign-off holding another sha256)", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, **{**L, "signoff": bad_signoff}), "LOCK pre-registration"))
        T.append(run(f"2b {sc}: wrong artifact hash (copy of v2 with the GBT hash altered)", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, **{**L, "signoff": bad_art_signoff, "prereg_path": bad_art_prereg}), "LOCK artifact"))
    T.append(run("3 phase4_predict.py: no calls file", lambda: P4.enforce_locks("phase4", "phase4_predict.py", sealed, **{**L, "calls_path": f"{tmp}/absent_calls.csv"}), "LOCK calls file"))
    T.append(run("4 phase4_score.py: calls sha256 not in the ledger (ledger holds the supplement sha256 only)", lambda: P4.enforce_locks("phase4", "phase4_score.py", sealed, **{**L, "ledger": ledger_nocalls, "results_path": f"{tmp}/absent.json"}), "LOCK calls file"))
    T.append(run("5 phase4_score.py: results file already exists", lambda: P4.enforce_locks("phase4", "phase4_score.py", sealed, **{**L, "results_path": results}), "LOCK single run"))
    T.append(run("6 any script: pair not admitted (development pair in phase4 mode)", lambda: P4.enforce_locks("phase4", "phase4_build.py", ["AR26__MGDS_31838"], **L), "LOCK pair list"))
    try:
        r = P4.enforce_locks("phase4", "phase4_build.py", sealed, **L); ok = set(r["manifest"].pair_id) == set(m0.pair_id) and r["supplement"]["n_rows"] == 5 and r["signoff_sha256"] == v2_sha and r["verify"]["prereg"]["sha256"] == v2_sha and r["acq_run_record"]["status"] == "complete" and all(r["verify"][sc]["sha256"] == r["verify"][sc]["expected"] for sc in P4.SCRIPTS)
        net = (P4._allowed == () and len(P4.R9.SEALED_TOKENS) == n_tokens); T.append(dict(test="7 positive control: real v2 + dummy sign-off holding its sha256 + dummy supplement / ledger / acquisition record -> every lock passes; stopped on the safety net (allow-list never applied, every sealed token still in the guard)", passed=bool(ok and net), exception=f"locks passed={ok}; safety net={net}"))
    except Exception as e: T.append(dict(test="7 positive control", passed=False, exception=str(e)[:300]))
    T.append(run("9 phase4_build.py: no supplement file", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": f"{tmp}/absent_sup.parquet"}), "LOCK supplement"))
    T.append(run("10 supplement sha256 not in the ledger", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "ledger": ledger_nosup}), "LOCK supplement: sha256"))
    T.append(run("11a supplement with a wrong pair id", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": sup_badid}), "LOCK supplement: pair ids"))
    T.append(run("11b supplement with a missing pair id", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": sup_missing}), "LOCK supplement: pair ids"))
    import joblib; gbt = joblib.load(P4.GBT_ARTIFACT); models = PP.load_models(); ew = P4.R8.EW_PAIR
    alt_dir = f"{tmp}/oak_r10"; os.makedirs(f"{alt_dir}/rehearsal/predictions", exist_ok=True)
    for sub in ("inputs", "build"): os.symlink(f"{P4.R10_OAK}/rehearsal/{sub}", f"{alt_dir}/rehearsal/{sub}")
    R = pd.read_parquet(f"{P4.R10_OAK}/rehearsal/predictions/{ew}.parquet"); R["P-V1"] = R["P-V1"] * 10 ** 1e-3; R.to_parquet(f"{alt_dir}/rehearsal/predictions/{ew}.parquet")
    real_verify = P4.R8.oak_verify
    def fake_verify(sub, rel, path):
        if rel == f"predictions/{ew}.parquet": return dict(sha256="dummy")
        return real_verify(sub, rel, path)
    P4.R8.oak_verify = fake_verify; P4.R10_OAK = alt_dir
    T.append(run("12 phase4_predict.py: EW0207 reference predictions altered by 1e-3 in log10", lambda: PP.environment_control(gbt, models, 8), "LOCK environment control"))
    P4.R8.oak_verify = real_verify; P4.R10_OAK = f"{P4.R8.OAK_ARCHIVE}/R10"
    mk = m0.copy(); i = mk.index[mk.pair_id == "KM0923__MGDS_31429"][0]; mk.at[i, "max_recoverable_k"] = np.nan; root13 = f"{tmp}/root13"
    try:
        st = PB.build_pair(mk, "KM0923__MGDS_31429", root13, 2, log=lambda *a, **k: None); built = sorted(os.listdir(f"{root13}/build/KM0923__MGDS_31429")); fields = os.path.exists(f"{root13}/fields/KM0923__MGDS_31429"); inputs = os.path.exists(f"{root13}/inputs/KM0923__MGDS_31429")
        key_skip = json.load(open(f"{root13}/build/KM0923__MGDS_31429/pair_stats.json")).get("status") == P4.NO_K
        T.append(dict(test="13 build: KM0923 row with k emptied (rehearsal, scratch root)", passed=(st["status"] == P4.NO_K and built == ["pair_stats.json"] and not fields and not inputs), exception=f"status={st['status']}; files built={built}; fields={fields}; inputs={inputs}"))
        T.append(dict(test="13 figures: no page for the no-k pair (pair_stats status read by phase4_figures)", passed=key_skip, exception=f"skip={key_skip}"))
        rows = [dict(pair="KM0923__MGDS_31429", unit="lu07", role="development", bmin=8, view="not_scorable", status=P4.NO_K, note="no recoverable k (B4 rule 0)", n_tiles_all=0, n_scored=0)]
        c1 = PS.c1_statement(rows); T.append(dict(test="13 score: no-k pair reported with status 'not scorable (no k)', outside C1", passed=(c1["reading"].startswith("not testable") and rows[0]["status"] == P4.NO_K), exception=f"C1 reading on a pool holding only the no-k pair: {c1['reading']}"))
    except Exception as e: T.append(dict(test="13 no-k behaviour", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    n2, probes2, _, _ = P4.R9.guard_selftest_r09(m0); T.append(dict(test="14 guard after the R10b tests: every sealed probe still blocked", passed=(n2 == probes2 == 702), exception=f"{n2}/{probes2}"))
    # ---- R11: 27 the acquisition run-record lock
    T.append(run("27a phase4_build.py: no acquisition run record (empty records directory)", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "acq_records_dir": acq_empty}), "LOCK acquisition run record: no step1_run_phase4"))
    T.append(run("27b phase4_build.py: acquisition run record citing a different freeze record sha256", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "acq_records_dir": acq_badfz}), "LOCK acquisition run record: no record"))
    T.append(run("27c phase4_build.py: acquisition run record citing a different supplement sha256", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "acq_records_dir": acq_badsup}), "LOCK acquisition run record: no record"))
    T.append(run("27d phase4_build.py: the acquisition repo's ACQ-R06b directory as it stands (only the stopped refusal-test record)", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "acq_records_dir": P4.ACQ_RECORDS_DIR}), "LOCK acquisition run record: no record"))
    try:
        r27 = P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": P4.SUPPLEMENT, "ledger": P4.LEDGER, "acq_records_dir": P4.ACQ_RECORDS_DIR}); ok27 = (r27["acq_run_record"]["sha256"] == "6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e" and P4._allowed == ())
        T.append(dict(test="27f phase4_build.py: the REAL supplement, ledger and ACQ-R06b directory (ACQ-R07c citation record) -> every lock passes, allow-list not applied", passed=bool(ok27), exception=f"accepted {os.path.basename(r27['acq_run_record']['path'])} {r27['acq_run_record']['sha256'][:12]}"))
    except Exception as e: T.append(dict(test="27f real acquisition directory", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    T.append(run("27e phase4_build.py: freeze record on disk with another sha256", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "acq_freeze_record": bad_art_prereg}), "LOCK acquisition run record: freeze record"))
    # ---- 28 a supplement row with ship_products_usable = false -> product features masked (harness build, rehearsal mode, scratch root)
    try:
        hs = pd.read_parquet("/scratch/users/scdobbs/grl_review/R11/rehearsal/harness/harness_supplement.parquet"); h = hs[hs.pair_id == "TN399__MGDS_33068"].copy(); h["ship_products_usable"] = False; h["ship_products_usable_reason"] = "refusal test 28: flag set false (harness)"
        mh = pd.concat([m0[m0.pair_id != "TN399__MGDS_33068"], h], ignore_index=True); root28 = f"{tmp}/root28"; st = PB.build_pair(mh, "TN399__MGDS_33068", root28, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), log=lambda *a, **k: None)
        f = pd.read_parquet(f"{root28}/build/TN399__MGDS_33068/features.parquet"); allnan = bool(np.isnan(f[B.CONTRACT].values.astype(float)).all()); frozen = P4.R8.load_features(); fr = frozen[frozen.pair == "TN399__MGDS_33068"]; frozen_used = bool(np.isfinite(fr[B.CONTRACT].values.astype(float)).any())
        T.append(dict(test="28 supplement row with ship_products_usable = false -> product features masked (TN399__MGDS_33068 harness row, rehearsal build)", passed=bool(allnan and st["product_features"] == "masked" and st["ship_products_usable"] is False and frozen_used), exception=f"contract columns all NaN={allnan} ({len(f)} rows); pair_stats product_features={st['product_features']}, ship_products_usable={st['ship_products_usable']}; frozen table (usable=true) holds finite product features={frozen_used}"))
    except Exception as e: T.append(dict(test="28 product features masked", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    # ---- R12c B2: the allow-list rule of revision v2.1 (string-only; nothing opened), then the guard state restored
    try:
        _r04, _c3, _ok = P4.C4._check_r04, P4.C3._check, True; sup5 = pd.read_parquet(P4.SUPPLEMENT); hp = {r.pair_id: os.path.join(P4.C4.DATA, r.harmonized_path_hr) for r in sup5.itertuples()}
        pre = {p: P4.R9._is_sealed(q) for p, q in hp.items()}; P4.apply_allowlist(P4.SEALED5); post = {p: P4.R9._is_sealed(q) for p, q in hp.items()}; rem = list(P4.R9.SEALED_TOKENS)
        fake1 = P4.R9._is_sealed(os.path.join(P4.C4.DATA, "harmonized_confirmatory/nu99/FAKE__MGDS_00000/hr.tif")); fake2 = P4.R9._is_sealed(os.path.join(P4.C4.DATA, "harmonized_confirmatory/hr.tif"))
        T.append(dict(test="30a B2: the five sealed HR paths sealed BEFORE the allow-list", passed=all(pre.values()), exception=str(pre)[:300]))
        T.append(dict(test="30b B2: the five sealed HR paths admitted AFTER apply_allowlist(SEALED5)", passed=not any(post.values()), exception=str(post)[:300]))
        T.append(dict(test="30c B2: harmonized_confirmatory/nu99/FAKE__MGDS_00000/hr.tif refused after the allow-list", passed=bool(fake1), exception=f"is_sealed={fake1}; tokens remaining {rem}"))
        T.append(dict(test="30d B2: harmonized_confirmatory/hr.tif (no pair id) refused after the allow-list", passed=bool(fake2), exception=f"is_sealed={fake2}"))
        P4.R9._is_sealed = P4._R09_IS_SEALED; P4.R9.SEALED_TOKENS[:] = P4.R9.sealed_tokens(m0); P4._allowed = (); P4.C4._check_r04 = _r04; P4.C3._check = _c3
        P4.apply_allowlist(["RC2511__MGDS_32239"]); fam = {"qc_plots AT42-03": os.path.join(P4.C4.DATA, "qc_plots/reaudit_real23/advancing__AT42-03__MGDS_32007.png"), "raw_lr TN159": os.path.join(P4.C4.DATA, "raw_lr/TN159"), "staging_phase1 MGDS_32007": os.path.join(P4.C4.DATA, "staging_phase1/MGDS_32007"), "SUM1004 hr (not admitted)": hp["SUM1004__MGDS_33090"]}
        f30 = {k: P4.R9._is_sealed(v) for k, v in fam.items()}; T.append(dict(test="30e B2: the other token families (real tokens of non-admitted pairs) still refused when only RC2511 is admitted; RC2511 admitted", passed=(all(f30.values()) and not P4.R9._is_sealed(hp["RC2511__MGDS_32239"])), exception=str(f30)[:300]))
        P4.R9._is_sealed = P4._R09_IS_SEALED; P4.R9.SEALED_TOKENS[:] = P4.R9.sealed_tokens(m0); P4._allowed = (); P4.C4._check_r04 = _r04; P4.C3._check = _c3
    except Exception as e: T.append(dict(test="30 B2 cases", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    n3, probes3, _, _ = P4.R9.guard_selftest_r09(m0); T.append(dict(test="29 guard after every test: every sealed probe still blocked", passed=(n3 == probes3 == 702 and P4._allowed == ()), exception=f"{n3}/{probes3}; allow-list {P4._allowed}"))
    shutil.rmtree(tmp); assert not os.path.exists(tmp)
    rec = dict(n_tests=len(T), n_passed=sum(t["passed"] for t in T), tests=T, prereg_v2=dict(path=V2, sha256=v2_sha), dummy_signoff_deleted=not os.path.exists(good_signoff), time=time.strftime("%Y-%m-%dT%H:%M:%S"), node=os.uname().nodename); json.dump(rec, open(f"{OUT}/refusal_tests.json", "w"), indent=1)
    for t in T: print(("PASS" if t["passed"] else "FAIL"), t["test"], "|", (t["exception"] or "")[:110])
    print(f"{rec['n_passed']}/{rec['n_tests']} passed")
