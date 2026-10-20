"""R12c must-hold 5 — derive refusal_tests_r12c.py from R11/scripts/refusal_tests_r11.py (the 26 R10b tests + 27a-e, 28, 29) with exactly
these edits, each asserted to apply once: (i) the output directory -> R12c; (ii) the harness supplement of test 28 pinned to the R11
rehearsal root (this script does not re-point ROOT); (iii) test 27d keeps its refusal (the dummy supplement differs from the one the real
records cite) and a new test 27f runs the locks on the REAL supplement, ledger and acquisition directory (the ACQ-R07c citation record must
be accepted); (iv) the B2 cases added as tests 30a-30e with the guard state restored afterwards (test 29 then checks it). Prints the diff."""
import difflib, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); SRC = f"{DR}/R11/scripts/refusal_tests_r11.py"; DST = f"{HERE}/refusal_tests_r12c.py"
B2 = '''    # ---- R12c B2: the allow-list rule of revision v2.1 (string-only; nothing opened), then the guard state restored
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
'''
P27F = '''    try:
        r27 = P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": P4.SUPPLEMENT, "ledger": P4.LEDGER, "acq_records_dir": P4.ACQ_RECORDS_DIR}); ok27 = (r27["acq_run_record"]["sha256"] == "6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e" and P4._allowed == ())
        T.append(dict(test="27f phase4_build.py: the REAL supplement, ledger and ACQ-R06b directory (ACQ-R07c citation record) -> every lock passes, allow-list not applied", passed=bool(ok27), exception=f"accepted {os.path.basename(r27['acq_run_record']['path'])} {r27['acq_run_record']['sha256'][:12]}"))
    except Exception as e: T.append(dict(test="27f real acquisition directory", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
'''
def sub1(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old)); return text.replace(old, new)
if __name__ == "__main__":
    t = open(SRC).read(); o = t
    t = sub1(t, 'OUT = "/scratch/users/scdobbs/grl_review/R11/refusal_tests"', 'OUT = "/scratch/users/scdobbs/grl_review/R12c/refusal_tests"')
    t = sub1(t, 'hs = pd.read_parquet(f"{P4.ROOT[\'rehearsal\']}/harness/harness_supplement.parquet")', 'hs = pd.read_parquet("/scratch/users/scdobbs/grl_review/R11/rehearsal/harness/harness_supplement.parquet")')
    t = sub1(t, '    T.append(run("27e phase4_build.py: freeze record on disk with another sha256"', P27F + '    T.append(run("27e phase4_build.py: freeze record on disk with another sha256"')
    t = sub1(t, "    n3, probes3, _, _ = P4.R9.guard_selftest_r09(m0); T.append(dict(test=\"29 guard after every test", B2 + "    n3, probes3, _, _ = P4.R9.guard_selftest_r09(m0); T.append(dict(test=\"29 guard after every test")
    t = sub1(t, 'prefix="r11_refusal_"', 'prefix="r12c_refusal_"')
    t = '"""R12c must-hold 5 — DERIVED from R11/scripts/refusal_tests_r11.py by derive_refusal_tests_r12c.py (diff in R12c/tables/refusal_tests_derivation.diff): R12c output dir; test 28 harness pinned to the R11 root; 27f (real acquisition directory accepted); 30a-e (B2). The R11 docstring follows.\n' + t[3:]
    open(DST, "w").write(t); d = "\n".join(difflib.unified_diff(o.splitlines(), t.splitlines(), "R11/scripts/refusal_tests_r11.py", "R12c/scripts/refusal_tests_r12c.py", lineterm="", n=0))
    os.makedirs(f"{DR}/R12c/tables", exist_ok=True); open(f"{DR}/R12c/tables/refusal_tests_derivation.diff", "w").write(d + "\n"); print(d[:3000]); print("... derived", DST)
