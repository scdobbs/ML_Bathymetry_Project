"""R10 §A5 — refusal tests of the phase4-mode locks, with the R09 guard installed underneath. Each test calls
phase4_common.enforce_locks() (or the lock functions it uses) with dummy files in a scratch directory for the positive side;
the allow-list step is never reached. Records each test as passed / failed with the exception message."""
import json, os, shutil, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
OUT = "/scratch/users/scdobbs/grl_review/R10/refusal_tests"; os.makedirs(OUT, exist_ok=True)
def run(name, fn, expect_substring):
    try: fn(); return dict(test=name, passed=False, exception=None, note="NO REFUSAL")
    except P4.Refusal as e: return dict(test=name, passed=(expect_substring in str(e)), exception=f"Refusal: {e}"[:300])
    except Exception as e: return dict(test=name, passed=False, exception=f"{type(e).__name__}: {e}"[:300])
if __name__ == "__main__":
    m0 = P4.C4.load_manifest(); P4.R9.install_guard_r09(m0); n, probes, _, _ = P4.R9.guard_selftest_r09(m0); assert n == probes == 702
    tmp = tempfile.mkdtemp(prefix="r10_refusal_", dir=OUT); sealed = [P4.CONFIRMATORY[0]]
    # dummy pre-registration v2 whose stated hashes are the TRUE artifact / code / script hashes, so that only the lock under test fails
    stated = {k: P4.sha256_file(p) for k, p in dict(GBT=P4.GBT_ARTIFACT, TVI=P4.TVI_PATH, FR=P4.FIELDS_R08, WR=P4.WIDE_R08).items()}
    scripts = {sc: P4.sha256_file(os.path.join(P4.HERE, sc)) for sc in P4.SCRIPTS}
    def prereg_text(gbt=None, bad_script=False):
        g = gbt or stated["GBT"]; t = f"dummy v2. `gbt_ship_all16_A_fine_tile.joblib`, sha256 `{g}`; `targets_v1_int.py` (sha256 `{stated['TVI']}`); `fields_r08.py` (sha256 `{stated['FR']}`); `wide_window_r08.py`, sha256 `{stated['WR']}`; R07 model code hash `{P4.R07_CODE_SHA}`; "
        t += "; ".join(f"seed {s} `{P4.PV1_SHA[s]}`" for s in P4.R7.SEEDS) + "; " + "; ".join(f"`{sc}` `{'0'*64 if bad_script else h}`" for sc, h in scripts.items()); return t
    good_prereg = f"{tmp}/PREREG_phase4_v2.md"; open(good_prereg, "w").write(prereg_text()); good_sha = P4.sha256_file(good_prereg)
    good_signoff = f"{tmp}/SIGNOFF_phase4_prereg_v2.md"; open(good_signoff, "w").write(f"sign-off: {good_sha}\n")
    bad_signoff = f"{tmp}/SIGNOFF_bad.md"; open(bad_signoff, "w").write("sign-off: " + "1" * 64 + "\n")
    bad_art_prereg = f"{tmp}/PREREG_badart.md"; open(bad_art_prereg, "w").write(prereg_text(gbt="2" * 64)); bad_art_signoff = f"{tmp}/SIGNOFF_badart.md"; open(bad_art_signoff, "w").write(P4.sha256_file(bad_art_prereg) + "\n")
    calls = f"{tmp}/calls.csv"; open(calls, "w").write("code,relief,criteria,dominant,ab_whole_survey,note\nQ01,yes,none,relief,na,\n"); calls_sha = P4.sha256_file(calls)
    ledger_with = f"{tmp}/ledger_with.md"; open(ledger_with, "w").write(f"| x | calls sha256 {calls_sha} |\n"); ledger_without = f"{tmp}/ledger_without.md"; open(ledger_without, "w").write("| nothing |\n")
    results = f"{tmp}/results_D1.json"; open(results, "w").write("{}")
    T = []
    for sc in P4.SCRIPTS[1:]:
        T.append(run(f"1 {sc}: no sign-off file", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, signoff=f"{tmp}/absent.md", prereg_path=good_prereg, calls_path=calls, ledger=ledger_with), "LOCK sign-off"))
        T.append(run(f"2a {sc}: wrong pre-registration sha256", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, signoff=bad_signoff, prereg_path=good_prereg, calls_path=calls, ledger=ledger_with), "LOCK pre-registration"))
        T.append(run(f"2b {sc}: wrong artifact hash", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, signoff=bad_art_signoff, prereg_path=bad_art_prereg, calls_path=calls, ledger=ledger_with), "LOCK artifact"))
    T.append(run("3 phase4_predict.py: no calls file", lambda: P4.enforce_locks("phase4", "phase4_predict.py", sealed, signoff=good_signoff, prereg_path=good_prereg, calls_path=f"{tmp}/absent_calls.csv", ledger=ledger_with), "LOCK calls file"))
    T.append(run("4 phase4_score.py: calls sha256 not in the ledger", lambda: P4.enforce_locks("phase4", "phase4_score.py", sealed, signoff=good_signoff, prereg_path=good_prereg, calls_path=calls, ledger=ledger_without, results_path=f"{tmp}/absent_results.json"), "not recorded"))
    T.append(run("5 phase4_score.py: results file already exists", lambda: P4.enforce_locks("phase4", "phase4_score.py", sealed, signoff=good_signoff, prereg_path=good_prereg, calls_path=calls, ledger=ledger_with, results_path=results), "LOCK single run"))
    T.append(run("6 any script: pair not admitted (development pair in phase4 mode)", lambda: P4.enforce_locks("phase4", "phase4_build.py", ["AR26__MGDS_31838"], signoff=good_signoff, prereg_path=good_prereg), "LOCK pair list"))
    # positive control: with every dummy lock satisfied, enforce_locks passes for build (no allow-list is applied here; the guard stays fully on)
    try: P4.enforce_locks("phase4", "phase4_build.py", sealed, signoff=good_signoff, prereg_path=good_prereg); T.append(dict(test="7 positive control: all locks satisfied with dummy files (build) -> passes (guard untouched)", passed=True, exception=None))
    except Exception as e: T.append(dict(test="7 positive control", passed=False, exception=str(e)[:300]))
    n2, probes2, _, _ = P4.R9.guard_selftest_r09(m0); T.append(dict(test="8 guard after the tests: every sealed probe still blocked", passed=(n2 == probes2 == 702), exception=f"{n2}/{probes2}"))
    shutil.rmtree(tmp); rec = dict(n_tests=len(T), n_passed=sum(t["passed"] for t in T), tests=T, time=time.strftime("%Y-%m-%dT%H:%M:%S")); json.dump(rec, open(f"{OUT}/refusal_tests.json", "w"), indent=1)
    for t in T: print(("PASS" if t["passed"] else "FAIL"), t["test"], "|", (t["exception"] or "")[:110])
    print(f"{rec['n_passed']}/{rec['n_tests']} passed")
