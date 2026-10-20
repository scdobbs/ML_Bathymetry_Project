"""R10b §B1 — the 18 R10 refusal tests re-run against the edited scripts, plus tests 9-13: supplement locks (9-11), the EW0207
environment-control lock (12) and the no-k behaviour (13: a scratch manifest copy of the KM0923 row with k emptied, rehearsal
mode -> status 'not scorable (no k)', nothing built, no figure page, reported by the score script). The R09 guard stays installed."""
import json, os, shutil, sys, tempfile, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, phase4_build as PB, phase4_score as PS, phase4_predict as PP
OUT = "/scratch/users/scdobbs/grl_review/R10b/refusal_tests"; os.makedirs(OUT, exist_ok=True)
def run(name, fn, expect):
    try: fn(); return dict(test=name, passed=False, exception=None, note="NO REFUSAL")
    except P4.Refusal as e: return dict(test=name, passed=(expect in str(e)), exception=f"Refusal: {e}"[:300])
    except Exception as e: return dict(test=name, passed=False, exception=f"{type(e).__name__}: {e}"[:300])
if __name__ == "__main__":
    m0 = P4.C4.load_manifest(); P4.R9.install_guard_r09(m0); n, probes, _, _ = P4.R9.guard_selftest_r09(m0); assert n == probes == 702
    tmp = tempfile.mkdtemp(prefix="r10b_refusal_", dir=OUT); sealed = [P4.CONFIRMATORY[0]]
    stated = {k: P4.sha256_file(p) for k, p in dict(GBT=P4.GBT_ARTIFACT, TVI=P4.TVI_PATH, FR=P4.FIELDS_R08, WR=P4.WIDE_R08).items()}; scripts = {sc: P4.sha256_file(os.path.join(P4.HERE, sc)) for sc in P4.SCRIPTS}
    def prereg_text(gbt=None):
        g = gbt or stated["GBT"]; t = f"dummy v2. `gbt_ship_all16_A_fine_tile.joblib`, sha256 `{g}`; `targets_v1_int.py` (sha256 `{stated['TVI']}`); `fields_r08.py` (sha256 `{stated['FR']}`); `wide_window_r08.py`, sha256 `{stated['WR']}`; R07 model code hash `{P4.R07_CODE_SHA}`; "
        return t + "; ".join(f"seed {s} `{P4.PV1_SHA[s]}`" for s in P4.R7.SEEDS) + "; " + "; ".join(f"`{sc}` `{h}`" for sc, h in scripts.items())
    good_prereg = f"{tmp}/PREREG_phase4_v2.md"; open(good_prereg, "w").write(prereg_text()); good_sha = P4.sha256_file(good_prereg)
    good_signoff = f"{tmp}/SIGNOFF.md"; open(good_signoff, "w").write(f"sign-off: {good_sha}\n"); bad_signoff = f"{tmp}/SIGNOFF_bad.md"; open(bad_signoff, "w").write("1" * 64 + "\n")
    bad_art_prereg = f"{tmp}/PREREG_badart.md"; open(bad_art_prereg, "w").write(prereg_text(gbt="2" * 64)); bad_art_signoff = f"{tmp}/SIGNOFF_badart.md"; open(bad_art_signoff, "w").write(P4.sha256_file(bad_art_prereg) + "\n")
    calls = f"{tmp}/calls.csv"; open(calls, "w").write("code,relief,criteria,dominant,ab_whole_survey,note\nQ01,yes,none,relief,na,\n"); calls_sha = P4.sha256_file(calls)
    # dummy supplement: the five sealed rows of manifest v2.1 (as they are: no spacing, no k) with the v2.1 columns; good / bad variants
    sup_good = f"{tmp}/pairs_phase4_sealed.parquet"; m0[m0.pair_id.isin(P4.SEALED5)].reset_index(drop=True).to_parquet(sup_good); sup_sha = P4.sha256_file(sup_good)
    sup_badid = f"{tmp}/pairs_phase4_sealed_badid.parquet"; bad = m0[m0.pair_id.isin(P4.SEALED5)].reset_index(drop=True).copy(); bad.loc[0, "pair_id"] = "AR26__MGDS_31838"; bad.to_parquet(sup_badid); sup_badid_sha = P4.sha256_file(sup_badid)
    sup_missing = f"{tmp}/pairs_phase4_sealed_missing.parquet"; m0[m0.pair_id.isin(P4.SEALED5[:4])].reset_index(drop=True).to_parquet(sup_missing); sup_missing_sha = P4.sha256_file(sup_missing)
    ledger_with = f"{tmp}/ledger_with.md"; open(ledger_with, "w").write(f"| calls {calls_sha} | supplement {sup_sha} | {sup_badid_sha} | {sup_missing_sha} |\n"); ledger_nosup = f"{tmp}/ledger_nosup.md"; open(ledger_nosup, "w").write(f"| calls {calls_sha} |\n"); ledger_nocalls = f"{tmp}/ledger_nocalls.md"; open(ledger_nocalls, "w").write(f"| supplement {sup_sha} |\n")
    results = f"{tmp}/results_D1.json"; open(results, "w").write("{}")
    L = dict(signoff=good_signoff, prereg_path=good_prereg, calls_path=calls, ledger=ledger_with, supplement=sup_good, manifest=m0); T = []
    for sc in P4.SCRIPTS[1:]:
        T.append(run(f"1 {sc}: no sign-off file", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, **{**L, "signoff": f"{tmp}/absent.md"}), "LOCK sign-off"))
        T.append(run(f"2a {sc}: wrong pre-registration sha256", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, **{**L, "signoff": bad_signoff}), "LOCK pre-registration"))
        T.append(run(f"2b {sc}: wrong artifact hash", lambda sc=sc: P4.enforce_locks("phase4", sc, sealed, **{**L, "signoff": bad_art_signoff, "prereg_path": bad_art_prereg}), "LOCK artifact"))
    T.append(run("3 phase4_predict.py: no calls file", lambda: P4.enforce_locks("phase4", "phase4_predict.py", sealed, **{**L, "calls_path": f"{tmp}/absent_calls.csv"}), "LOCK calls file"))
    T.append(run("4 phase4_score.py: calls sha256 not in the ledger (ledger holds the supplement sha256 only)", lambda: P4.enforce_locks("phase4", "phase4_score.py", sealed, **{**L, "ledger": ledger_nocalls, "results_path": f"{tmp}/absent.json"}), "LOCK calls file"))
    T.append(run("5 phase4_score.py: results file already exists", lambda: P4.enforce_locks("phase4", "phase4_score.py", sealed, **{**L, "results_path": results}), "LOCK single run"))
    T.append(run("6 any script: pair not admitted (development pair in phase4 mode)", lambda: P4.enforce_locks("phase4", "phase4_build.py", ["AR26__MGDS_31838"], **L), "LOCK pair list"))
    try: r = P4.enforce_locks("phase4", "phase4_build.py", sealed, **L); ok = set(r["manifest"].pair_id) == set(m0.pair_id) and r["supplement"]["n_rows"] == 5; T.append(dict(test="7 positive control: all locks satisfied with dummy files (build) -> passes; merged manifest = v2.1 rows with the 5 sealed rows from the supplement", passed=bool(ok), exception=None))
    except Exception as e: T.append(dict(test="7 positive control", passed=False, exception=str(e)[:300]))
    T.append(run("9 phase4_build.py: no supplement file", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": f"{tmp}/absent_sup.parquet"}), "LOCK supplement"))
    T.append(run("10 supplement sha256 not in the ledger", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "ledger": ledger_nosup}), "LOCK supplement: sha256"))
    T.append(run("11a supplement with a wrong pair id", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": sup_badid}), "LOCK supplement: pair ids"))
    T.append(run("11b supplement with a missing pair id", lambda: P4.enforce_locks("phase4", "phase4_build.py", sealed, **{**L, "supplement": sup_missing}), "LOCK supplement: pair ids"))
    # 12: environment control with an altered reference (dummy copy of the R10 OAK predictions shifted by 1e-3 in log10)
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
    # 13: a KM0923 row with k emptied, rehearsal mode, scratch root -> no-k status, nothing built, no page, reported by the score script
    mk = m0.copy(); i = mk.index[mk.pair_id == "KM0923__MGDS_31429"][0]; mk.at[i, "max_recoverable_k"] = np.nan; root13 = f"{tmp}/root13"
    try:
        st = PB.build_pair(mk, "KM0923__MGDS_31429", root13, 2, log=lambda *a, **k: None); built = sorted(os.listdir(f"{root13}/build/KM0923__MGDS_31429")); fields = os.path.exists(f"{root13}/fields/KM0923__MGDS_31429"); inputs = os.path.exists(f"{root13}/inputs/KM0923__MGDS_31429")
        key_skip = json.load(open(f"{root13}/build/KM0923__MGDS_31429/pair_stats.json")).get("status") == P4.NO_K
        T13 = dict(test="13 build: KM0923 row with k emptied (rehearsal, scratch root)", passed=(st["status"] == P4.NO_K and built == ["pair_stats.json"] and not fields and not inputs), exception=f"status={st['status']}; files built={built}; fields={fields}; inputs={inputs}")
        T.append(T13); T.append(dict(test="13 figures: no page for the no-k pair (pair_stats status read by phase4_figures)", passed=key_skip, exception=f"skip={key_skip}"))
        rows = [dict(pair="KM0923__MGDS_31429", unit="lu07", role="development", bmin=8, view="not_scorable", status=P4.NO_K, note="no recoverable k (B4 rule 0)", n_tiles_all=0, n_scored=0)]
        c1 = PS.c1_statement(rows); T.append(dict(test="13 score: no-k pair reported with status 'not scorable (no k)', outside C1", passed=(c1["reading"].startswith("not testable") and rows[0]["status"] == P4.NO_K), exception=f"C1 reading on a pool holding only the no-k pair: {c1['reading']}"))
    except Exception as e: T.append(dict(test="13 no-k behaviour", passed=False, exception=f"{type(e).__name__}: {e}"[:300]))
    n2, probes2, _, _ = P4.R9.guard_selftest_r09(m0); T.append(dict(test="14 guard after the tests: every sealed probe still blocked", passed=(n2 == probes2 == 702), exception=f"{n2}/{probes2}"))
    shutil.rmtree(tmp); rec = dict(n_tests=len(T), n_passed=sum(t["passed"] for t in T), tests=T, time=time.strftime("%Y-%m-%dT%H:%M:%S"), node=os.uname().nodename); json.dump(rec, open(f"{OUT}/refusal_tests.json", "w"), indent=1)
    for t in T: print(("PASS" if t["passed"] else "FAIL"), t["test"], "|", (t["exception"] or "")[:110])
    print(f"{rec['n_passed']}/{rec['n_tests']} passed")
