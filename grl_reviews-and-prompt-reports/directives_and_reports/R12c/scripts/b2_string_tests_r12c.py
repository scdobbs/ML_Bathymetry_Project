"""R12c §B2 — string-only tests of the guard before and after the allow-list (no file opened): install the guard, self-test it (702 / 702),
record `_is_sealed` on the five sealed pairs' HR / LR paths (from the supplement rows, data only) BEFORE the allow-list; then
`apply_allowlist(SEALED5)` as `start` does after the locks, and record `_is_sealed` on: the same ten paths (must be False), a fabricated
`harmonized_confirmatory/nu99/FAKE__MGDS_00000/hr.tif` (True), `harmonized_confirmatory/hr.tif` (True), fabricated paths of the other
token families naming no admitted pair (qc_plots/, raw_lr/, staging_phase1/) and the REAL sealed tokens of those families when only
RC2511 is admitted. Also the R04 check after the allow-list. Writes R12c/tables/b2_string_tests.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, f"{DR}/phase4/scripts")
import phase4_common as P4, pandas as pd
R9, C4 = P4.R9, P4.C4; DATA = C4.DATA
def r04_refuses(p):
    try: C4._check_r04(p); return False
    except PermissionError: return True
if __name__ == "__main__":
    m0 = C4.load_manifest(); R9.install_guard_r09(m0); n, probes, _, _ = R9.guard_selftest_r09(m0); out = dict(guard=[n, probes], n_tokens_before=len(R9.SEALED_TOKENS), phase4_common_sha256=P4.sha256_file(f"{DR}/phase4/scripts/phase4_common.py"))
    sup = pd.read_parquet(P4.SUPPLEMENT); paths = {}
    for r in sup.itertuples(): paths[r.pair_id] = dict(hr=os.path.join(DATA, r.harmonized_path_hr), lr=os.path.join(DATA, r.harmonized_path_lr))
    out["before_allowlist"] = {pid: {k: R9._is_sealed(p) for k, p in d.items()} for pid, d in paths.items()}
    fam_fake = {"harmonized_confirmatory fake id": f"{DATA}/harmonized_confirmatory/nu99/FAKE__MGDS_00000/hr.tif", "harmonized_confirmatory no id": f"{DATA}/harmonized_confirmatory/hr.tif",
                "qc_plots fake": f"{DATA}/qc_plots/reaudit_real23/advancing__FAKE__MGDS_00000.png", "raw_lr fake": f"{DATA}/raw_lr/FAKE", "staging_phase1 fake": f"{DATA}/staging_phase1/MGDS_00000"}
    out["before_allowlist_fabricated"] = {k: R9._is_sealed(p) for k, p in fam_fake.items()}
    al = P4.apply_allowlist(P4.SEALED5); out["allowlist"] = list(al); out["tokens_remaining"] = list(R9.SEALED_TOKENS)
    out["after_allowlist"] = {pid: {k: dict(is_sealed=R9._is_sealed(p), r04_refuses=r04_refuses(p)) for k, p in d.items()} for pid, d in paths.items()}
    out["after_allowlist_fabricated"] = {k: R9._is_sealed(p) for k, p in fam_fake.items()}
    # the other token families: the real sealed tokens of AT42-03 / TN159 when only RC2511 is admitted (fresh guard state)
    R9._is_sealed = P4._R09_IS_SEALED; R9.SEALED_TOKENS[:] = R9.sealed_tokens(m0); P4._allowed = (); P4.apply_allowlist(["RC2511__MGDS_32239"])
    real_fam = {"qc_plots real (AT42-03)": f"{DATA}/qc_plots/reaudit_real23/advancing__AT42-03__MGDS_32007.png", "raw_lr real (TN159)": f"{DATA}/raw_lr/TN159", "staging_phase1 real (MGDS_32007)": f"{DATA}/staging_phase1/MGDS_32007", "harmonized real (AT42-03 hr)": paths["AT42-03__MGDS_32007"]["hr"], "harmonized_confirmatory real (SUM1004 hr, not admitted)": paths["SUM1004__MGDS_33090"]["hr"], "harmonized_confirmatory real (RC2511 hr, admitted)": paths["RC2511__MGDS_32239"]["hr"]}
    out["only_RC2511_admitted"] = {k: R9._is_sealed(p) for k, p in real_fam.items()}
    R9._is_sealed = P4._R09_IS_SEALED; R9.SEALED_TOKENS[:] = R9.sealed_tokens(m0); P4._allowed = (); n2, p2, _, _ = R9.guard_selftest_r09(m0); out["guard_after_reset"] = [n2, p2]
    exp = dict(before=all(all(v.values()) for v in out["before_allowlist"].values()), after_sealed_false=all(not v["is_sealed"] and not v["r04_refuses"] for d in out["after_allowlist"].values() for v in d.values()),
               fake_conf_true=out["after_allowlist_fabricated"]["harmonized_confirmatory fake id"] and out["after_allowlist_fabricated"]["harmonized_confirmatory no id"],
               real_families_true=all(out["only_RC2511_admitted"][k] for k in ("qc_plots real (AT42-03)", "raw_lr real (TN159)", "staging_phase1 real (MGDS_32007)", "harmonized real (AT42-03 hr)", "harmonized_confirmatory real (SUM1004 hr, not admitted)")) and not out["only_RC2511_admitted"]["harmonized_confirmatory real (RC2511 hr, admitted)"])
    out["expectations"] = exp; out["passed"] = bool(all(exp.values()) and n == probes == 702 == n2 == p2)
    os.makedirs(f"{DR}/R12c/tables", exist_ok=True); json.dump(out, open(f"{DR}/R12c/tables/b2_string_tests.json", "w"), indent=1); print(json.dumps(out, indent=1))
