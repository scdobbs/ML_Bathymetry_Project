"""R12c Part E1.3 (the R12 C1 preconditions against v2.1) (read-only; nothing sealed is opened):
  1. the sign-off is committed, clean, and holds the v2 sha256; v2 is at its tag (phase4-prereg-v2 -> the committed PREREG_phase4_v2.md hashes to it);
  2. the deviation note is committed and clean (its sha256 and last commit);
  3. the five Phase 4 scripts match the B6 table of v2 (parse_prereg_hashes vs the files on disk) and are clean in git;
  4. the supplement and the completion run record match the §B hashes; both sha256 are in CHANGE_LEDGER.md;
  5. the R09 guard self-test: 702 of 702 probes blocked.
Also (challenge evidence, no sealed file opened): the §A2.2 acquisition run-record lock, called with the supplement's sha256, and the field
the completion record carries the freeze-record citation under."""
import json, os, subprocess, sys
import r12c_common as R12
P4 = R12.P4; C4 = P4.C4; R9 = P4.R9; REPO = P4.REPO
SUP_SHA = "6968e04554ddeef3c335587d602cb5fab2185db32b46cbd152475a919eb59206"; REC_SHA = "fb1e07b24681e8b7b15dec3b232ea78266c6c7cd82ae64518e82b76ae99dcc07"
DEV_SHA = "962fd19cd9b3b13dc14ba1e1d37a7860515fb8b7168b33732cbaafe77eb95894"; DEV_COMMIT = "84fd9a189f1505dd30f6a92a6045b8ff9767a2d8"; V2_SHA = "d6af268bc8211fe9b4ae17439a47e6fdcbe7e5949d8ab387a7be080b2c748334"
def git(*a): return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
def rel(p): return os.path.relpath(p, REPO)
if __name__ == "__main__":
    out = {}
    # 1 sign-off / v2 at its tag
    so = P4.SIGNOFF; out["1_signoff"] = dict(path=so, sha256=P4.sha256_file(so), holds_v2_sha=(P4.read_signoff(so) == V2_SHA), last_commit=git("log", "-1", "--format=%H", "--", rel(so)), clean=(git("status", "--porcelain", "--", rel(so)) == ""))
    tag_commit = git("rev-parse", "phase4-prereg-v2.1^{}"); v2_at_tag = subprocess.run(["git", "show", f"{tag_commit}:{rel(P4.PREREG_V2)}"], cwd=REPO, capture_output=True, text=True).stdout
    out["1_prereg_v2"] = dict(path=P4.PREREG_V2, sha256_on_disk=P4.sha256_file(P4.PREREG_V2), sha256_at_tag=P4.sha256_text(v2_at_tag), tag_commit=tag_commit, expected=V2_SHA, clean=(git("status", "--porcelain", "--", rel(P4.PREREG_V2)) == ""))
    out["1_passed"] = bool(out["1_signoff"]["holds_v2_sha"] and out["1_signoff"]["clean"] and out["1_signoff"]["last_commit"] and out["1_prereg_v2"]["sha256_on_disk"] == V2_SHA == out["1_prereg_v2"]["sha256_at_tag"] and out["1_prereg_v2"]["clean"])
    # 2 deviation note
    dv = f"{P4.DR}/R11/DEVIATION_phase4_01.md"; out["2_deviation"] = dict(path=dv, sha256=P4.sha256_file(dv), expected=DEV_SHA, last_commit=git("log", "-1", "--format=%H", "--", rel(dv)), expected_commit=DEV_COMMIT, clean=(git("status", "--porcelain", "--", rel(dv)) == ""))
    for n, pth, exp in (("02", f"{P4.DR}/R11/DEVIATION_phase4_02.md", "03dd3aa286e0075417a19af622765740fac34e49b47f7dab1558533f0c94f0de"), ("03", f"{P4.DR}/R12c/DEVIATION_phase4_03.md", "dfccdadcaef1fcdaa0c0d063ff1f1b59df373fef9357765a70e24b869b0333af")):
        out[f"2_deviation_{n}"] = dict(path=pth, sha256=P4.sha256_file(pth), expected=exp, last_commit=git("log", "-1", "--format=%H", "--", rel(pth)), clean=(git("status", "--porcelain", "--", rel(pth)) == ""), signed=any(l.startswith("Signed") and "____" not in l for l in open(pth).read().splitlines()))
    out["2_passed"] = bool(out["2_deviation"]["sha256"] == DEV_SHA and out["2_deviation"]["last_commit"] == DEV_COMMIT and out["2_deviation"]["clean"] and all(out[f"2_deviation_{n}"]["sha256"] == out[f"2_deviation_{n}"]["expected"] and out[f"2_deviation_{n}"]["clean"] and out[f"2_deviation_{n}"]["last_commit"] and out[f"2_deviation_{n}"]["signed"] for n in ("02", "03")))
    # 3 scripts vs B6
    stated = P4.parse_prereg_hashes(open(P4.PREREG_V2).read()); sc = {}
    for s in P4.SCRIPTS:
        p = os.path.join(P4.HERE, s); sc[s] = dict(on_disk=P4.sha256_file(p), b6=stated.get(s), equal=(P4.sha256_file(p) == stated.get(s)), clean=(git("status", "--porcelain", "--", rel(p)) == ""))
    out["3_scripts"] = sc; out["3_passed"] = bool(all(v["equal"] and v["clean"] for v in sc.values()))
    # 4 supplement + completion record hashes, in the ledger
    rp = f"{P4.ACQ_RECORDS_DIR}/step1_run_phase4_completion.json"; led = open(P4.LEDGER).read()
    out["4_supplement"] = dict(path=P4.SUPPLEMENT, sha256=P4.sha256_file(P4.SUPPLEMENT), expected=SUP_SHA, in_ledger=(SUP_SHA in led)); out["4_completion_record"] = dict(path=rp, sha256=P4.sha256_file(rp), expected=REC_SHA, in_ledger=(REC_SHA in led))
    out["4_passed"] = bool(out["4_supplement"]["sha256"] == SUP_SHA and out["4_completion_record"]["sha256"] == REC_SHA and out["4_supplement"]["in_ledger"] and out["4_completion_record"]["in_ledger"])
    # 5 guard self-test
    m0 = C4.load_manifest(); R9.install_guard_r09(m0); n, probes, readers, rows = R9.guard_selftest_r09(m0); out["5_guard"] = dict(blocked=n, probes=probes, readers=readers, rows=rows); out["5_passed"] = bool(n == probes == 702)
    # challenge evidence: the acquisition run-record lock as it stands
    try: a = P4.load_acq_run_record(SUP_SHA); out["acq_lock"] = dict(accepted=True, path=a["path"], sha256=a["sha256"])
    except P4.Refusal as e: out["acq_lock"] = dict(accepted=False, refusal=str(e)[:600])
    rec = json.load(open(rp)); L = rec.get("locks") or {}
    out["acq_lock"]["completion_record_fields"] = dict(status=rec.get("status"), mode=rec.get("mode"), supplement_sha256=rec.get("supplement_sha256"), top_level_freeze_record_sha256=rec.get("freeze_record_sha256"), lock_keys=list(L.keys()),
                                                        locks_lock2_freeze_record=L.get("lock2_freeze_record"), locks_lock2_frozen_code=L.get("lock2_frozen_code"), lock_read_by_phase4_common="locks.lock2_freeze_record.freeze_record_sha256 (phase4_common.load_acq_run_record)")
    out["all_passed"] = bool(all(out[f"{i}_passed"] for i in range(1, 6))); out["git_head"] = git("rev-parse", "HEAD")
    os.makedirs(f"{R12.BASE}/checks", exist_ok=True); json.dump(out, open(f"{R12.BASE}/checks/c1_preconditions.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
