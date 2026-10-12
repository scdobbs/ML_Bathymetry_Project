"""Splice the generated tables (R12/tables/*.md) and the build-job outcome into REPORT_R12_2026-10-11.md in place of its placeholders.
Idempotent on the table placeholders (they are replaced once); JOB_PRESERVE / R12_COMMIT are replaced by the hash-note step (--preserve, --commit)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); REP = f"{R}/REPORT_R12_2026-10-11.md"
def tab(name): return open(f"{R}/tables/{name}").read().strip()
BUILD_OUTCOME = ("**stopped at the acquisition run-record lock** on `sh03-08n52` (AMD EPYC 7543) 12 s after start, exit code 1: `phase4_common.Refusal: LOCK acquisition run record: no record in …/ACQ-R06b is complete in phase4 mode, cites freeze record 331c94f01d7d… and supplement 6968e04554dd…` "
                 "(`phase4_common.py:136`, reached from `enforce_locks` line 189, from `start` line 241). The locks before it — sign-off, pre-registration sha256, the GBT artifact, the three P-V1 checkpoints, `targets_v1_int.py`, `fields_r08.py`, `wide_window_r08.py`, the five scripts, the R07 code hash, the R05 frozen check and the supplement (sha256 in the ledger, five ids, v2.1 columns) — all passed on the real files")
BUILD_NARRATIVE = ("**Outcome (CONFIRMED from `logs/phase4_build_47445500.out` / `.err`).** The guard was installed and self-tested first (702 / 702, as in every run; the `[phase4_common.start]` line, printed only after the locks and the allow-list, is absent from the log). `enforce_locks` passed the sign-off lock (sha256 `9795db3b…` read from `R11/SIGNOFF_phase4_prereg_v2.md`), `verify_frozen` in phase4 mode (the pre-registration's sha256; the artifact, frozen-code and script hashes parsed from its B2 / B6 text; the R07 code hash; `r06_common.check_frozen`) and `load_supplement` (sha256 `6968e045…` found in `CHANGE_LEDGER.md` row 35; exactly the five ids; the 64 v2.1 columns), then **refused at `load_acq_run_record`**: the completion record has status `complete`, mode `phase4` and the supplement's sha256, but no `locks.lock2_freeze_record` entry; the stopped ACQ-R07 record has the citation but no supplement. "
                   "The refusal was predicted by the login-node pre-check (§2 note 1). Per §C2 the job was not rerun and nothing was changed; the allow-list was never reached, so no sealed path was opened; `grl_review_archive/phase4/` was not created (`start` makes it only after the locks pass). **HOLD H1.**")
if __name__ == "__main__":
    t = open(REP).read()
    for ph, name in (("A1_TABLE", "a1_code_reading.md"), ("A2_TABLE", "a2_gate.md"), ("A3_TABLE", "a3_control.md"), ("C1_TABLE", "c1_preconditions.md"), ("C2_TABLE", "c2_build.md"), ("FILES_TABLE", "files_r12.md")):
        if ph in t: t = t.replace(ph, tab(name))
    t = t.replace("BUILD_OUTCOME", BUILD_OUTCOME).replace("BUILD_NARRATIVE", BUILD_NARRATIVE)
    a = sys.argv[1:]
    if "--preserve" in a: t = t.replace("JOB_PRESERVE", a[a.index("--preserve") + 1])
    if "--commit" in a: t = t.replace("R12_COMMIT", f"`{a[a.index('--commit') + 1]}`")
    open(REP, "w").write(t); print("placeholders left:", re.findall(r"\b(A1_TABLE|A2_TABLE|A3_TABLE|C1_TABLE|C2_TABLE|FILES_TABLE|BUILD_OUTCOME|BUILD_NARRATIVE|JOB_PRESERVE|R12_COMMIT)\b", t))
