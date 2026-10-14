# DIRECTIVE R16c (CNN repo, CPU only): pre-registration v3.1 amendment for the acquisition freeze v2

**Date:** 2026-10-13
**Repo:** CNN repo, `grl_reviews-and-prompt-reports/directives_and_reports/`
**Report to:** `R16c/REPORT_R16c_<date>.md`
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.
**Report length:** at most 8 bullets, plus at most 3 HOLDs.
**Inputs (acquisition repo, read only):** REPORT_ACQ-R11b; `reports_post_grl_review/PHASE5/freeze_record_phase5_v2.json`
(sha256 `f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438`, tag `phase5-acq-freeze-v2`); the ACQ-R11b
rehearsal record `ACQ-R11b/rehearsal/step1_run_rehearsal_r11b.json` and its outputs; `ACQ-R11b/vcheck_calibration_v2.csv`
(sha256 `41973318d9ac55cffc7baa3f57a359da2538f2fd0a7a3a952d39770a14b7d8b2`).

## 0. Purpose and rulings

ACQ-R11b added an automated vertical-datum check to the Phase 5 acquisition chain and re-froze it. Signed
pre-registration v3 and `phase5_common.py` pin the old freeze (`67695831…`), so ACQ-R10 cannot run until the CNN side
points to freeze v2 under an amended, re-signed pre-registration.

**Steve's rulings (2026-10-13):** option (b) with the fallback approved in advance; the fallback applied. Ship
line-to-line inconsistency is unchecked in Phase 5 and v3.1 says so.

**This is an amendment made before any sealed file is opened or any Phase 5 swath is fetched.** No criterion,
threshold, seed, model, site or product rule changes. Signed v3, its sign-off, its tag and every Phase 4 file stay
as they are.

**Stop after §4.** Do not tag.

**Challenge clause.** If any instruction here is wrong or underspecified against the code or the files, stop at
that point, give the evidence, and report it as a HOLD. Do not work around it.

## 1. Script change (`phase5/scripts/phase5_common.py` only)

1. l.47: `ACQ_FREEZE_RECORD = f"{ACQ_RECORDS_DIR}/freeze_record_phase5_v2.json"` and
   `ACQ_FREEZE_SHA = "f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438"`.
2. l.48: `ACQ_STEP1_SHA = "83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47"`.
3. The pre-registration and sign-off path constants: `R16c/PREREG_phase5_v3_1.md` and
   `R16c/SIGNOFF_phase5_prereg_v3_1.md`. Report their file and line.
4. Nothing else. `eligibility()` (l.155 to 164) and the l.134 check stay as they are. The other four Phase 5 scripts
   are not edited. Report the diff, which must be these lines only.

## 2. Dress rehearsal (as R16 Part C, against freeze v2)

1. Rebuild the mock sealed layout from the ACQ-R11b rehearsal outputs (NA080__MGDS_31289, KM0923__MGDS_31429, the
   FK151121 harness), with the ACQ-R11b rehearsal record, freeze v2 and the mock `eligible = False` row. Copies, not
   links.
2. Run all four scripts in order, with a mock calls file, exactly as R16 Part C.
3. **Must hold:** every R16 Part C item 3.1 to 3.9, plus:
   1. the acquisition lock accepts the ACQ-R11b record against freeze v2, and refuses freeze v1 (`67695831…`), a
      wrong freeze sha256 and a wrong supplement sha256;
   2. the predictions and scores on the three mock sites equal the R16 rehearsal values (the ACQ-R11b rasters are
      bit-identical to ACQ-R09's). Report any difference as a HOLD.
4. The refusal suite (R16's, all tests), all passing.
5. **Gate:** any failure stops before §3 and becomes a HOLD. One fix iteration at most without a ruling.

## 3. Pre-registration v3.1

Render `R16c/PREREG_phase5_v3_1.md` from `R16/PREREG_phase5_v3.md` with exactly these changes:

1. **B1 item 1 (v3 l.35)**, replaced by: "co-registration passes the development QA thresholds (ACQ-R03: offset ≤ 30 m,
   overlap ≥ 1 km², ≥ 1 256-px tile with ≥ 50 % joint-valid cells) and the vertical-datum check (ACQ-R11b): on the
   ship grid's joint-valid cells, the 95th percentile of |r − median r|, where r is the area-averaged high-resolution
   field minus the ship field after the rigid shift, divided by |median ship depth| there, is ≤ T_G = 0.200712
   (calibrated on development pairs only). Ship line-to-line inconsistency is not checked."
2. **Blindness sentence (v3 l.47):** add "and the vertical-datum check" to the automated steps, and "only its pass or
   fail is recorded".
3. **Acquisition freeze:** every citation of `67695831…` becomes `f5a30b87…` (full sha256), with the tag
   `phase5-acq-freeze-v2` and the entry script sha256 `83af2875…`. The text must state the full `ACQ_FREEZE_SHA`, as
   l.134 requires.
4. **B6 table:** the new `phase5_common.py` sha256; the other four rows unchanged.
5. **A short "Amendment v3.1" section** (at most 5 lines): why (ACQ-R10 stopped because two development QA steps were
   human judgements); what changed (the datum check, freeze v2, the constants); what did not (criteria, thresholds,
   seeds, sites, models); the line check fallback and why (lattices not on disk for 36 of 38 development pairs); that
   no sealed file was opened and no Phase 5 swath fetched before v3.1; that v3.1 supersedes v3.
6. **Exposure:** add that Steve has seen the development and known-bad G values (ACQ-R11, ACQ-R11b), and no value
   from any Phase 5 site.

Report the diff of v3 against v3.1, section by section. Any other difference is a HOLD.

## 4. Sign-off draft, commit, then STOP

1. Draft `R16c/SIGNOFF_phase5_prereg_v3_1_DRAFT.md` on the v3 sign-off pattern. Its first 64-hex string must be the
   v3.1 sha256 (check with the lock's own `read_signoff`). Signature blank.
2. Commit by explicit path (`phase5_common.py`, v3.1, the draft, R16c scripts and tables, the report); never
   `git add -A`. One `CHANGE_LEDGER.md` row, status "awaiting signature". Do not tag.
3. R16c rehearsal outputs to OAK `grl_review_archive/R16c/` with `SHA256SUMS` verified.
4. **Stop.** After Steve signs: commit the signed sign-off and tag `phase5-prereg-v3.1`.

## 5. Out of scope

- Opening any file under a sealed root except mock copies; running phase5 mode on real sites.
- Editing signed v3, its sign-off, the `phase5-prereg-v3` tag, any Phase 4 file, or the other four Phase 5 scripts.
- Any write to the acquisition repo, including `sbatch/acq_r10/step1.sbatch` (ACQ-R10 updates it).
- Changing a frozen function, model, threshold or seed.
- Manuscript text.
