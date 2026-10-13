# DIRECTIVE R16 (CNN repo, CPU only): Phase 5 scripts, full dress rehearsal, deviation note 04, pre-registration v3 for signature

**Date:** 2026-10-13
**Repo:** CNN repo, `grl_reviews-and-prompt-reports/directives_and_reports/`
**Report to:** `R16/REPORT_R16_<date>.md`
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.
**Report length:** at most 8 bullets, plus at most 3 HOLDs.
**Inputs:** `phase5/PHASE5_design_spec_2026-10-13.md`; from the acquisition repo (read only): REPORT_ACQ-R09,
`reports_post_grl_review/ACQ-R09/RECORD_CONTRACT_phase5.md`, `reports_post_grl_review/PHASE5/freeze_record_phase5.json`
(sha256 `6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1`), the rehearsal supplement and run record
`step1_run_rehearsal_r09.json` and its rehearsal outputs.

## 0. Purpose

Prepare everything the CNN side of Phase 5 needs, prove it end to end on a mock sealed layout, and render
pre-registration v3 for Steve's signature. **Stop after Part E.** Nothing sealed is opened; nothing is predicted or
scored on a sealed site. Phase 4 files, records, tags and outputs are not edited.

**Rulings carried into v3 (Steve, 2026-10-13):** the design spec; ACQ-R09 S1 to S3; ACQ-R09 H-1: keep the 3.88
lower bound as written, with a note that the frozen chain now gives FK151121 3.8746 cells.

**Challenge clause.** If any instruction here is wrong or underspecified against the code or the files, stop at
that point, give the evidence, and report it as a HOLD. Do not work around it.

## Part A: code reading (read only; file and line)

List every place in the five Phase 4 scripts that is specific to Phase 4: pair list (SEALED5), supplement path and
column check, acquisition record directory and freeze record, pre-registration and sign-off constants, guard tokens
and allow-list, output root, calls lock, the figures template (keyed by Q-code) and the score script's calls input
(keyed by pair), the C1 label logic, the pooled row, and whether CIs are computed for GBT_ship and P-V1. This list
is what Part B parameterises, and nothing else.

## Part B: Phase 5 scripts

Write `phase5/scripts/phase5_common.py`, `phase5_build.py`, `phase5_figures.py`, `phase5_predict.py`,
`phase5_score.py` as copies of the Phase 4 scripts with only these changes:

1. **Sites:** the admitted pair ids are the eligible rows of `manifest/pairs_phase5_sealed.parquet` in the
   acquisition repo; its sha256 must be in the ledger. The column check accepts the 64 Phase 4 columns plus the six
   eligibility columns, and asserts the 64 are unchanged. A row with `eligible = False` is reported "not testable
   (<eligibility_reason>)", writes only its status, and gets no page, prediction or score.
2. **Acquisition record lock:** against `reports_post_grl_review/PHASE5/` and `freeze_record_phase5.json`, using the
   keys in `RECORD_CONTRACT_phase5.md`. Messages say "phase5".
3. **Guard and allow-list:** seal the Phase 5 roots (`sealed_candidates_v3` and the harmonized output root named in
   the contract) with every site id; after every lock, admit a path only if it contains an admitted full pair id;
   any other path under a sealed root stays refused. The R09 guard and its 702-probe self-test stay as they are;
   add Phase 5 probes.
4. **Output root:** `grl_review_archive/phase5/`. **Constants:** `R16/PREREG_phase5_v3.md` and
   `R16/SIGNOFF_phase5_prereg_v3.md`.
5. **Calls (Phase 4 deviation note 04):** Steve's file stays keyed by Q-code. `phase5_score.py` takes the committed
   code-keyed file (its sha256 must be in the ledger) and joins it to pairs through the figures key inside the
   script, asserting every value unchanged. No separate joined file.
6. **C1 primary (spec §3.4):** the pooled row over the C1-set sites; "confirmed" if its CI lower bound is above zero,
   "not confirmed" otherwise, "not testable" if fewer than 3 sites are in the C1 set. The v2.1 per-site label and
   counts are written beside it as the secondary reading.
7. **C2 (spec §3.5):** GBT_ship and P-V1, per site and pooled, each with a CI from the same bootstrap resamples, plus
   the differences from s_lr with CIs.
8. **Unchanged:** every frozen function for targets, tiles, interior masks, features, model inputs, the three models,
   the EW0207 control, the artifact rule, the single-run lock and the bootstrap seeds. Import them; do not copy them.

## Part C: dress rehearsal on a mock sealed layout

1. **Mock sites:** the ACQ-R09 rehearsal outputs for NA080__MGDS_31289, KM0923__MGDS_31429 and the FK151121
   harness, copied (not linked) into a scratch layout whose paths match the real Phase 5 roots name for name, with
   a mock supplement and the ACQ-R09 rehearsal run record and freeze record. Add one mock row with
   `eligible = False`.
2. **Run all four scripts in order:** build, figures, a mock calls file keyed by Q-code committed to a scratch
   ledger, predict (EW0207 control first), score. Every call goes through the same `start`, locks, allow-list and
   guard as phase5 mode.
3. **Must hold:**
   1. all four complete with no guard refusal, traceback or new warning category;
   2. the acquisition lock accepts the rehearsal record and refuses a wrong freeze sha256 and a wrong supplement
      sha256;
   3. the allow-list admits the mock site paths after the locks, refuses them before, and refuses a non-admitted id
      and the bare roots, on the real Phase 5 root names;
   4. the `eligible = False` row is "not testable" with nothing built;
   5. the score writes the pooled primary label, the per-site secondary, and C2 with CIs for GBT_ship and P-V1;
      with fewer than 3 C1 sites it writes "not testable";
   6. the calls join inside the score script reproduces the mock calls exactly;
   7. every path pattern each script opens in phase5 mode was exercised (report the list against the opened-path
      logs, as R12c did);
   8. the R10 reproduction (KM0923, AR26, EW0207) holds with the Phase 5 modules, and the EW0207 control passes;
   9. the refusal suite: R12c's 39 tests ported to Phase 5, plus tests for items 3.2 to 3.6; all pass.
4. **Gate:** any failure stops before Part D and becomes a HOLD. One fix iteration at most without a ruling.

## Part D: deviation note 04 for Phase 4

Draft `R13/DEVIATION_phase4_04.md` (at most 6 lines, unsigned): `phase4_score.py` required calls keyed by pair;
the executor built `phase4_calls_joined.csv` from Steve's committed file through the key after the commit, with
values asserted unchanged; its sha256 was recorded as ledger row 41 before scoring; no value, rule or result
changed. Do not edit any Phase 4 file.

## Part E: pre-registration v3 and sign-off draft (then STOP)

1. **Render `R16/PREREG_phase5_v3.md`** from v2.1 (`R12c/PREREG_phase4_v2_1.md`) with exactly the spec's changes:
   sites and product rules (§3.1, §3.2 with S1 to S3); eligibility (§3.3); primary and secondary criteria with the
   manuscript reason (§3.4); C2 with CIs (§3.5); expectation (§3.6); prior exposure (§4); the B6 table with the five
   Phase 5 scripts' sha256; the acquisition freeze record sha256 `67695831…`; and notes on: the 3.88 bound and
   FK151121; the current gridding region rule; the `vertical_offset` 0.5 m fallback for every Phase 5 site; no step
   requires a person to view candidate data (co-registration pass or fail by the automated thresholds only).
   Report the diff of v2.1 against v3, section by section.
2. **Draft `R16/SIGNOFF_phase5_prereg_v3_DRAFT.md`** on the v2.1 sign-off pattern. Its first 64-hex string must be
   the v3 sha256 (check with the lock's own `read_signoff`). Signature blank. Exposure statement: Steve has seen no
   bathymetry of the eight sites; the only candidate numbers seen are the ACQ-R08 and ACQ-R08b metadata, listed.
3. **Commit by explicit path** (the Phase 5 scripts, R16 scripts and tables, v3, the drafts, the report); one ledger
   row, status "awaiting signature". Do not tag.
4. **Stop.** After Steve signs: tag `phase5-prereg-v3` (that is R17's first step, not this directive's).

## Preservation

R16 rehearsal outputs to OAK `grl_review_archive/R16/` with `SHA256SUMS` verified.

## Out of scope

- Opening any file under a sealed root except mock copies; running phase5 mode on real sites.
- Editing any Phase 4 script, record, pre-registration, sign-off, deviation note or tag.
- Any write to the acquisition repo.
- Changing a frozen function, model, threshold or seed.
- Manuscript text.

## Not pursued

- The read-only CNN audit (deferred by Steve).
- Any change to the target at k = 32 or to RC2511.
