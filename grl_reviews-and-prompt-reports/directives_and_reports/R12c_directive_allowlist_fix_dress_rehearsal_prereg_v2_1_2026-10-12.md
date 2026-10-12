# DIRECTIVE R12c (Phase 4, CNN repo, CPU only): fix the confirmatory allow-list, run a full dress rehearsal on a sealed-style layout, render pre-registration v2.1 for signature, then build and render the blind pages

**Date:** 2026-10-12
**Author:** steering instance, for Claude Code on Sherlock (CNN repo)
**Predecessors:** R12, R12b (HOLD H1, ruled 2026-10-12: option (a)); ACQ-R07c; deviation notes 01 and 02.
**Report to:** `directives_and_reports/R12c/REPORT_R12c_<date>.md` (Parts A to D), then
`directives_and_reports/R12c/REPORT_R12c_partE_<date>.md` (Part E).
**Compute:** CPU only. No training, no refitting. **No prediction and no score of any sealed pair.**
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.

---

## 0. Purpose and ruling

R12b stopped because `phase4_common.apply_allowlist` leaves the bare R09 token `harmonized_confirmatory` in place, so
the guard refuses every confirmatory path after all locks pass. Steve ruled **option (a)**: fix the admit step in
`phase4_common.py` with a narrow rule, re-pin it in a pre-registration revision v2.1, and re-sign.

**The only scientific content that may change is none.** Targets, tiles, features, candidates, artifact rule, C1 to
C3, evaluation, seeds and every other pinned script stay exactly as in v2. The revision changes one B6 hash, adds a
short revision section, and (if needed) one constant that names the pre-registration file.

**Two stops in a row came from path handling that rehearsals never exercised.** Part C therefore rehearses all four
Phase 4 scripts end to end on a scratch copy of a development pair laid out exactly like a sealed confirmatory pair.
Nothing is signed until that passes.

**Steve stays blind**, as in R12 §4. **Stop point:** Part D ends with a report and waits for Steve's signature.
Part E runs only after Steve's signed sign-off for v2.1 is committed and he tells you to continue.

**Scope discipline:** do only what is listed; anything else goes under "Not pursued" in one line.
**Challenge clause:** report wrong or underspecified instructions with evidence. At most 3 HOLDs.

---

## Part A: code reading (read only; report file and line)

1. `r09_common.sealed_tokens`, `_is_sealed`, `_check_r09`, and the guard self-test: how tokens are built, how a path is
   judged, and which self-test probes involve `harmonized_confirmatory`.
2. `phase4_common.apply_allowlist` and `start`: what is admitted, when (before or after the locks), and how.
3. How the locks locate the pre-registration and the sign-off: file paths, constants, arguments. State whether a v2.1
   file can be used without changing any script other than `phase4_common.py`.
4. For each of `phase4_build.py`, `phase4_figures.py`, `phase4_predict.py` and `phase4_score.py` in phase4 mode: every
   path pattern it opens or writes, and every sealed token that could match it after the allow-list. This is the list
   Part C must cover.

## Part B: the fix (`phase4_common.py` only)

### B1. Rule

After every lock has passed, and only then:

1. A path that contains an admitted pair id (full id, for example `RC2511__MGDS_32239`) is allowed, as now.
2. A path that contains `harmonized_confirmatory` and **no** admitted full pair id is refused.
3. Every other sealed token behaves as now.
4. Before the locks nothing changes: the guard self-test must still block 702 of 702.

- Implement it in `phase4_common.py` alone. **Do not edit `r09_common.py` or any other pinned script.**
- If the rule cannot be implemented without editing another file, stop and report a HOLD.
- If Part A.3 shows the pre-registration file name is a constant in `phase4_common.py`, you may change that constant to
  the v2.1 path in the same edit. Report it.

### B2. String-only tests (no file opened)

Install the guard, self-test it, run the locks' allow-list step on the real five sealed rows (SEALED5), and report
`_is_sealed` for:

- the HR and LR paths of KN182L03, RC2511, SUM1004, AT42-03 and TN159: all must be False;
- a fabricated `.../harmonized_confirmatory/nu99/FAKE__MGDS_00000/hr.tif`: must be True;
- a fabricated `.../harmonized_confirmatory/hr.tif` (no pair id): must be True;
- one path for each other sealed-token family (for example `qc_plots/`, `raw_lr/`, `staging_phase1/`) that names no
  admitted pair: must be True;
- the same five sealed paths **before** the allow-list: must be True.

## Part C: dress rehearsal on a sealed-style layout (development data, scratch only)

### C1. Mock layout

1. Copy, not symlink, the harmonized HR and LR files of one development pair into scratch at
   `/scratch/users/scdobbs/grl_review/R12c/mock/harmonized_confirmatory/nu98/<DEVPAIR>/`.
   - Use the same file names and structure as the real confirmatory pairs.
   - Pick a pair with k defined and usable ship products, for example FK151121__MGDS_22383.
2. Add a second mock pair with k defined and products unusable, to cover the RC2511 path. Use a development pair
   whose products are unusable, or the same pair with the flag set in its mock row only.
3. Write mock supplement rows that point at these paths. Use the real supplement's column schema.
4. Also place one development lockbox-style pair under a scratch `harmonized/` path.

### C2. Run

Use the scripts' rehearsal mode and the R11 `--rows` harness.

- Rehearsal-only scaffolding lives in `R12c/scripts/`, not in a pinned script. Examples: treating the mock pair ids
  as sealed tokens and then admitting them, and a mock calls file.
- Every call must go through the same functions phase4 mode uses: `start`, the locks that can run in rehearsal,
  `apply_allowlist`, the guard, and the same open paths.

Run in order:

1. `phase4_build.py`
2. `phase4_figures.py`
3. a mock calls file committed to scratch
4. `phase4_predict.py`, with the EW0207 control first
5. `phase4_score.py`

**Must hold:**

1. All four scripts complete on the mock pairs with no guard refusal, no warning other than the known ResourceWarnings,
   and no exception.
2. A mock path under `harmonized_confirmatory` with a non-admitted id is refused during the run (add one such probe in
   the harness).
3. Every path pattern listed in Part A.4 was actually exercised. Report the list against the files opened.
4. The R10 reproduction check holds: the rehearsal build on the same three development pairs as R10 reproduces the
   frozen artifacts.
5. The full existing refusal-test suite (R10b: 26 tests) passes with the edited `phase4_common.py`. Add the B2 cases as
   new tests.
6. The EW0207 environment control passes.

**Gate:** any failure stops before Part D and becomes a HOLD. Do not iterate the fix more than once without a ruling.

## Part D: pre-registration v2.1, sign-off draft, deviation note 03 (then STOP)

1. **Render `R12c/PREREG_phase4_v2_1.md`.** Leave `R11/PREREG_phase4_v2.md` and its tag untouched. v2.1 is v2 with only
   these changes:
   - the B6 row for `phase4_common.py` (new sha256), plus the file-name constant if changed under B1;
   - a section "Revision v2.1" of at most 10 lines: the defect, the rule in B1, that no sealed file was read and no
     sealed value computed before the revision, and references to deviation notes 01 to 03 and to R12, R12b and R12c.

   Report a `diff` of v2 against v2.1. It must contain nothing else.
2. **Draft `R12c/SIGNOFF_phase4_prereg_v2_1.md`** on the pattern of the v2 sign-off: the v2.1 sha256, the new
   `phase4_common.py` sha256, the unchanged acquisition hashes, and the three v2 errata carried forward unchanged. Add
   this line to the exposure statement:

   > Since v2 was signed, Steve has seen only step-1 metadata (spacings, k, product usability) and lock and stop
   > reports; no HR statistic, amplitude, tile count, target, feature, prediction or image of a sealed pair.

   Leave the signature line blank.
3. **Draft `R12c/DEVIATION_phase4_03.md`** (at most 8 lines): the defect, the fix, the rehearsal result, and the
   revision to v2.1. Leave it unsigned.
4. **Commit by explicit path:** the edited `phase4_common.py`, the R12c scripts, v2.1, both drafts and the report.
   - Do not tag yet.
   - Add one `CHANGE_LEDGER.md` row: the v2.1 render, its sha256, the new `phase4_common.py` sha256, and status "awaiting
     signature".
5. **Stop.** Report Parts A to D and wait.

## Part E: after Steve's signature (only when he says to continue)

### E1. Preconditions (any failure stops)

1. The v2.1 sign-off is committed, signed and holds the v2.1 sha256. Deviation note 03 is signed and committed.
2. Tag `phase4-prereg-v2.1` on the commit that holds v2.1 and the signed sign-off. Add a ledger row.
3. The C1 checks of R12, updated to v2.1 (the five scripts against the v2.1 B6 table), all pass. The guard is at
   702 / 702.
4. **Move the R12b leftovers,** do not delete them:
   - leftovers: `build/KN182L03__MGDS_30193/pair_stats.json`, the empty `build/RC2511__MGDS_32239/` and `fields/`, and
     the two provenance files;
   - destination: `grl_review_archive/phase4_stopped_R12b/` with `SHA256SUMS` verified;
   - then `grl_review_archive/phase4/` must not exist.

### E2. Run

Run R12 Part C2 (`phase4_build.py --mode phase4`, one job, five pairs). Then run R12 Part D (`phase4_figures.py --mode
phase4`, pages shuffled with seed 20261011 and coded Q01 onward, the key in a separate file, and
`phase4_calls_TEMPLATE.csv`). Use the R12b prepared jobs where they still apply.

**If anything stops:** do not rerun and change nothing. Report a HOLD.

### E3. Report

R12 §4 and §5 apply in full:

- **May report:** counts, the view, D₁, cells across, whether features are used or masked, paths and sha256.
- **Must not report:** any amplitude, value, image, description of a page, or the key.

Deliver `figset_A_blind.pdf` and `phase4_calls_TEMPLATE.csv` beside the report. Do not deliver the key. Restate the
do-not-open list.

## Preservation

R12c rehearsal outputs go to OAK `grl_review_archive/R12c/` with `SHA256SUMS` verified. Phase 4 outputs are written by
the scripts as before.

## Out of scope

- Editing any file except `phase4_common.py`, the R12c files and the ledger.
- Editing `r09_common.py`, the other four Phase 4 scripts, frozen artifacts, v2, the v2 sign-off and tag, or deviation
  notes 01 and 02.
- Any write to the acquisition repo.
- Opening any sealed file outside Part E's phase4 run.
- Prediction or scoring of any sealed pair.
- Describing any page.
- Manuscript text.
