# DIRECTIVE R12 (Phase 4, steps 2 and 3a, CNN repo, CPU only): k = 32 rehearsal gate, then build the sealed pairs and render the blind pages

**Date:** 2026-10-11
**Author:** steering instance, for Claude Code on Sherlock (CNN repo)
**Predecessors:** R11 (tag `phase4-prereg-v2`); sign-off `R11/SIGNOFF_phase4_prereg_v2.md` (commit 18e51397); deviation note
`R11/DEVIATION_phase4_01.md` (commit 84fd9a18); acquisition ACQ-R07 and ACQ-R07b (Phase 4 step 1 complete).
**Report to:** `directives_and_reports/R12/REPORT_R12_<date>.md`
**Compute:** CPU only. No training, no refitting. **No prediction and no score of any sealed pair.**
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK. Meters unless marked.

---

## 0. Purpose

`PREREG_phase4_v2.md` (sha256 `9795db3b22d6c54f7a85618dbaccafa054e88487b33236a7b2083ecb8c605979`) B3:

- **step 2:** build the sealed pairs (windows, tiles, fields, targets, interior masks, features, amplitude statistic);
- **step 3:** render figure set A for Steve's blind artifact call.

This directive runs step 2 and renders the pages. Steve's calls, then prediction and the single scoring, come later.

**One gate first.** Phase 4 step 1 gave k = 32 to two confirmatory pairs. No development pair has k above 16, so the
CNN-side code has never built or predicted at k = 32, or at a ship cell of 125–151 m. Part A rehearses exactly that on
development data before anything sealed is built. This is a code-path check. It changes no rule, value or candidate.

**Step-1 results (ACQ-R07b; supplement sha256 `6968e04554ddeef3c335587d602cb5fab2185db32b46cbd152475a919eb59206`):**

| pair | role | hr (m) | lr_native (m) | k | products usable |
|---|---|---|---|---|---|
| KN182L03__MGDS_30193 | confirmatory | 1.0 | 128.42 | none | no |
| RC2511__MGDS_32239 | confirmatory | 1.037 | 124.62 | 32 | no (`vertical_offset`) |
| SUM1004__MGDS_33090 | confirmatory | 1.0 | 150.77 | 32 | yes |
| AT42-03__MGDS_32007 | lockbox | 0.989 | 19.7 | 4 | yes |
| TN159__MGDS_21981 | lockbox | 0.991 | 41.0 | 4 | yes |

**Steve must stay blind** to every amplitude, target value, s_lr value, feature value and image of the sealed pairs
until his calls are committed. The only exception is the blind pages themselves. §4 says what the report may contain.

**Scope discipline.** Do the items below and nothing else. If a result suggests a new analysis, list it in one line under
"Not pursued". Do not run it.

**Challenge clause.** Report any wrong or underspecified instruction in §2 of the report, with evidence. Items that are
Steve's call become HOLDs (at most 3). **No Phase 4 script is edited in this directive.** A needed edit would change a
hash pinned in the pre-registration, so it is a HOLD.

---

## Part A: the k = 32 rehearsal gate (development data, rehearsal mode, scratch only)

### A1. Code reading (report with file and line)

1. Where k and lr_native enter, for each of:
   - the band-pass that defines the target (`targets_v1`), and any clip on the upper frequency (R08 cites 0.45);
   - D₁ and the interior masks;
   - `s_lr` (`tile_ship_side` / `build_inputs`, including `lr_px`);
   - the feature table (the `k`, `lr_native` and `log_depth_over_lr` columns, and any column that depends on them);
   - the GBT_ship inputs;
   - the P-V1 inputs (state whether k or lr_native is a model input, a normalisation, or neither).
2. The range of k and lr_native in the training data of GBT_ship (R09 artifact) and of P-V1 (R07).
3. For RC2511 and SUM1004, from their supplement rows only:
   - D₁ = ⌈lr_native / hr_res⌉;
   - f_hi = k · hr_res / lr_native in cycles per pixel, and whether it reaches the clip;
   - `lr_px`;
   - cells across.

### A2. Harness build at sealed-like values

On a scratch copy of manifest v2.1, use the R11 `--rows` harness mechanism (rehearsal only). Write two harness rows
that keep the development data and change only `lr_native_res_m` and `max_recoverable_k`:

| harness row | from | lr_native_res_m | max_recoverable_k |
|---|---|---|---|
| H1 | FK151121__MGDS_22383 (HR 1.0 m) | 150.77 | 32 |
| H2 | NA080__MGDS_31289 (HR 1.0 m) | 124.62 | 32 |

The ship and AUV data are unchanged. These rows are code-path tests, not science, and no value from them is
interpreted.

Run `phase4_build.py --mode rehearsal --rows <harness>` on H1 and H2, then `phase4_predict.py --mode rehearsal` on the
same rows (not scored).

**Must hold:**

1. The build completes: tiles, `F_hr`, `joint_v1`, `targets_v1` with A_fine finite on every tile with valid data,
   `targets_v1_int`, the D₁ interior flags (D₁ = 151 and 125 px), the feature table, and the P-V1 input arrays.
2. The band's upper frequency is below the clip and recorded, and no warning or exception is raised.
3. s_lr is defined, with `lr_px` = 2, on at least 99 % of tiles.
4. GBT_ship and P-V1 predictions are finite on every tile.
5. The fine-band field is not NaN-dominated: report the fraction of finite `F_hr` cells against the unmodified build of
   the same pair.

Report each item as passed or failed, with the counts. Report no target or prediction value.

### A3. The EW0207 environment control

Run it once more in this directive's environment. It must pass, as in R11.

**Gate:** if any A2 item or A3 fails, stop before Part B and report a HOLD. If all pass, continue.

---

## Part B: ledger entries (before the build)

Add to `CHANGE_LEDGER.md`:

1. Phase 4 step 1 supplement: `manifest/pairs_phase4_sealed.parquet` (acquisition repo), sha256
   `6968e04554ddeef3c335587d602cb5fab2185db32b46cbd152475a919eb59206`. Completion run record
   `reports_post_grl_review/ACQ-R06b/step1_run_phase4_completion.json`, sha256
   `fb1e07b24681e8b7b15dec3b232ea78266c6c7cd82ae64518e82b76ae99dcc07`.
2. Deviation note `R11/DEVIATION_phase4_01.md`, commit `84fd9a189f1505dd30f6a92a6045b8ff9767a2d8`, sha256
   `962fd19cd9b3b13dc14ba1e1d37a7860515fb8b7168b33732cbaafe77eb95894`.

Verify both supplement-side sha256 values on disk before writing the rows. Commit the ledger. The supplement lock reads
this row.

---

## Part C: Phase 4 step 2, the build

### C1. Preconditions (any failure stops before the build)

1. The sign-off is committed and holds the v2 sha256. v2 is at its tag.
2. The deviation note is committed and clean.
3. The five Phase 4 scripts match the B6 table.
4. The supplement and the completion run record match the hashes in §B.
5. The guard self-test passes, with 702 of 702 blocked.

### C2. Run

`phase4_build.py --mode phase4` on the five sealed pairs: one Slurm CPU job, with no rehearsal-only arguments.

- The script runs its own locks: sign-off, pre-registration, artifacts, code, supplement, and acquisition run record.
- Record the command, job, node and CPU.
- KN182L03 has no k. Its expected outcome is `not scorable (no k)`, with only `pair_stats.json` written.

**If it stops** (a lock refusal or an error): do not rerun and change nothing. Report it as a HOLD.

---

## Part D: Phase 4 step 3a, the blind pages

Run `phase4_figures.py --mode phase4` on the pairs the build made scorable for figures. A no-k pair gets no page.

- Pages are shuffled with seed 20261011 and coded Q01 onward.
- The key goes in a separate file.
- Record the sha256 of the PDF and of the key.

**Checks, without describing any page's content:**

- the number of pages, against the number of pairs built;
- every page carries only its code and the scale-bar label (the R10 check);
- the key is not in the PDF or the report.

**The executor does not describe, summarize or characterize any page** in the report or anywhere else.

Write `phase4_calls_TEMPLATE.csv`: one row per page code, with the pre-registered columns
`code, relief, criteria, dominant, ab_whole_survey, note` and the value column empty.

**Deliver the PDF and the calls template as separate files beside the report.** Do not deliver the key.

---

## 4. What the report may and may not contain

**May contain, per sealed pair:**

- the build status (`built`, `not scorable (no k)`, or stopped);
- the number of tiles, of interior tiles at D₁, of tiles with a defined A_fine_int, and of non-empty blocks on each
  view;
- the view that B4 will use (primary, fallback, or not scorable), which follows mechanically from those counts;
- D₁, cells across, and whether the window is thinner than any development pair;
- whether product features are used or masked;
- file paths and sha256.

**Must not contain:**

- the amplitude statistic, or any value of A_fine, A_fine_int, `F_hr`, s_lr, any feature or any prediction;
- any image, or any description of a page;
- the page key.

List the files that hold amplitudes or the key (for example `pair_stats.json` and the key file), so that Steve knows not
to open them before his calls are committed.

## 5. Report

- **§1 Bottom line**, at most 8 points, each citing its source file:
  1. challenge findings and the guard;
  2. the A1 code reading;
  3. the A2 / A3 gate;
  4. the ledger entries;
  5. the C1 preconditions;
  6. the build, per pair (counts and view only);
  7. the pages and calls template (count and hashes);
  8. preservation, and the list of files Steve should not open.
- **HOLDs:** at most 3.
- **"Not pursued":** one line per item, no analysis.

## 6. Preservation

- Phase 4 outputs go to `grl_review_archive/phase4/` with `SHA256SUMS`, as the scripts write them.
- Copy the Part A rehearsal outputs to OAK `grl_review_archive/R12/` with `SHA256SUMS` verified.
- Commit the report, the ledger rows, the calls template and the PDF's sha256. Do not commit the key's content to any
  file Steve is told to read.

## 7. Out of scope

- Any edit to a Phase 4 script, frozen code, target, feature, tile or artifact.
- `phase4_predict.py` or `phase4_score.py` in phase4 mode.
- Opening, describing or summarizing any sealed field, page or amplitude beyond §4.
- Rerunning the phase4 build or figures after a stop.
- Anything in the acquisition repo other than reading the supplement and the run record.
- Manuscript text.
- Any analysis not listed above.
