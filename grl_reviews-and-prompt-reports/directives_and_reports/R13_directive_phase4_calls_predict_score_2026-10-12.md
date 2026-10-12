# DIRECTIVE R13 (CNN repo, CPU only): commit the blind artifact calls, then predict and score Phase 4 once

**Date:** 2026-10-12
**Repo:** CNN repo, `grl_reviews-and-prompt-reports/directives_and_reports/`
**Report to:** `R13/REPORT_R13_<date>.md`
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.
**Report length:** at most 8 bullets, plus at most 3 HOLDs.

## 0. Purpose and authority

This directive runs B3 steps 3b, 4 and 5 of pre-registration v2.1 (`R12c/PREREG_phase4_v2_1.md`, sha256
`d6af268bc8211fe9b4ae17439a47e6fdcbe7e5949d8ab387a7be080b2c748334`, tag `phase4-prereg-v2.1`):

1. commit Steve's blind artifact calls and record their sha256 in the ledger (Part B);
2. run `phase4_predict.py --mode phase4` once, EW0207 control first (Part C);
3. run `phase4_score.py --mode phase4` once (Part D).

The order is strict. No prediction exists before the calls are committed and in the ledger. No score is computed
twice. Results are reported whatever they are; no retuning, re-selection or re-scoring with changed rules.

**No rerun after a stop.** Any stop is a HOLD and is not rerun without Steve's ruling. Do not clear any marker or
delete any output.

**Challenge clause.** If any instruction here is wrong or underspecified against the code or the files on disk,
stop at that point, give the evidence, and report it as a HOLD. Do not work around it.

## 1. Preconditions (read only; any failure stops the directive)

1. **State since Part E.** List every commit from `dd732a3` to HEAD with message and paths. None may touch the five
   Phase 4 scripts, v2.1, the v2.1 sign-off, deviation notes 01 to 03, or ledger rows 1 to 39.
2. **v2.1 C1 checks.** Rerun `R12c/scripts/preconditions_r12c.py` (or an equivalent read-only check). All rows pass:
   sign-off holds `d6af268b…`; v2.1 on disk equals v2.1 at its tag; deviation notes 01 to 03 at their sha256; the
   five scripts equal the v2.1 B6 table and are clean; supplement and completion record in the ledger; the citation
   record `6e51efd6…` accepted by the acquisition lock; guard 702 / 702.
3. **Build and pages unchanged.** `sha256sum -c` on `grl_review_archive/phase4/SHA256SUMS` passes (82 files).
   Report the count only.
4. **Prediction and score outputs absent.** Confirm that `grl_review_archive/phase4/predictions/` and every
   `results_*.json` the score lock checks do not exist.

## 2. Part A: read the calls interface (read only; before committing anything)

The calls file cannot be changed once committed. Check first that the scripts read it as Steve wrote it.

1. Read every place in `phase4_common.py`, `phase4_predict.py` and `phase4_score.py` that locates, hashes or parses
   the calls file. Report file and line for:
   1. the path the calls lock expects (or the argument that supplies it);
   2. how the lock finds the sha256 in the ledger (full 64-hex string, which ledger file);
   3. how `phase4_score.py` parses each column, in particular `criteria` (separator for several letters) and the
      accepted values of `dominant` and `ab_whole_survey`;
   4. how Q-codes are joined to pairs through the key, without opening the key yourself.
2. Parse Steve's file (§2.3 below) with the same functions the score script uses, or with an exact copy of its
   parsing lines in a scratch script, and report per Q-code only: the parsed `criteria` set, `dominant`,
   `ab_whole_survey`, and whether each value is accepted. Do not join to the key.
3. **Steve's calls file.** Write it to `R13/phase4_calls.csv` with exactly this content (UTF-8, LF line endings,
   final newline):

```
code,relief,criteria,dominant,ab_whole_survey,note
Q01,yes,a;c,artifact,no,"Edge halos are the strongest pattern, followed by relief. Several blank spots where survey lines do not intersect carry bright edge halos along their edges, running NE-SW, with smaller stripes NW-SE. Relief is present but less prominent than the edge artifacts."
Q02,yes,c,relief,na,"Relief dominates except next to edges, where edge halos dominate. Bright edge halos where there are no-data pixels and along the map boundary. A few small artifacts from poor data acquisition, but these are small."
Q03,weak,a,relief,yes,"Relief visible but very subtle. Very subtle NE-SW survey-line stripes present through the whole survey area. Survey lines are clear in the hillshade; in the other panel they are hard to make out and relief appears to dominate."
Q04,yes,a,relief,no,"Survey-line stripes visible in the hillshade, most visible in the NE quadrant; relief dominates. Tile stitching in the fine-band map is more obvious than on any other page: square boundaries where tiles were computed and stitched are visible. Unsure whether this affects the model."
```

   Its sha256 must be `ae7910210b75d341539642213585652f72b2459d7787cce2bf8c1abc3a06a55f` (1,130 bytes). If not,
   stop and report the difference; do not edit the content to match.
4. **If any value is not accepted** (for example `a;c` is not parsed as two criteria, or `na` or `weak` is
   rejected), or the lock expects the file at a path or name that cannot be met without editing a pinned script,
   stop and report a HOLD with the evidence. Do not reformat Steve's calls.

## 3. Part B: commit the calls and record them

1. If the lock expects the file at another path, copy `R13/phase4_calls.csv` there byte for byte and confirm the
   sha256 is unchanged.
2. Commit the calls file alone, by explicit path, with the message
   `R13: Steve's blind artifact calls (B3 step 3), committed before any prediction`. Report the commit.
3. Append ledger row 40 to `grl_reviews-and-prompt-reports/CHANGE_LEDGER.md`: the calls file's path, full sha256,
   commit, the PDF it was made from (`figset_A_blind.pdf`, `5db55bcf55de1fa347dd813671c4fb9537dfb84f5af0014bf5975245701732d5`),
   and status "committed before prediction". Commit the ledger and this directive by explicit path.
4. **Must hold:** the calls lock passes in a fresh `python -B` process (guard first, 702 / 702) when called with
   the committed file, and nothing else is called. Report it.

## 4. Part C: predict (B3 step 4)

1. Submit one CPU job: `phase4_predict.py --mode phase4` with the calls file argument the lock requires. The EW0207
   environment control runs first inside the script.
2. **Must hold:**
   1. every lock passes, including the calls lock with the committed sha256;
   2. the EW0207 control passes (s_lr identical; GBT ≤ 1e-6 m; P-V1 seed-averaged ≤ 1e-4 log₁₀);
   3. predictions are written for RC2511, SUM1004, AT42-03 and TN159 and are finite for s_lr, GBT_ship and P-V1 on
      every tile; KN182L03 has none.
3. Write `SHA256SUMS` for the new files outside the script and verify it.

## 5. Part D: score once (B3 step 5)

1. Submit one CPU job: `phase4_score.py --mode phase4`, once, with the inputs, calls and stats the script requires.
2. **Must hold:** every lock passes, including the single-run lock; the results and per-pair files are written.
3. Write `SHA256SUMS` for the new files and verify it.

## 6. Report

Steve's calls are committed, so the blind is lifted. Report in full.

1. **Per pair:** role, Q-code, Steve's calls, the artifact-rule outcome (in C1, artifact-dominated, low-amplitude
   artifact-affected, flagged), amplitude statistic, view, n tiles scored, n blocks, and for s_lr, GBT_ship and P-V1
   the point rho, the 95 % CI, undefined-resample fraction and CI estimability. Secondary rows as the pre-registration
   defines them.
2. **C1:** the label (confirmed / partially / not / not testable), the C1 set, and both statements as written in v2.1.
   The pooled row.
3. **C2:** the model comparisons as pre-registered, descriptive only.
4. **C3:** the lockbox results, labelled supporting only.
5. **KN182L03:** not scorable (no k), stated.
6. Take every number from the score script's own output files and cite the file. Do not compute any new statistic
   and do not interpret beyond the pre-registered labels.
7. Note, without analysis, the two items carried from Part E: TN159's 3.25 m HR raster pixel, and Steve's note on
   Q04's tile stitching.

## 7. Preservation

Copy R13 scripts, tables, logs and the report to OAK `grl_review_archive/R13/` with `SHA256SUMS` verified. Phase 4
outputs stay where the scripts write them, with their `SHA256SUMS`.

## 8. Out of scope

- Editing any hash-pinned file, v2.1, its sign-off, any deviation note, or Steve's calls.
- Running predict or score more than once, or after a stop.
- Any new statistic, sensitivity analysis, plot or re-scoring not in v2.1.
- Any write to the acquisition repo.
- Manuscript text.

## 9. Not pursued

- Whether Q04's tile stitching affects the target or the models.
- TN159's HR pixel against its June manifest row.
- The Goff (2010) comparison.
