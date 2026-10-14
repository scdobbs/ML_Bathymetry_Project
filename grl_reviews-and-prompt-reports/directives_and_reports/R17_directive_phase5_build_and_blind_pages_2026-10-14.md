# DIRECTIVE R17 (CNN repo): Phase 5 build and blind pages

**Date:** 2026-10-14
**Report to:** `grl_reviews-and-prompt-reports/directives_and_reports/R17/REPORT_R17_<date>.md`
**Report length:** at most 6 bullets.

## 0. Purpose and rulings

Step 1 was rerun with local co-registration (ACQ-R13, freeze v4.2). 5 of 8 sites are eligible:
- Escanaba__MGDS_30368;
- KN161L04__MGDS_22436 (Mid-Atlantic Ridge 16.5°N);
- KN173L02__MGDS_31675 (Bermuda);
- MV1003__MGDS_31860 (Chile margin);
- AT26-24__MGDS_31954 (Dorado Outcrop).

The v3.1 pooled primary criterion (at least 3) is testable. Build these five sites, make the blind pages, and stop. Prediction and scoring are R18, after Steve's calls are committed.

**Steve's rulings (2026-10-14):**
1. Pre-registration v3.1 governs. No further amendments, sign-offs or dress rehearsals.
2. The 30 m co-registration bound is replaced by the ACQ-R13 local co-registration test. Sites that pass it are fully eligible.
3. **The blind stays.** Steve sees only the blind PDF until his calls are committed. Your report shows no sealed amplitude, target, s_lr, feature, prediction or image, and no site-to-Q mapping.
4. **Fix plumbing yourself.** A failure that changes no computation, threshold, rule or site set (a path, a version constant, a lock naming the wrong freeze or pre-registration, the l.134 freeze-sha text check, the supplement path) is fixed, committed by explicit path, logged in one line, and you continue. No rehearsal.
5. **Stop only if** a fix would change a computation, a threshold, a rule or the site set, or would expose a sealed value to Steve.

## 1. Inputs

`phase5/scripts/phase5_common.py`:
- l.47: freeze record `freeze_record_phase5_v4_2.json`, `ACQ_FREEZE_SHA = "168df0c68cf0e4493b62c78be6da2755020ce3bd53c9bd1f2ed02ad9270eadc3"`;
- l.48: `ACQ_STEP1_SHA = "2c9ea8488e88a411aed3b63319a879fcbfe70e4a6d6ec0c420ccdcc7f05273a7"`;
- the supplement path: `manifest/pairs_phase5_sealed_v4_1.parquet`;
- l.44 stays on v3.1.

Companion line from ACQ-R13:
- supplement `manifest/pairs_phase5_sealed_v4_1.parquet` sha256 `6fe0d59057842481b06513fd731951fe9177a6898fc55f8677004800616949f0`;
- run record `PHASE5/step1_run_phase5_20261014T204117Z.json` sha256 `947ede388682ea3969099beac01abb95f62cad7f3b3cd988eed631e71c00763e`;
- site rasters under OAK `phase5_v4_1/`.

The ACQ-R10c supplement and the superseded ACQ-R13 v4 records are not used.

## 2. Build and pages

1. Build tiles and features for the five eligible sites with `phase5_build.py`, from the ACQ-R13 site rasters.
2. **Must hold:** tiles that ACQ-R13 marked `joint_valid = 0` at quadrant boundaries (Escanaba, Chile margin) are excluded by the frozen tile filter. Report the count excluded per site, as counts only.
3. **Check which ship input the frozen primary score uses** (s_lr from the LR grid, or the ship products). MV1003's ship products are unusable. If the primary needs them, stop and report. If only C2 needs them, GBT_ship and any other product-dependent row are reported as unavailable for MV1003 in R18.
4. Make one set of Q-coded blind pages covering the five sites in a single random order, as v3.1 specifies, with `phase5_figures.py`. The site-to-Q key stays sealed.
5. Write the calls template for Steve in the v3.1 format.
6. Commit by explicit path; one `CHANGE_LEDGER.md` row; OAK archive with `SHA256SUMS` verified.

## 3. Report

1. The tiles per site and the boundary tiles excluded, as counts only.
2. The §2.3 answer.
3. Every plumbing fix, one line each.
4. The blind PDF path and sha256, and the calls template path.
5. Confirmation that no sealed value, image or Q key appears in the report or in any file Steve is pointed to.

## 4. Out of scope

- Prediction, scoring, or opening predictions.
- Any change to models, thresholds, rules or the site set.
- Any write to the acquisition repo.
