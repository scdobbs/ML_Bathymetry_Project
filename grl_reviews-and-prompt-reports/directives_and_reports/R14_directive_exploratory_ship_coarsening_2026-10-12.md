# DIRECTIVE R14 (CNN repo, CPU only): exploratory ship-coarsening test on development pairs

**Date:** 2026-10-12
**Repo:** CNN repo, `grl_reviews-and-prompt-reports/directives_and_reports/`
**Report to:** `R14/REPORT_R14_<date>.md`
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.
**Report length:** at most 8 bullets, plus at most 3 HOLDs.
**Status of the analysis:** EXPLORATORY. It was designed after the Phase 4 result was known. It is reported as
exploratory and does not change any Phase 4 label.

## 0. Purpose

Phase 4 (R13) did not confirm C1. Both scorable confirmatory pairs had about 2 ship cells across a 256 px tile
(RC2511 2.13, SUM1004 1.70), against a development minimum of 3.88. The working explanation is that s_lr computed
from a 2 x 2 coarse grid carries almost no roughness information. That explanation is post hoc.

This directive tests it on development data only. It coarsens the ship grids of the development real-relief pairs
to sealed-like sampling, recomputes s_lr and the GBT_ship features through the frozen code, and rescores ranking
with the frozen evaluation. The question is how ranking skill changes as ship cells per tile fall from native to
about 2.

**No sealed file is opened.** The R09 guard is installed and self-tested (702 / 702) in every process. No Phase 4
script, frozen artifact or pinned file is edited. New code lives in `R14/scripts/` and calls frozen functions.

**Challenge clause.** If any instruction here is wrong or underspecified against the code or the data, stop at that
point, give the evidence, and report it as a HOLD. Do not work around it.

## 1. Expectations stated before running (exploratory, not a test)

Recorded here so the reading is not shaped by the result:

1. If coarse sampling explains the Phase 4 failure, pooled s_lr at 1.7 to 2.1 cells across falls to a 95 % CI
   whose lower bound is at or below zero, or drops by at least 0.20 from the native level.
2. If the GBT's positive sign at Phase 4 reflects its use of wider windows and the ship-resolution inputs, pooled
   GBT_ship at the same levels keeps a CI lower bound above zero.
3. If neither s_lr nor GBT_ship degrades materially at 2 cells across, coarse sampling does not explain the
   Phase 4 failure, and site differences or the target are the more likely cause.

Report which of these the data match, using these exact definitions.

## 2. Preconditions (read only; any failure stops the directive)

1. The five Phase 4 scripts equal the v2.1 B6 table and are clean. Report `git status` of `phase4/` before and
   after the directive; they must be identical.
2. The R09 guard self-test passes, 702 / 702.
3. Identify the development pair set used for the R10 C1 calibration: the real-relief pairs scored with each pair as
   its own unit (R10 §6, 14 pairs). List them with unit, hr_res, lr_native, k, D1 and native cells across
   (`256 * hr_res / lr_native`). If R10's list cannot be reproduced exactly, stop and report.
4. Locate the GBT_ship leave-one-unit-out fold models with the frozen configuration (cfg 18), if they were
   persisted (R04, R05 or R09). Report their paths and sha256, and confirm that the fold for each pair's unit
   excludes that unit. If none are persisted, see §4.3.

## 3. The coarsened ship grids

1. **Levels.** For each pair, target cells across `c` in {8.0, 4.0, 3.0, 2.1, 1.7}. The target ship cell size is
   `lr_target = 256 * hr_res / c`. Skip a level when `lr_target <= lr_native` (that pair is already at or coarser
   than the level); report skipped levels. The native grid is the reference level.
2. **Method.** Area-weighted block averaging of the pair's native harmonized LR raster onto a grid of cell size
   `lr_target`, aligned to the native grid's origin, in the pair's own CRS. A coarse cell is valid when at least
   50 % of its area is covered by valid native cells. Write the rasters and validity masks to
   `/scratch/users/scdobbs/grl_review/R14/coarse/<pair>/c<level>/`. Record each raster's sha256.
3. **Must hold.** At the native level the code path reproduces the frozen tiles, s_lr and features exactly (as R10
   and R12c did). Report the comparison.
4. **Caveat to state in the report, not to correct.** Block averaging a modern grid is not the same as an old
   wide-beam sonar: it removes resolution but not sonar noise, footprint shape or sounding density. The test
   isolates sampling, not instrument.

## 4. Predictors

1. **s_lr.** Recompute through the frozen `extract_clean_patches` path with the pair row's `lr_native_res_m` set to
   `lr_target` and its LR raster and validity paths pointed at the coarsened files, as the R12 harness did for
   spacing and k. Report per level the fraction of tiles where s_lr is undefined (fewer than 2 valid coarse cells)
   or at the 0.5 m floor.
2. **GBT_ship features.** Recompute through the frozen `features_v1` path with the same row change, so that every
   ship-side feature, `lr_native` and `log_depth_over_lr` reflect the coarsened grid. Keep `k` and `hr_res` as in
   the frozen row. Product features: use them where the pair's products are usable at native resolution, block
   averaged the same way; if the frozen code cannot take coarsened products without editing it, mask them (as for
   RC2511) and report it.
3. **GBT_ship model.** Use the leave-one-unit-out fold model whose training excluded the pair's unit, trained on
   native-resolution data, so that each prediction is out of unit and applied to coarsened inputs, as at Phase 4.
   If no fold models were persisted, refit leave-one-unit-out with cfg 18 exactly as the frozen bar was fit, on
   native features, and confirm that the refit reproduces the frozen native-level per-pair ρ within 0.01 before
   using it. **Do not use the all-16 Phase 4 artifact**: it was trained on these pairs.
4. **Not scored:** P-V1 and any other CNN (development is closed; the all-16 checkpoints are in-sample here).

## 5. Targets and scoring

1. **Primary (target fixed).** Keep each pair's frozen target, tile set, interior view at D1 and blocks exactly as
   in R10. Only the ship side changes. This isolates the predictors.
2. **Secondary (target rebuilt).** Rebuild the band with lower edge `f_lr = hr_res / lr_target` and the frozen
   upper edge, and D1 = ceil(lr_target / hr_res), with the frozen B4 view rules (interior, then fallback, then not
   scorable). This mimics how the sealed pairs' targets were defined. Report the view used per pair and level.
3. **Statistic.** eval_v1.1 within-pair Spearman ρ, block bootstrap over 4 x 4-tile blocks, 2,000 resamples,
   `default_rng(20261009)` per pair, B_min = 8; pooled mean over pairs with the unit-stratified bootstrap,
   seed 20261013. Use the frozen scoring functions.
4. **Must hold.** The native level of the primary analysis reproduces R10 §6: s_lr passes on 9 of 14 pairs, and
   the pooled mean is +0.392 [+0.317, +0.430]. If not, stop and report.

## 6. Report

1. **Main table, primary analysis:** one row per level (native, 8, 4, 3, 2.1, 1.7): number of pairs scored;
   for s_lr and GBT_ship, the pooled mean ρ with 95 % CI and the number of pairs whose CI lower bound is above
   zero; the fraction of tiles with undefined or floored s_lr.
2. **Per-pair table:** ρ and CI for s_lr and GBT_ship at every level scored, with native cells across.
3. **Secondary analysis:** the same main table with the rebuilt target, and the views used.
4. **Reading:** which of the §1 expectations the data match, under their exact definitions. No other interpretation.
5. State in the report that the analysis is exploratory and post hoc, and the caveat of §3.4.

## 7. Preservation and record

Commit the R14 scripts, tables and report by explicit path; never `git add -A`. Add one `CHANGE_LEDGER.md` row
(status "exploratory; does not change Phase 4"). Copy scripts, tables, logs and the coarsened-grid hash list to OAK
`grl_review_archive/R14/` with `SHA256SUMS` verified. The coarsened rasters may stay on scratch; their sha256 list
is preserved.

## 8. Out of scope

- Opening, listing or computing on any sealed or lockbox file.
- Editing any Phase 4 script, frozen code, frozen artifact, the pre-registration or any deviation note.
- Retraining any model on coarsened data, or tuning anything to the coarsened levels.
- Any change to a Phase 4 result or label.
- Manuscript text.

## 9. Not pursued

- P-V1 or any CNN at coarsened levels.
- The six artifact-affected units and EW0207.
- Simulating old-sonar noise or footprints.
- Recomputing k for coarsened grids.
