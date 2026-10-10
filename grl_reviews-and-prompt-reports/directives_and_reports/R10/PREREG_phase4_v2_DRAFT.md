# PREREG_phase4_v2 — Phase 4 pre-registration, version 2 (frozen; tag `phase4-prereg-v2`)

> **Version 2, written 2026-10-10 by directive R10. It supersedes `PREREG_phase4_v1.md` (sha256 `5c636f3d…4c1f`, tag `phase4-prereg`), which was never executed. No sealed file had been opened, listed or computed on when v2 was frozen (guard self-tests, REPORT_R09 and REPORT_R10). Changes from v1 are listed in the final section.**

**Written:** 2026-10-09 by directive R09 Part B, from verified records; not executed in R09. **Rulings in force (Steve, 2026-10-09):** CNN development is closed for this paper; the CNN enters Phase 4 as one arm of a model-complexity comparison; the frozen candidates are s_lr, GBT_ship and P-V1 (seed-averaged); development scoring uses interior tiles at D₁ on the primary ten real-relief units (lu14 shown on all tiles beside the interior rows and kept out of every median); Phase 4's main question is whether within-site ranking of unresolved relief transfers to the sealed units, whose ship cells (103–151 m) are coarser than anything in development. Candidate comparisons are secondary and descriptive.

**Fallback status from R09 Part A:** **validated** (|paired median Δ| = 0.020 for s_lr and 0.002 for GBT_ship on 9 primary units, both ≤ 0.05; REPORT_R09 §1): B4 rule 2 is in force.

## B1. Sealed units

| set | pairs | role |
|---|---|---|
| confirmatory | KN182L03__MGDS_30193 (nu01a), RC2511__MGDS_32239 (nu01b), SUM1004__MGDS_33090 (nu03) | primary |
| lockbox | AT42-03__MGDS_32007, TN159__MGDS_21981 | supporting only |

**Lockbox history (R09 §B1, CONFIRMED with file):** the submitted manuscript's '10 never-trained units' table (S5_external_test; REPORT_57 maps it to dir55/external_test/results.json) scores AT42-03__MGDS_32007 (lu03, 237 patches, within-pair R2 0.654) and TN159__MGDS_21981 (lu17, 113 patches, 0.202) with the frozen all-native model labelled 'het_production_v1_repro (all-native)' in the results file, i.e. the §C reproduction of het_production_v1 that REPORT_55 §C labels het_retrain_nonrepro (the production weights were never persisted; REPORT_55 §A). Earlier, the first het training (report 22, 2026-06-30) used TN159 as a test pair and AT42-03 as a validation pair, and report 26 reports calibration on both; REPORT_55 then declared them the 'never-trained reserve' and asserted them absent from every train / held set of the expanded retrain. Source: `/oak/stanford/groups/hilley/auv_ship_colocated_bathy/analysis_archive_2026-09-28/dir55/external_test/results.json` (sha256 `7df91a6387d966de160581d668f96b235927ef0f59e3899a44ae28aff35e10a9`); reports 22, 26, 55, 57 (sha256 in `scores/lockbox_history.json`). Lockbox results are therefore reported as supporting and labelled "scored by the submitted pipeline's het_production_v1_repro model (2026-07-08) and used as validation / test splits of the first het training (2026-06-30)".

## B2. Frozen candidates (artifact paths and sha256)

1. **s_lr:** the `ref_s_lr` definition: `s_lr = max(SD of the valid LR cells of the 256-px window, 0.5 m)` from `preprocessing/extract_clean_patches.py:build_inputs` (the tiles_v1.1 `s_lr` column), computed from the sealed pairs' LR only. No fitting. Code sha256 (`extract_clean_patches.py` at this commit): `b130a34395b1d718764b73f5b6ee8ea3d1d48f79032abef12faa260e68460498`.
2. **GBT_ship:** no persisted all-16 model existed (R05 fitted it in memory to score EW0207, configuration = the most frequent per-fold choice, 2 of 16). Refit once in R09 with the frozen bar code (`bar_v1_1.py`, features `bar_v1_1.ALL`, target log₁₀ A_fine at tile scale, cap 200 rows per unit, seed 20261001), configuration chosen by the frozen inner criterion over all 16 units (leave-one-unit-out, eval_v1_1 within-pair ρ, median over units): cfg 18 = {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.1, "min_child_samples": 20}; artifact `grl_review_archive/R09/artifacts/gbt_ship_all16_A_fine_tile.joblib`, sha256 `ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f`; record `gbt_ship_all16_record.json`. The inner criterion selects a different configuration from R05's EW0207 mode (cfg 16 = {'max_depth': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'min_child_samples': 20}); the artifact therefore differs from R05's in-memory EW0207 model (max |Δ| 4.72e-01 m on EW0207's tiles; REPORT_R09 §1, HOLD). It is not refit after this.
3. **P-V1:** the three R07 all-unit fine-tuned models (`R07/models/P-V1/seed<seed>/fold_ALL/ckpt.pt`, OAK `grl_review_archive/R07/models/`), seed-averaged as in R07 (mean of the three seeds' log₁₀ block predictions; tile = RMS over the 16 blocks): seed 20261002 `d3f4d92824f2d0ac2a062d07a759e459d82debe79f1a671270e7810957665b32`, seed 20261003 `a8c8cf7ebfc8de079f47c1c90b94f4a6f12ff8cb51818ec819c33c8a594c50a2`, seed 20261004 `6f2d050108a35f5329b4df5bddbdd9b6b217568cc6f5006f47b70e2883ef6c3b`.

4. **s_lr_W64 (secondary, labelled; [HOLD A6.5: confirmatory cells_across could not be checked from the manifest] by the rule of R10 §A6.5):** the frozen `features_v1.native_features` s_lr on the 64 × 64-cell window of the pair's context grid centred on the tile centre, as run in R08 §6 (`wide_window_r08.py`, sha256 `14dfecb7ebc12220d8dca0c288fbedf0128e61fb916f292ba5a4f969f1324995`); windows with < 50 % valid cells give NaN. It is not part of C1.

No other predictor is scored at Phase 4.

## B3. Order of operations at Phase 4 (each step a separate directive or job; nothing out of order)

1. **Acquisition repo:** build the sealed pairs' contract-v2.1 ship products with the frozen code. [HOLD A6.5: whether the context grids are needed depends on the s_lr_W64 ruling.] The manifest rows of the sealed pairs must carry `hr_native_res_m`, `lr_native_res_m`, `max_recoverable_k` and the harmonized paths before step 2 (`phase4_build.py` refuses a row without them).
2. **CNN repo:** `phase4_build.py --mode phase4`. It also writes each pair's amplitude statistic (B4). **No prediction or score yet.**
3. **Blind artifact judgement:** `phase4_figures.py --mode phase4` produces figure set A for the 5 sealed pairs, pages shuffled with seed 20261011 and coded Q01 to Q05, key in a separate file. Steve judges each page against the R08 §3.1 criteria and fills one row per page:

   | field | values |
   |---|---|
   | `code` | Q01 to Q05 |
   | `relief` | yes / weak / no |
   | `criteria` | any of a, b, c, or none |
   | `dominant` | relief / artifact / mixed |
   | `ab_whole_survey` | yes / no / na: do stripes (a) or ribbing (b) extend across the whole survey? |
   | `note` | free text |

   The calls file is committed and its sha256 recorded in `CHANGE_LEDGER.md` **before** any prediction is computed.
4. **Predict:** `phase4_predict.py --mode phase4`, on CPU.
5. **Score once:** `phase4_score.py --mode phase4`. Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.

## B4. Scoring rules

- **Statistic:** the `eval_v1_1` within-pair Spearman ρ at tile scale. Each sealed pair is its own unit.
- **Blocks:** 4 × 4-tile blocks in tile-grid coordinates (r0 / 128, c0 / 128). A block is non-empty if it holds at least one scored tile. **B_min = 8.**
- **Tiles:**
  1. **Primary view:** interior tiles at D₁ (≥ 95 % of joint-valid px at ≥ D₁ = ⌈lr_native / hr_res⌉ px from any invalid px), scored against A_fine. Used if the pair has ≥ 10 interior tiles in ≥ B_min non-empty blocks.
  2. **Fallback view:** tiles with a defined A_fine_int (≥ 2,048 interior px), scored against A_fine_int. Used if rule 1 fails and the pair has ≥ 10 such tiles in ≥ B_min non-empty blocks.
  3. Otherwise the pair is **not scorable** and is outside C1. If either view has ≥ 10 tiles, its point ρ is reported, labelled "too few blocks, not in C1", with no CI.
- **Uncertainty per pair:** a bootstrap over non-empty blocks, resampled with replacement to the same number of blocks, 2,000 resamples, a fresh `numpy.random.default_rng(20261009)` per pair, 95 % percentile CI. Resamples with an undefined ρ are counted and excluded; if they exceed 5 %, the CI is reported as not estimable and the pair is outside C1.
- **Amplitude statistic:** the median of the scored target over the scored tiles of the view in use (m). It is computed at B3 step 2, before any prediction.
- **Secondary rows, for every pair with ≥ 10 tiles on that view:** all-tile ρ against A_fine; A_fine_int ρ where the primary view is in use. They are reported beside the primary and do not enter C1.
- **Artifact handling:**
  - **artifact-dominated** (`dominant = artifact`): reported separately and outside C1;
  - **low-amplitude artifact-affected** (`ab_whole_survey = yes` and amplitude statistic < 0.04 m): outside C1, and included in the second statement of C1;
  - any other pair with criteria a, b or c is flagged and stays in.

  The 0.04 m floor lies between the development groups: under interior scoring at D₁ the six artifact-affected units have unit medians of at most 0.029 m and the ten real-relief units at least 0.049 m (REPORT_R08b §5).

## B5. Criteria (fixed now)

- **C1, primary: ranking transfers.** The **C1 set** is the confirmatory units that are scorable, not artifact-dominated and not low-amplitude artifact-affected. Its size is stated.
  - **Confirmed** if s_lr's block-bootstrap CI lower bound is above zero on every unit of the C1 set.
  - **Partially confirmed** if on some but not all.
  - **Not confirmed** if on none.
  - **Not testable** if the C1 set is empty.

  The same reading is also stated with the low-amplitude artifact-affected units included.

  Expectation from development (R10 §A6.2, each pair as its own unit, held-out predictions): on the pairs of the ten real-relief units, s_lr met the C1 condition on 9 of 14 at D₁ and 8 of 12 at the 125 m distance; pair ρ ranged from -0.09 to 0.82, median 0.41, at D₁. Report each sealed unit's ρ and CI against these.
- **C2, secondary and descriptive: model complexity.** Per sealed unit, report GBT_ship − s_lr, P-V1 − s_lr and P-V1 − GBT_ship [and s_lr_W64 − s_lr, if the A6.5 HOLD puts it in force]. One sentence per comparison stating its direction across units. No superiority claim is made: with at most 3 units, even 3 of 3 gives a one-sided sign test of p = 0.125. State this in the result.
- **C3, supporting:** C1 and C2 on the lockbox pairs, reported separately with their history label.

## B6. Provenance requirements at Phase 4

Every Phase 4 script runs in `phase4` mode only if `SIGNOFF_phase4_prereg_v2.md` exists and holds this document's sha256. Before running, it verifies that sha256, the candidate artifacts (B2), the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`; `wide_window_r08.py` if in force) and the five Phase 4 scripts against this document:

| script | sha256 |
|---|---|
| `phase4_common.py` `0348070cf60f3cf5f4874887b8910b81225a6851cca181cd9e55dbac2587c72c` |
| `phase4_build.py` `f6ec3bee726e0846199a22dce70654710500abafd59a228efaab3ed123c4f7c7` |
| `phase4_figures.py` `4db422d2f76b4cacf41ff26e2ab846c437ba31261a07252262e77c499d7ce84a` |
| `phase4_predict.py` `dc211e0409f633cdd696fb2b17ab3c842fed13b3a8c1f0b2eee9a89ba4631a94` |
| `phase4_score.py` `03d760d55582742b9154589a3f62afa747d7abf3304c037fadb80c3feb7a5a02` |

Outputs go to `grl_review_archive/phase4/` with `SHA256SUMS`. `phase4_score.py` refuses to run a second time.

## Changes from v1

- **G1** — a pair counts in C1 only with ≥ B_min non-empty bootstrap blocks (B_min from R10 §A6.3); a pair with enough tiles but too few interior blocks uses the fallback target. Reason: a block bootstrap on a handful of blocks has no coverage.
- **G2** — the interior-pixel score (A_fine_int) is reported beside the interior-tile score for every pair, as secondary. Reason: the two views agreed on the development units (R09 Part A) and the second row shows where they do not.
- **G3** — a pair leaves C1 if Steve's blind call marks whole-survey stripes or ribbing and the median of its scored target is below 0.04 m; C1 is stated with and without such pairs. Reason: R08 / R08b showed the fine-band target on the six low-amplitude units is dominated by survey artifacts.
- **G4** — if a sealed 256-px window holds fewer ship cells across than the development minimum, the 64-cell s_lr of R08 §6 enters as a labelled secondary row. Reason: the sealed ship cells (103–151 m) are coarser than any development unit.
- the "not testable" outcome of C1 (v1 had no outcome for an empty set);
- the calls-file fields (`code, relief, criteria, dominant, ab_whole_survey, note`) and the figure seed 20261011 with codes Q01–Q05;
- the sign-off file `SIGNOFF_phase4_prereg_v2.md` and the single-run lock of `phase4_score.py`;
- CPU inference for P-V1 (R10 §A4: CPU float32 against the R07 GPU bf16 values).
