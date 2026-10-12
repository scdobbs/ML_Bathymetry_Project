# PREREG_phase4_v2 — Phase 4 pre-registration, version 2 (frozen; tag `phase4-prereg-v2`)

> **Version 2, written 2026-10-10 by directives R10 and R10b and R11. It supersedes `PREREG_phase4_v1.md` (sha256 `5c636f3d…4c1f`, tag `phase4-prereg`), which was never executed. No sealed file had been opened, listed or computed on when v2 was frozen (guard self-tests, REPORT_R09, REPORT_R10, REPORT_R10b, REPORT_R11). Changes from v1 are listed in the final section.**

**Written:** 2026-10-09 by directive R09 Part B, from verified records; not executed in R09. **Rulings in force (Steve, 2026-10-09):** CNN development is closed for this paper; the CNN enters Phase 4 as one arm of a model-complexity comparison; the frozen candidates are s_lr, GBT_ship and P-V1 (seed-averaged); development scoring uses interior tiles at D₁ on the primary ten real-relief units (lu14 shown on all tiles beside the interior rows and kept out of every median); Phase 4's main question is whether within-site ranking of unresolved relief transfers to the sealed units, whose ship cells (103–151 m) are coarser than anything in development. Candidate comparisons are secondary and descriptive.

**Fallback status from R09 Part A:** **validated** (|paired median Δ| = 0.020 for s_lr and 0.002 for GBT_ship on 9 primary units, both ≤ 0.05; REPORT_R09 §1): B4 rule 2 is in force.

## B1. Sealed units

| set | pairs | role |
|---|---|---|
| confirmatory | KN182L03__MGDS_30193 (nu01a), RC2511__MGDS_32239 (nu01b), SUM1004__MGDS_33090 (nu03) | primary |
| lockbox | AT42-03__MGDS_32007, TN159__MGDS_21981 | supporting only |

**Lockbox history (R09 §B1, CONFIRMED with file):** the submitted manuscript's '10 never-trained units' table (S5_external_test; REPORT_57 maps it to dir55/external_test/results.json) scores AT42-03__MGDS_32007 (lu03, 237 patches, within-pair R2 0.654) and TN159__MGDS_21981 (lu17, 113 patches, 0.202) with the frozen all-native model labelled 'het_production_v1_repro (all-native)' in the results file, i.e. the §C reproduction of het_production_v1 that REPORT_55 §C labels het_retrain_nonrepro (the production weights were never persisted; REPORT_55 §A). Earlier, the first het training (report 22, 2026-06-30) used TN159 as a test pair and AT42-03 as a validation pair, and report 26 reports calibration on both; REPORT_55 then declared them the 'never-trained reserve' and asserted them absent from every train / held set of the expanded retrain. Source: `/oak/stanford/groups/hilley/auv_ship_colocated_bathy/analysis_archive_2026-09-28/dir55/external_test/results.json` (sha256 `7df91a6387d966de160581d668f96b235927ef0f59e3899a44ae28aff35e10a9`); reports 22, 26, 55, 57 (sha256 in `scores/lockbox_history.json`). Lockbox results are therefore reported as supporting and labelled "scored by the submitted pipeline's het_production_v1_repro model (2026-07-08) and used as validation / test splits of the first het training (2026-06-30)".

## Prior exposure

Before this version was frozen, Steve had seen the sealed pairs' bathymetry three times, and no score:
- **The original co-registration sign-off,** which covered the lockbox pairs.
- **The ACQ-R03 co-registration review** of nu01a and nu03: hillshades before and after the shift, contours and the search surface (ruling R6, 2026-10-01).
- **A re-view on 2026-10-10** of the harmonized AUV and ship bathymetry of the sealed pairs (depth and hillshade). He noted that nu03 shows very low relief, with survey track lines as the only visible structure. No fine-band, difference or roughness layer was viewed, and no rule was changed as a result.

The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.

## B2. Frozen candidates (artifact paths and sha256)

1. **s_lr:** the `ref_s_lr` definition: `s_lr = max(SD of the valid LR cells of the 256-px window, 0.5 m)` from `preprocessing/extract_clean_patches.py:build_inputs` (the tiles_v1.1 `s_lr` column), computed from the sealed pairs' LR only. No fitting. Code sha256 (`extract_clean_patches.py` at this commit): `b130a34395b1d718764b73f5b6ee8ea3d1d48f79032abef12faa260e68460498`.
2. **GBT_ship:** no persisted all-16 model existed (R05 fitted it in memory to score EW0207, configuration = the most frequent per-fold choice, 2 of 16). Refit once in R09 with the frozen bar code (`bar_v1_1.py`, features `bar_v1_1.ALL`, target log₁₀ A_fine at tile scale, cap 200 rows per unit, seed 20261001), configuration chosen by the frozen inner criterion over all 16 units (leave-one-unit-out, eval_v1_1 within-pair ρ, median over units): cfg 18 = {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.1, "min_child_samples": 20}; artifact `grl_review_archive/R09/artifacts/gbt_ship_all16_A_fine_tile.joblib`, sha256 `ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f`; record `gbt_ship_all16_record.json`. The inner criterion selects a different configuration from R05's EW0207 mode (cfg 16 = {'max_depth': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'min_child_samples': 20}); the artifact therefore differs from R05's in-memory EW0207 model (max |Δ| 4.72e-01 m on EW0207's tiles; REPORT_R09 §1, HOLD). It is not refit after this.
3. **P-V1:** the three R07 all-unit fine-tuned models (`R07/models/P-V1/seed<seed>/fold_ALL/ckpt.pt`, OAK `grl_review_archive/R07/models/`), seed-averaged as in R07 (mean of the three seeds' log₁₀ block predictions; tile = RMS over the 16 blocks): seed 20261002 `d3f4d92824f2d0ac2a062d07a759e459d82debe79f1a671270e7810957665b32`, seed 20261003 `a8c8cf7ebfc8de079f47c1c90b94f4a6f12ff8cb51818ec819c33c8a594c50a2`, seed 20261004 `6f2d050108a35f5329b4df5bddbdd9b6b217568cc6f5006f47b70e2883ef6c3b`. Inference at Phase 4 is CPU float32. On EW0207 it agrees with the R07 GPU bf16 predictions to 6.5e-03 in max |Δ log₁₀| (seed-averaged) and gives the same all-tile ρ to three decimals (REPORT_R10 §5).

No other predictor is scored at Phase 4.

## B3. Order of operations at Phase 4 (each step a separate directive or job; nothing out of order)

1. **Acquisition repo:** The acquisition step is the frozen `src/phase4/sealed_step1.py --mode phase4` (tag `phase4-acq-freeze`; `freeze_record.json` sha256 `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb`; script sha256 `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1`), rehearsed in ACQ-R06 and ACQ-R06b.
   - Confirmatory pairs get `hr_native_res_m`, `lr_native_res_m` and `max_recoverable_k` from the ACQ-R02 chain: the C2a rule, with depth = median of the harmonized `lr.tif`, and the C2c sweep `stage_c2c_recalibrate.sweep`, where k is the largest k with the verdict `recoverable_signal`.
   - The lockbox pairs keep their June values (0.989 / 19.7 m and 0.991 / 41.0 m; k = 4 from the June sweep record). Their swath is fetched from the June plan; if the fetch is incomplete, their products are unavailable.
   - `ship_products_usable` is true if and only if the products were built and carry no contract-v2.1 flag.
   - A data-tied builder failure makes the products unavailable. A pair with no k gets nothing built. Any other error stops the step before the supplement is written.
   - The step writes `manifest/pairs_phase4_sealed.parquet` once, with the per-k verdicts and the source of k.

   Its sha256 is recorded in `CHANGE_LEDGER.md`. No context grid is built.
2. **CNN repo:** `phase4_build.py --mode phase4`. It also writes each pair's amplitude statistic (B4). `max_recoverable_k` is not set here: it comes from the supplement of step 1 (the acquisition repo owns the rule). A pair with no recoverable k is not built. It verifies the acquisition run record (complete; cites the freeze record and the supplement's sha256) before building. A pair's product features are used only if its supplement row has `ship_products_usable = true`; otherwise they are masked as for a development pair whose products are not used. **No prediction or score yet.**
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
4. **Predict:** `phase4_predict.py --mode phase4`, on CPU. The script first reproduces the EW0207 reference predictions (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and refuses to continue otherwise.
5. **Score once:** `phase4_score.py --mode phase4`. Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.

## B4. Scoring rules

- **Statistic:** the `eval_v1_1` within-pair Spearman ρ at tile scale. Each sealed pair is its own unit.
- **Blocks:** 4 × 4-tile blocks in tile-grid coordinates (r0 / 128, c0 / 128). A block is non-empty if it holds at least one scored tile. **B_min = 8.**
- **Tiles:**
  0. A pair with no recoverable k has no fine band. It is **not scorable (no k)**, has no figure page, and is outside C1.
  1. **Primary view:** interior tiles at D₁ (≥ 95 % of joint-valid px at ≥ D₁ = ⌈lr_native / hr_res⌉ px from any invalid px), scored against A_fine. Used if the pair has ≥ 10 interior tiles in ≥ B_min non-empty blocks.
  2. **Fallback view:** tiles with a defined A_fine_int (≥ 2,048 interior px), scored against A_fine_int. Used if rule 1 fails and the pair has ≥ 10 such tiles in ≥ B_min non-empty blocks.
  3. Otherwise the pair is **not scorable** and is outside C1. If either view has ≥ 10 tiles, its point ρ is reported, labelled "too few blocks, not in C1", with no CI.
- **Uncertainty per pair:** a bootstrap over non-empty blocks, resampled with replacement to the same number of blocks, 2,000 resamples, a fresh `numpy.random.default_rng(20261009)` per pair, 95 % percentile CI. Resamples with an undefined ρ are counted and excluded; if they exceed 5 %, the CI is reported as not estimable and the pair is outside C1.
- **Cells per window:** `256 × hr_res / lr_native` is reported for each sealed pair beside the development minimum of 3.88. A pair below it is flagged "window thinner than any development pair". No rule is attached. From the ACQ-R02 records, the three confirmatory pairs are expected at about 2.5, 2.1 and 1.7, all below the development minimum.
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

  Expectation from development (REPORT_R10 §6.3; each pair as its own unit, held-out predictions): on the pairs of the ten real-relief units in the C1 set, s_lr met the C1 condition on 9 of 14 at D₁ and 8 of 12 at the 125 m distance. Pair ρ at D₁ ranged from -0.09 to +0.82, median +0.41. Without AR26, which the 2026-10-09 ruling keeps out of development medians: 8 of 13, range -0.09 to +0.63, median +0.39. If three independent units each met the condition at the 9-of-14 rate, all three would do so with probability 0.27, some but not all with 0.69, and none with 0.05. If the C1 set has two units, both pass with probability 0.41, one with 0.46, and neither with 0.13, at the same rate. At sealed-like coarseness (ship grid block-averaged to about 125 m, target unchanged; REPORT_R11 §5), s_lr met the condition on 9 of 14 pairs, pooled mean +0.37 [+0.29, +0.41]. Report each sealed unit's ρ and CI against these.
- **C1 summary, secondary: pooled ranking.** The mean of s_lr's ρ over the units of the C1 set, with a 95 % CI from a block bootstrap stratified by unit (non-empty blocks resampled within each unit, 2,000 resamples, one `numpy.random.default_rng(20261013)`, units in ascending pair-id order). It is reported with a statement of whether its lower bound is above zero, and again with the low-amplitude artifact-affected units included. It does not change the C1 label. Expectation from development (REPORT_R10b §4.4): over all triples of the 14 development pairs at D₁, the lower bound was above zero in 346 of 364; the pooled mean ranged from +0.08 to +0.68, median +0.39. At the 125 m interior distance, the lower bound was above zero in 199 of 220 triples.
- **C2, secondary and descriptive: model complexity.** Per sealed unit, report GBT_ship − s_lr, P-V1 − s_lr and P-V1 − GBT_ship. One sentence per comparison stating its direction across units. No superiority claim is made: with at most 3 units, even 3 of 3 gives a one-sided sign test of p = 0.125. State this in the result.
- **C3, supporting:** C1, the C1 summary and C2 on the lockbox pairs, reported separately with their history label.

## B6. Provenance requirements at Phase 4

Every Phase 4 script runs in `phase4` mode only if `SIGNOFF_phase4_prereg_v2.md` exists and holds this document's sha256. Before running, it verifies that sha256, the candidate artifacts (B2), the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`) and the five Phase 4 scripts against this document:

| script | sha256 |
|---|---|
| `phase4_common.py` | `99407ec6ce43f2f4005130d67c9025f229df4de20c9560ee79b4bf713e805111` |
| `phase4_build.py` | `98370164406f1c75a8c76ea342432644d896d89bc5e6548c68b0ff65ec0b46d5` |
| `phase4_figures.py` | `44e87590d247b67cf27f02ef0761af39a9a522af385c35b10bd0ce25b8551052` |
| `phase4_predict.py` | `7267a9c85579565044088c931a60c6dd88d12cdacf68d92795bf6b323ed01315` |
| `phase4_score.py` | `9aefb6ff7f67eef2a066478949786fa597af920a048fe34b3d3c8d0545a1382f` |

In `phase4` mode the five sealed manifest rows are read only from `manifest/pairs_phase4_sealed.parquet` (beside manifest v2.1), which must hold exactly the five pre-registered pair ids with the v2.1 columns and whose sha256 must be recorded in `CHANGE_LEDGER.md`; development rows come from the frozen v2.1 file. After the supplement lock, every script also verifies the acquisition run record: the acquisition repo's `reports_post_grl_review/ACQ-R06b/step1_run_phase4_*.json` must have status complete in `phase4` mode, cite the freeze record sha256 `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb` (the `freeze_record.json` on disk must hash to it and list `src/phase4/sealed_step1.py` at `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1`) and cite the supplement's sha256; otherwise it refuses. `phase4_predict.py` first reproduces the EW0207 reference predictions from the R10 inputs on OAK (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and writes the comparison to its provenance record. Outputs go to `grl_review_archive/phase4/` with `SHA256SUMS`. `phase4_score.py` refuses to run a second time.

## Changes from v1

- **G1** — a pair counts in C1 only with ≥ B_min non-empty bootstrap blocks (B_min from R10 §A6.3); a pair with enough tiles but too few interior blocks uses the fallback target. Reason: a block bootstrap on a handful of blocks has no coverage.
- **G2** — the interior-pixel score (A_fine_int) is reported beside the interior-tile score for every pair, as secondary. Reason: the two views agreed on the development units (R09 Part A) and the second row shows where they do not.
- **G3** — a pair leaves C1 if Steve's blind call marks whole-survey stripes or ribbing and the median of its scored target is below 0.04 m; C1 is stated with and without such pairs. Reason: R08 / R08b showed the fine-band target on the six low-amplitude units is dominated by survey artifacts.
- G4 (a wide-window s_lr as a secondary row) was accepted on 2026-10-10 and withdrawn the same day, before any sealed file was opened. In development, s_lr showed no loss of skill down to 3.88 cells across the window, and the 64-cell row was below s_lr on 14 of 15 pairs of the ten real-relief units (+0.12 to +0.51 against +0.39 to +0.63 on the pairs with ship cells of 57 m or more; REPORT_R10 §6.6).
- the "not testable" outcome of C1 (v1 had no outcome for an empty set);
- the calls-file fields (`code, relief, criteria, dominant, ab_whole_survey, note`) and the figure seed 20261011 with codes Q01–Q05;
- the sign-off file `SIGNOFF_phase4_prereg_v2.md` and the single-run lock of `phase4_score.py`;
- CPU inference for P-V1 (R10 §A4: CPU float32 against the R07 GPU bf16 values).
- the pooled row (C1 summary, secondary; R10b §A1);
- the "not scorable (no k)" outcome for a pair with no recoverable k (B4 rule 0);
- cells per window reported for each sealed pair, no rule attached;
- the sealed-row supplement `manifest/pairs_phase4_sealed.parquet` and its ledger lock (R10b §A3);
- the EW0207 environment control before any Phase 4 prediction (R10b §A5);
- the development expectation stated with and without AR26.
- B3 step 1 names the frozen acquisition step (`sealed_step1.py --mode phase4`, tag `phase4-acq-freeze`) and its rules for the confirmatory and the lockbox pairs, `ship_products_usable`, builder failures and the supplement (R11 §C2);
- B3 step 2: the acquisition run record is verified before building, and a sealed pair's product features are used only if its supplement row has `ship_products_usable = true` (R11 §C3);
- the C1 expectation for a two-unit C1 set and at sealed-like coarseness (ship grid block-averaged to about 125 m; R11 §C5);
- the prior-exposure statement after B1 (R11 §C7);
- the acquisition run-record lock of B6 (R11 §A2.2).

## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)

- **Defect (REPORT_R12b H1):** `r09_common.sealed_tokens` holds the bare directory token `harmonized_confirmatory`; `phase4_common.apply_allowlist` dropped only tokens naming an admitted pair, cruise or MGDS id, so that token survived and the guard refused every confirmatory raster after all locks had passed (job 47541164 stopped at RC2511's first `rasterio.open`; no byte was read).
- **Rule (R12c §B1), implemented in `phase4_common.apply_allowlist` only, applied only after every lock has passed:** a path containing an admitted full pair id is allowed; a path containing `harmonized_confirmatory` and no admitted full pair id is refused; every other sealed token behaves as before; before the locks nothing changes (the guard self-test still blocks 702 of 702).
- **Also changed in `phase4_common.py`:** the two constants naming the pre-registration file and the sign-off file now point at this revision (`R12c/PREREG_phase4_v2_1.md`, `R12c/SIGNOFF_phase4_prereg_v2_1.md`). No other script, artifact, target, feature, candidate, rule, seed or threshold changed; `r09_common.py` is untouched.
- **Blindness:** before this revision no sealed file was read and no sealed value was computed (R12: both runs stopped in the locks; R12b: the guard refused the first open). The only sealed-pair output that exists is KN182L03's `not scorable (no k)` record, to be moved aside under `grl_review_archive/phase4_stopped_R12b/` before the build reruns (R12c Part E1).
- **Rehearsal (R12c Part C):** the four Phase 4 scripts ran end to end in rehearsal mode on development pairs copied into a sealed-style `harmonized_confirmatory/nu98/<pair>/` layout (products used and masked) and a `harmonized/<pair>/` lockbox-style layout, through the same `start`, locks, `apply_allowlist` and guard as phase4 mode; a non-admitted `harmonized_confirmatory` probe was refused during the run; the R10 reproduction and refusal-test suites passed. v2 (`R11/PREREG_phase4_v2.md`, tag `phase4-prereg-v2`) is untouched.
- References: deviation notes 01 (`84fd9a18`), 02 (`faedeb08`) and 03; REPORT_R12, REPORT_R12b, REPORT_R12c.
