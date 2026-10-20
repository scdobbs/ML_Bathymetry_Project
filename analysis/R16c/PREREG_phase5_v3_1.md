# PREREG_phase5_v3 — Phase 5 pre-registration, version 3.1 (a second pre-registered blind test)

> **Version 3.1, rendered 2026-10-13 by directive R16c from `R12c/PREREG_phase4_v2_1.md` (sha256 `d6af268b…8334`, tag `phase4-prereg-v2.1`) with exactly the changes of `phase_5/PHASE5_design_spec_2026-10-13.md` (ruled by Steve on 2026-10-13) and the directive's notes. Phase 4 stays closed and is reported as it is: C1 not confirmed (0 of 2), pooled −0.20 [−0.34, −0.02] (REPORT_R13). "Phase 5" keeps its paths, locks, records and outputs separate from Phase 4. When v3 was rendered no Phase 5 candidate file had been opened by the CNN repo and the acquisition step had not run in phase5 mode; the only candidate information that exists is the ACQ-R08 / ACQ-R08b metadata (Prior exposure). Changes from v2.1 are listed in "Changes in v3".**

**Why (spec §1):** Phase 4 did not confirm C1. Both scorable confirmatory sites had about 2 ship cells per tile, outside the development range, and R14 showed that coarse sampling alone does not explain the failure. Phase 5 tests the same claim, with the same frozen models, on new sites inside the development sampling range.

**Carried from v2.1 (rulings of 2026-10-09, in force):** **Written:** 2026-10-09 by directive R09 Part B, from verified records; not executed in R09. **Rulings in force (Steve, 2026-10-09):** CNN development is closed for this paper; the CNN enters Phase 4 as one arm of a model-complexity comparison; the frozen candidates are s_lr, GBT_ship and P-V1 (seed-averaged); development scoring uses interior tiles at D₁ on the primary ten real-relief units (lu14 shown on all tiles beside the interior rows and kept out of every median); Phase 4's main question is whether within-site ranking of unresolved relief transfers to the sealed units, whose ship cells (103–151 m) are coarser than anything in development. Candidate comparisons are secondary and descriptive.

**Fallback status from R09 Part A:** **validated** (|paired median Δ| = 0.020 for s_lr and 0.002 for GBT_ship on 9 primary units, both ≤ 0.05; REPORT_R09 §1): B4 rule 2 is in force.

## B1. Sites (spec §3.1–§3.3)

| unit | site | high resolution | ship partner | pair id |
|---|---|---|---|---|
| p5u01 | Galápagos Spreading Center | MGDS:18210, 1 m product only (MGDS:31811 not used) | TN188, Simrad EM300, 2005 | `TN188__MGDS_18210` |
| p5u02 | Escanaba Trough | MGDS:30368 | Escanaba, Simrad EM300, 1998 | `Escanaba__MGDS_30368` |
| p5u03 | Monterey Canyon | MGDS:24756 | Monterey, Simrad EM300, 1998 | `Monterey__MGDS_24756` |
| p5u04 | Mid-Atlantic Ridge 16.5°N | MGDS:22436 | KN161L04, SeaBeam 2112, 2000 | `KN161L04__MGDS_22436` |
| p5u05 | La Jolla Canyon | MGDS:21415 | MV1209, Kongsberg EM122, 2012 | `MV1209__MGDS_21415` |
| p5u06 | Bermuda | MGDS:31675 | KN173L02, SeaBeam 2112, 2003 | `KN173L02__MGDS_31675` |
| p5u07 | Chile margin | MGDS:31860 | MV1003, Simrad EM122, 2010 | `MV1003__MGDS_31860` |
| p5u08 | Dorado Outcrop | MGDS:31954 | AT26-24, Kongsberg EM122, 2014 | `AT26-24__MGDS_31954` |

Each site is one unit and one pair: one high-resolution mosaic, one ship cruise. The sites come from ACQ-R08b; no other site may be added and no site is replaced. Pair ids follow `RECORD_CONTRACT_phase5.md` (`<ship cruise>__<hr id with ':' → '_'>`); units follow the acquisition site list (`inputs/phase5_sites.csv`) order. The Phase 4 sealed pairs are not used, and no Phase 4 sealed pair joins the development set.

**Mechanical product rules (applied by script at step 1, without viewing):**

1. **Resolution group:** the finest group of grids with cell size 0.985 to 4.99 m whose union covers at least 80 % of the largest such group (the ACQ-R08b rule).
2. **Repeat surveys** (overlapping grids of the same seafloor, for example Monterey): for each overlap, keep the grid whose acquisition date is closest to the ship cruise's year; ties go to the larger valid area. Dates come from file names or metadata. Non-overlapping grids are mosaicked. Ruling S2: two grids overlap when their validity masks share ≥ 50 % of the smaller grid's valid area; below that they are mosaicked with the rule order as priority; an undated grid counts as a tie.
3. **Dispersed grids (ruling S1):** after rules 1 and 2 the kept grids go to the unchanged development mosaic logic (`grid_lr.grid_region`, `harmonize_new.hr_mosaic`).
4. **Ship side:** the named cruise's raw swath, gridded by the frozen development chain with contract v2.1 products; harmonized, co-registered and k-swept exactly as the development pairs were. Ruling S3: the current `grid_lr` region rule is used.

**Eligibility (fixed now; decided at step 1 from metadata only).** A site enters the C1 set only if all hold:

1. co-registration passes the development QA thresholds (ACQ-R03: offset ≤ 30 m, overlap ≥ 1 km², ≥ 1 256-px tile with ≥ 50 % joint-valid cells) and the vertical-datum check (ACQ-R11b): on the ship grid's joint-valid cells, the 95th percentile of |r − median r|, where r is the area-averaged high-resolution field minus the ship field after the rigid shift, divided by |median ship depth| there, is ≤ T_G = 0.200712 (calibrated on development pairs only). Ship line-to-line inconsistency is not checked.
2. the frozen sweep recovers a k;
3. the gridded ship spacing gives 3.88 to 51.2 ship cells across a 256-px window (256 × unrounded HR pixel / lr_native), and the HR spacing is 0.985 to 4.99 m;
4. at scoring, the B4 view rules give a scorable view (10 tiles in at least 8 blocks).

A site failing 1 to 3 is **not testable** at step 1, with its reason (supplement column `eligibility_reason`); it has no build, figure page, prediction or score. A site failing 4 is **not scorable** at step 5. Then the artifact rule applies at the blind call. No site is replaced or added.

**Notes (v3):**

- **The 3.88 lower bound and FK151121.** The bound is FK151121's development value, 256 × 1.000274 / 65.99 = 3.8804 cells. The frozen Phase 5 chain, with today's region rule, gives FK151121 lr_native 66.09 m and 3.8746 cells, which would be "not testable". The bound is kept as written (ruling ACQ-R09 H-1, Steve, 2026-10-13). Nothing changes unless a site lands within about 0.3 % of 3.88; the lowest ACQ-R08b estimate is 4.32 cells (Mid-Atlantic Ridge).
- **The gridding region rule.** Step 1 grids the ship swath with the current `grid_lr` rule (`grid_region`: the ship grid on the HR patch cluster; ruling S3). FK151121's development grid predates that rule (whole footprint, commit `ce0ebb3`). ACQ-R09 reproduced NA080 and KM0923 bit-identically under the current rule, and FK151121 bit-identically only under its recorded region in a separate harness.
- **`vertical_offset` fallback.** The contract-v2.1 `vertical_offset` flag uses its 0.5 m fallback threshold for every Phase 5 site, because no CNN s_lr exists for them. This is the frozen builder's behaviour, as it was for the Phase 4 sealed pairs.
- **No person views candidate data at step 1.** Co-registration passes or fails by the automated development thresholds and the vertical-datum check only (only its pass or fail is recorded); no QC figure is produced, and the step-1 report carries only status, eligibility fields, spacings, cells across, k, product status and usability, file counts and record sha256 values (ACQ-R09 §2.10). The only viewing in Phase 5 is Steve's blind call on the Q-coded pages of step 2 (B3 step 3).

## Prior exposure

- Steve has seen none of the eight sites' bathymetry. That includes the June QC figure for Bermuda and the La Jolla Zephyr survey (stated 2026-10-13).
- Everyone has seen the ACQ-R08 and ACQ-R08b metadata (listed in REPORT_ACQ-R08b §5). That includes one median depth per site, rounded to 100 m.
- The executor that ran ACQ-R08b saw only those numbers.
- Steve has seen the development and known-bad G values of the vertical-datum check (ACQ-R11, ACQ-R11b), and no value from any Phase 5 site.

The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.

## B2. Frozen candidates (artifact paths and sha256)

1. **s_lr:** the `ref_s_lr` definition: `s_lr = max(SD of the valid LR cells of the 256-px window, 0.5 m)` from `preprocessing/extract_clean_patches.py:build_inputs` (the tiles_v1.1 `s_lr` column), computed from the sealed pairs' LR only. No fitting. Code sha256 (`extract_clean_patches.py` at this commit): `b130a34395b1d718764b73f5b6ee8ea3d1d48f79032abef12faa260e68460498`.
2. **GBT_ship:** no persisted all-16 model existed (R05 fitted it in memory to score EW0207, configuration = the most frequent per-fold choice, 2 of 16). Refit once in R09 with the frozen bar code (`bar_v1_1.py`, features `bar_v1_1.ALL`, target log₁₀ A_fine at tile scale, cap 200 rows per unit, seed 20261001), configuration chosen by the frozen inner criterion over all 16 units (leave-one-unit-out, eval_v1_1 within-pair ρ, median over units): cfg 18 = {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.1, "min_child_samples": 20}; artifact `grl_review_archive/R09/artifacts/gbt_ship_all16_A_fine_tile.joblib`, sha256 `ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f`; record `gbt_ship_all16_record.json`. The inner criterion selects a different configuration from R05's EW0207 mode (cfg 16 = {'max_depth': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'min_child_samples': 20}); the artifact therefore differs from R05's in-memory EW0207 model (max |Δ| 4.72e-01 m on EW0207's tiles; REPORT_R09 §1, HOLD). It is not refit after this.
3. **P-V1:** the three R07 all-unit fine-tuned models (`R07/models/P-V1/seed<seed>/fold_ALL/ckpt.pt`, OAK `grl_review_archive/R07/models/`), seed-averaged as in R07 (mean of the three seeds' log₁₀ block predictions; tile = RMS over the 16 blocks): seed 20261002 `d3f4d92824f2d0ac2a062d07a759e459d82debe79f1a671270e7810957665b32`, seed 20261003 `a8c8cf7ebfc8de079f47c1c90b94f4a6f12ff8cb51818ec819c33c8a594c50a2`, seed 20261004 `6f2d050108a35f5329b4df5bddbdd9b6b217568cc6f5006f47b70e2883ef6c3b`. Inference at Phase 4 is CPU float32. On EW0207 it agrees with the R07 GPU bf16 predictions to 6.5e-03 in max |Δ log₁₀| (seed-averaged) and gives the same all-tile ρ to three decimals (REPORT_R10 §5).

No other predictor is scored at Phase 4.

B2 applies unchanged at Phase 5 (spec §2): no retraining, refitting or tuning; the development set is unchanged.

## B3. Order of operations at Phase 5 (each step a separate directive or job; nothing out of order)

1. **Acquisition repo (ACQ-R10):** the frozen `src/phase5/sealed_step1_p5.py --mode phase5` (tag `phase5-acq-freeze-v2`; code commit `f88e21c`; freeze record `reports_post_grl_review/PHASE5/freeze_record_phase5_v2.json` sha256 `f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438`; script sha256 `83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47`), rehearsed in ACQ-R09 and, against this freeze, in ACQ-R11b.
   - It runs locks 1 to 4: the sign-off holding this document's sha256, the freeze record, exactly the eight listed sites, and no second completed run.
   - It fetches the named cruise's raw swath and grids it with the frozen development chain. It selects and mosaics the HR grids by the B1 product rules, then harmonizes and co-registers. It runs the frozen C2c sweep (k = the largest k with verdict `recoverable_signal`) and builds the contract-v2.1 products.
   - `ship_products_usable` is true if and only if the products were built and carry no contract-v2.1 flag (frozen Phase 4 rule C1).
   - It decides eligibility items 1 to 3 from metadata only.
   - It writes `manifest/pairs_phase5_sealed.parquet` once: the 64 columns of the Phase 4 supplement, then `coreg_pass, k_recoverable, cells_across, hr_res_in_range, eligible, eligibility_reason`. It also writes the run record `reports_post_grl_review/PHASE5/step1_run_phase5_<UTC>.json` with the keys of `RECORD_CONTRACT_phase5.md`. No record key holds a depth statistic, amplitude or residual.

   The supplement's sha256 is recorded in `CHANGE_LEDGER.md`.
2. **CNN repo, build and blind pages:**
   - `phase5_build.py --mode phase5` verifies the acquisition run record and builds each eligible site. It writes each site's amplitude statistic (B4). `max_recoverable_k` comes from the supplement; the acquisition repo owns the rule. A site's product features are used only if its supplement row has `ship_products_usable = true`; otherwise they are masked.
   - A not-testable site is not opened; only its status is written.
   - `phase5_figures.py --mode phase5` then produces figure set A for the eligible sites. Pages are shuffled with seed 20261011 and coded Q01 onward; the key is in a separate file.
   - **No prediction or score yet.**
3. **Blind artifact judgement:** Steve judges each page against the R08 §3.1 criteria and fills one row per page:

   | field | values |
   |---|---|
   | `code` | Q01 onward, one per page |
   | `relief` | yes / weak / no |
   | `criteria` | any of a, b, c, or none |
   | `dominant` | relief / artifact / mixed |
   | `ab_whole_survey` | yes / no / na: do stripes (a) or ribbing (b) extend across the whole survey? |
   | `note` | free text |

   The calls file stays keyed by Q-code. It is committed and its sha256 recorded in `CHANGE_LEDGER.md` **before** any prediction is computed. The Q-code to pair join is part of this procedure: `phase5_score.py` joins the committed file to the pairs through the figure key, asserting every value unchanged, and writes no separate joined file (Phase 4 deviation note 04).
4. **Predict:** `phase5_predict.py --mode phase5`, on CPU. The script first reproduces the EW0207 reference predictions (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and refuses to continue otherwise. Not-testable sites are not predicted.
5. **Score once:** `phase5_score.py --mode phase5`. It assembles its own scoring inputs (pinned, R16b): it reads only the build and prediction outputs under `grl_review_archive/phase5/` (per eligible site the build statistics, tiles_v1.1, the D₁ interior table, and the predictions, whose sha256 must equal the prediction record's), and it refuses any external inputs table or statistics file. Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.

## B4. Scoring rules

- **Statistic:** the `eval_v1_1` within-pair Spearman ρ at tile scale. Each sealed pair is its own unit.
- **Blocks:** 4 × 4-tile blocks in tile-grid coordinates (r0 / 128, c0 / 128). A block is non-empty if it holds at least one scored tile. **B_min = 8.**
- **Tiles:**
  0. A site that failed eligibility at step 1 (B1 items 1–3) is **not testable**, with its reason; it has no build, figure page, prediction or score, and is outside C1. A pair with no recoverable k has no fine band. It is **not scorable (no k)**, has no figure page, and is outside C1 (at Phase 5 this cannot occur for an eligible site: k is eligibility item 2).
  1. **Primary view:** interior tiles at D₁ (≥ 95 % of joint-valid px at ≥ D₁ = ⌈lr_native / hr_res⌉ px from any invalid px), scored against A_fine. Used if the pair has ≥ 10 interior tiles in ≥ B_min non-empty blocks.
  2. **Fallback view:** tiles with a defined A_fine_int (≥ 2,048 interior px), scored against A_fine_int. Used if rule 1 fails and the pair has ≥ 10 such tiles in ≥ B_min non-empty blocks.
  3. Otherwise the pair is **not scorable** and is outside C1. If either view has ≥ 10 tiles, its point ρ is reported, labelled "too few blocks, not in C1", with no CI.
- **Uncertainty per pair:** a bootstrap over non-empty blocks, resampled with replacement to the same number of blocks, 2,000 resamples, a fresh `numpy.random.default_rng(20261009)` per pair, 95 % percentile CI. Resamples with an undefined ρ are counted and excluded; if they exceed 5 %, the CI is reported as not estimable and the pair is outside C1.
- **Cells per window:** `256 × hr_res / lr_native` is eligibility item 3 at step 1 (3.88 to 51.2; B1). It is reported for each site, and the score output keeps v2.1's flag "window thinner than any development pair" (below 3.88), with no rule attached. From the ACQ-R08b metadata the eight sites are expected at about 4.3 to 24.5 cells.
- **Amplitude statistic:** the median of the scored target over the scored tiles of the view in use (m). It is computed at B3 step 2, before any prediction.
- **Secondary rows, for every pair with ≥ 10 tiles on that view:** all-tile ρ against A_fine; A_fine_int ρ where the primary view is in use. They are reported beside the primary and do not enter C1.
- **Artifact handling:**
  - **artifact-dominated** (`dominant = artifact`): reported separately and outside C1;
  - **low-amplitude artifact-affected** (`ab_whole_survey = yes` and amplitude statistic < 0.04 m): outside C1, and included in the second statement of C1;
  - any other pair with criteria a, b or c is flagged and stays in.

  The 0.04 m floor lies between the development groups: under interior scoring at D₁ the six artifact-affected units have unit medians of at most 0.029 m and the ten real-relief units at least 0.049 m (REPORT_R08b §5).

## B5. Criteria (fixed now)

- **C1 set:** the eligible sites (B1 items 1–3) that are scorable (item 4), not artifact-dominated and not low-amplitude artifact-affected. Its size is stated.
- **C1, primary: pooled ranking transfers (spec §3.4).** The mean within-site s_lr ρ over the C1-set sites, with a 95 % CI from a block bootstrap stratified by unit (non-empty blocks resampled within each site, 2,000 resamples, one `numpy.random.default_rng(20261013)`, sites in ascending pair-id order; a resample with any undefined site ρ is dropped and counted, > 5 % → not estimable).
  - **Confirmed** if the CI lower bound is above zero.
  - **Not confirmed** otherwise.
  - **Not testable** if fewer than 3 sites are in the C1 set.

  The pooled row is also reported with the low-amplitude artifact-affected sites included. That row does not change the label.
- **C1-per-site, secondary.** Each site's s_lr CI as in v2.1 (B4), with the count of sites whose lower bound is above zero and the v2.1 all-sites label, reported beside the primary:
  - **confirmed** if the lower bound is above zero on every site of the C1 set;
  - **partially confirmed** if on some;
  - **not confirmed** if on none;
  - **not testable** if the set is empty.

  It is also stated with the low-amplitude artifact-affected sites included.
- **Why the change, for the manuscript (Steve's condition):** with n sites and a per-site pass rate of 9 in 14 at development, "every site passes" occurs with probability about 0.64^n even if the method works as in development (about 3 % at n = 8). So the v2.1 per-site label cannot confirm transfer with more than two or three sites. The pooled mean estimates average transfer directly and has adequate power. The change is made after Phase 4 was known. Under it, Phase 4 would still read "not confirmed" (pooled −0.20 [−0.34, −0.02]), so the change does not alter that result. Both readings are reported for Phase 5.
- **Expectation (spec §3.6):** at development, pooled s_lr is +0.392 [+0.317, +0.430] over 14 pairs, and +0.35 at 1.7 cells per tile (R14). Phase 4 was −0.20 on two sites. Phase 5 tests which of these the new sites resemble. For the secondary reading: development s_lr met the per-site condition on 9 of 14 pairs at D₁ (REPORT_R10 §6.3), pair ρ −0.09 to +0.82, median +0.41.
- **C2, descriptive: model complexity (spec §3.5).**
  - GBT_ship and P-V1 against s_lr, per site and pooled over the C1 set. Each candidate's ρ and each difference from s_lr (GBT_ship − s_lr, P-V1 − s_lr) gets a block-bootstrap 95 % CI.
  - The CIs use the same resamples as s_lr: per site, the block draws of `default_rng(20261009)`; pooled, the draws of `default_rng(20261013)`, sites in ascending pair-id order. The same undefined-resample rule applies.
  - `phase5_score.py` (pinned in B6) computes them. The frozen Phase 4 score gives no CI for GBT_ship or P-V1.
  - P-V1 − GBT_ship is reported as a point value.
  - One sentence per comparison states its direction across sites. No superiority claim is made.
- C3 (the lockbox role) has no counterpart at Phase 5.

## B6. Provenance requirements at Phase 5

Every Phase 5 script runs in `phase5` mode only if `R16c/SIGNOFF_phase5_prereg_v3_1.md` exists and holds this document's sha256. Before running, it verifies the following against this document:

- that sha256;
- the candidate artifacts (B2);
- the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`);
- the five Phase 5 scripts;
- the five Phase 4 scripts, which the Phase 5 scripts import unchanged for every frozen function (targets, tiles, interior masks, features, model inputs, the three models, the EW0207 control, the artifact rule, the single-run lock and the bootstrap seeds).

It also requires the acquisition freeze record sha256 `f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438` to be stated here.

| script | sha256 |
|---|---|
| `phase5_common.py` | `cdca2c9b699f8a06eb29bc095d3c3636304cabfb941c88a22fdae8b555df1b66` |
| `phase5_build.py` | `973b775dda073fa3b4def8073fc7ce71b22a64df20d84a30e4c1b2b892d45633` |
| `phase5_figures.py` | `47921a51699d370f90f7082590f03f6a69477d83c25978c154b3312535d811a0` |
| `phase5_predict.py` | `96797b85309f360b971c15f49a4c02a3fdf4ce7d2e276af33c31e621ee8c6429` |
| `phase5_score.py` | `7eb7c1cc3fc0acbfebf9b4f2c517c145b035699612f6e8ee31865c263787d2a5` |

Imported unchanged (v2.1 B6, tag `phase4-prereg-v2.1`):

| script | sha256 |
|---|---|
| `phase4_common.py` | `99407ec6ce43f2f4005130d67c9025f229df4de20c9560ee79b4bf713e805111` |
| `phase4_build.py` | `98370164406f1c75a8c76ea342432644d896d89bc5e6548c68b0ff65ec0b46d5` |
| `phase4_figures.py` | `44e87590d247b67cf27f02ef0761af39a9a522af385c35b10bd0ce25b8551052` |
| `phase4_predict.py` | `7267a9c85579565044088c931a60c6dd88d12cdacf68d92795bf6b323ed01315` |
| `phase4_score.py` | `9aefb6ff7f67eef2a066478949786fa597af920a048fe34b3d3c8d0545a1382f` |

**Supplement lock.** In `phase5` mode the eight site rows are read only from `manifest/pairs_phase5_sealed.parquet`, beside manifest v2.1. The file must meet all of these:

- it holds exactly the eight pre-registered pair ids;
- its first 64 columns are the Phase 4 supplement's names, order and arrow types (`manifest/pairs_phase4_sealed.parquet`, sha256 `6968e045…9206`);
- they are followed by exactly the six eligibility columns;
- `eligible` agrees with B1 items 1–3 on every row;
- its sha256 is recorded in `CHANGE_LEDGER.md`.

Development rows come from the frozen v2.1 file. The eligible rows are the admitted sites; a row with `eligible = False` is "not testable (<eligibility_reason>)".

**Acquisition run-record lock.** After the supplement lock, every script verifies the acquisition run record. The acquisition repo's `reports_post_grl_review/PHASE5/step1_run_phase5_*.json` must:

- have status complete in `phase5` mode;
- cite the freeze record sha256 `f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438` in `locks.lock2_freeze_record`;
- cite the supplement's sha256.

The `freeze_record_phase5_v2.json` on disk must hash to that value and list `src/phase5/sealed_step1_p5.py` at `83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47`. Otherwise the script refuses.

**Guard.**
- The R09 guard of Phase 4 is installed unchanged and self-tests 702 of 702 probes.
- The Phase 5 extension seals the roots `sealed_candidates_v3` and `auv_ship_colocated_bathy/phase5`, every site's pair id, ship cruise and HR id, and MGDS:31811. It self-tests on the real Phase 5 names before any lock.
- After every lock, a path is admitted only if it contains an eligible site's full pair id. Every other path holding a Phase 5 token stays refused, and the Phase 4 tokens are never admitted.

**Calls and single run.** `phase5_predict.py` and `phase5_score.py` require the committed code-keyed calls file, with its sha256 in `CHANGE_LEDGER.md`. `phase5_predict.py` first reproduces the EW0207 reference predictions from the R10 inputs on OAK and writes the comparison to its provenance record. Outputs go to `grl_review_archive/phase5/` with `SHA256SUMS`. `phase5_score.py` refuses to run a second time.

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

## Changes in v3 (Phase 5; directive R16; spec of 2026-10-13)

- **Name and separation:** Phase 5 has its own scripts (`phase5/scripts/`), locks, records and outputs (`grl_review_archive/phase5/`). Phase 4, its files, records, tags and outputs are unchanged.
- **Sites and product rules** (B1; spec §3.1, §3.2; rulings S1–S3), replacing the five Phase 4 sealed pairs, the lockbox role and its history.
- **Eligibility** (B1; spec §3.3): items 1–3 at step 1 from metadata (`eligible` / `eligibility_reason` in the supplement), item 4 at scoring. A failing site is "not testable" or "not scorable" and is never replaced. Cells per window becomes eligibility item 3 (B4).
- **C1:** the primary criterion is the pooled row; the v2.1 per-site label is secondary (B5; spec §3.4, with the manuscript reason).
- **C2** gets block-bootstrap CIs for GBT_ship, P-V1 and their differences from s_lr, per site and pooled, from the same resamples (B5; spec §3.5).
- **Expectation** (B5; spec §3.6) and **prior exposure** (spec §4) restated for Phase 5.
- **Order** (B3; spec §2): step 2 is build and blind pages; the calls stay keyed by Q-code and the join to pairs is done inside `phase5_score.py` (Phase 4 deviation note 04).
- **Locks** (B6): the sign-off `R16/SIGNOFF_phase5_prereg_v3.md`; the five Phase 5 scripts and the five imported Phase 4 scripts; the supplement `manifest/pairs_phase5_sealed.parquet` with 64 + 6 columns; the run record of `RECORD_CONTRACT_phase5.md` in `reports_post_grl_review/PHASE5/`; the acquisition freeze record sha256 `6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1` (tag `phase5-acq-freeze`); the Phase 5 guard extension and allow-list.
- **Unchanged:** the models and artifacts (B2), targets, tiles, interior masks, features, model inputs, views, B_min = 8, the 10-tile minimum, the bootstrap (2,000 resamples, seeds 20261009 and 20261013), the amplitude statistic, the artifact rule (0.04 m floor), the EW0207 control, the figure seed 20261011 and codes, the calls fields, CPU inference, the single-run lock.
- **Rehearsal (R16 Part C):** before signature, the four Phase 5 scripts ran end to end in rehearsal mode on the ACQ-R09 rehearsal outputs copied into a layout named for the real Phase 5 roots, through the same `start`, data locks, allow-list and guard as phase5 mode; the R10 reproduction and the refusal-test suite passed (REPORT_R16).

## Amendment v3.1 (2026-10-13; directive R16c; rulings of Steve, 2026-10-13)

- **Why:** ACQ-R10 stopped because two development QA steps, the vertical-datum check and the ship line-to-line check, were human judgements.
- **What changed:** the vertical-datum check is automated in the acquisition chain (B1 item 1, the blindness note; T_G = 0.200712 from development pairs only); the acquisition freeze is v2 (`freeze_record_phase5_v2.json` `f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438`, tag `phase5-acq-freeze-v2`, entry script `83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47`); in `phase5_common.py` only the freeze, entry-script, pre-registration and sign-off constants changed (B6).
- **What did not change:** the criteria, thresholds, seeds, sites, models and product rules, the other four Phase 5 scripts and every Phase 4 file.
- **Line check (fallback approved in advance):** ship line-to-line inconsistency is not checked, because the LR grid lattices it needs are not on disk for 36 of the 38 accepted development pairs (ACQ-R11b).
- **Blindness:** no sealed file was opened and no Phase 5 swath was fetched before v3.1. v3.1 supersedes v3 (`3df236e008466919a2b5acf94fb258a6f42abe6235a1ee1b77c0785ca8939d58`, signed, tag `phase5-prereg-v3`), which stays as it is.
