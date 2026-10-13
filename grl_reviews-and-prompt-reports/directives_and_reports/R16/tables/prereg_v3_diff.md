# PREREG v2.1 -> v3: section-by-section diff (R16 Part E1)

v2.1 `R12c/PREREG_phase4_v2_1.md` sha256 `d6af268bc8211fe9b4ae17439a47e6fdcbe7e5949d8ab387a7be080b2c748334`; v3 `R16/PREREG_phase5_v3.md` sha256 `3df236e008466919a2b5acf94fb258a6f42abe6235a1ee1b77c0785ca8939d58`.

| v2.1 section | v3 section | change | lines −/+ |
|---|---|---|---|
| (header) | (header) | replaced (v3 header; v2.1 rulings paragraph and fallback status carried verbatim) | −3 / +5 |
| ## B1. Sealed units | ## B1. Sites (spec §3.1–§3.3) | replaced (spec §3.1–§3.3, rulings S1–S3, four notes) | −5 / +34 |
| ## Prior exposure | ## Prior exposure | replaced (spec §4) | −4 / +3 |
| ## B2. Frozen candidates (artifact paths and sha256) | ## B2. Frozen candidates (artifact paths and sha256) | verbatim + 1 sentence (spec §2) | −0 / +2 |
| ## B3. Order of operations at Phase 4 (each step a separate directive or job; nothing out of order) | ## B3. Order of operations at Phase 5 (each step a separate directive or job; nothing out of order) | replaced (spec §2 order; deviation note 04 join) | −13 / +17 |
| ## B4. Scoring rules | ## B4. Scoring rules | verbatim except rule 0 (+ not testable) and the cells-per-window bullet | −2 / +2 |
| ## B5. Criteria (fixed now) | ## B5. Criteria (fixed now) | replaced (spec §3.4–§3.6; C3 dropped) | −10 / +21 |
| ## B6. Provenance requirements at Phase 4 | ## B6. Provenance requirements at Phase 5 | replaced (Phase 5 locks; B6 tables; freeze sha256) | −2 / +43 |
| ## Changes from v1 | ## Changes from v1 | verbatim | −0 / +0 |
| ## Revision v2.1 (2026-10-12; directive R12c; deviation note 03) | ## Revision v2.1 (2026-10-12; directive R12c; deviation note 03) | verbatim | −0 / +1 |
| — | ## Changes in v3 (Phase 5; directive R16; spec of 2026-10-13) | added | −0 / +12 |

## Full diff per section

### (header)
```diff
--- 
+++ 
@@ -1 +1 @@
-# PREREG_phase4_v2 — Phase 4 pre-registration, version 2 (frozen; tag `phase4-prereg-v2`)
+# PREREG_phase5_v3 — Phase 5 pre-registration, version 3 (a second pre-registered blind test)
@@ -3 +3 @@
-> **Version 2, written 2026-10-10 by directives R10 and R10b and R11. It supersedes `PREREG_phase4_v1.md` (sha256 `5c636f3d…4c1f`, tag `phase4-prereg`), which was never executed. No sealed file had been opened, listed or computed on when v2 was frozen (guard self-tests, REPORT_R09, REPORT_R10, REPORT_R10b, REPORT_R11). Changes from v1 are listed in the final section.**
+> **Version 3, rendered 2026-10-13 by directive R16 from `R12c/PREREG_phase4_v2_1.md` (sha256 `d6af268b…8334`, tag `phase4-prereg-v2.1`) with exactly the changes of `phase_5/PHASE5_design_spec_2026-10-13.md` (ruled by Steve on 2026-10-13) and the directive's notes. Phase 4 stays closed and is reported as it is: C1 not confirmed (0 of 2), pooled −0.20 [−0.34, −0.02] (REPORT_R13). "Phase 5" keeps its paths, locks, records and outputs separate from Phase 4. When v3 was rendered no Phase 5 candidate file had been opened by the CNN repo and the acquisition step had not run in phase5 mode; the only candidate information that exists is the ACQ-R08 / ACQ-R08b metadata (Prior exposure). Changes from v2.1 are listed in "Changes in v3".**
@@ -5 +5,3 @@
-**Written:** 2026-10-09 by directive R09 Part B, from verified records; not executed in R09. **Rulings in force (Steve, 2026-10-09):** CNN development is closed for this paper; the CNN enters Phase 4 as one arm of a model-complexity comparison; the frozen candidates are s_lr, GBT_ship and P-V1 (seed-averaged); development scoring uses interior tiles at D₁ on the primary ten real-relief units (lu14 shown on all tiles beside the interior rows and kept out of every median); Phase 4's main question is whether within-site ranking of unresolved relief transfers to the sealed units, whose ship cells (103–151 m) are coarser than anything in development. Candidate comparisons are secondary and descriptive.
+**Why (spec §1):** Phase 4 did not confirm C1. Both scorable confirmatory sites had about 2 ship cells per tile, outside the development range, and R14 showed that coarse sampling alone does not explain the failure. Phase 5 tests the same claim, with the same frozen models, on new sites inside the development sampling range.
+
+**Carried from v2.1 (rulings of 2026-10-09, in force):** **Written:** 2026-10-09 by directive R09 Part B, from verified records; not executed in R09. **Rulings in force (Steve, 2026-10-09):** CNN development is closed for this paper; the CNN enters Phase 4 as one arm of a model-complexity comparison; the frozen candidates are s_lr, GBT_ship and P-V1 (seed-averaged); development scoring uses interior tiles at D₁ on the primary ten real-relief units (lu14 shown on all tiles beside the interior rows and kept out of every median); Phase 4's main question is whether within-site ranking of unresolved relief transfers to the sealed units, whose ship cells (103–151 m) are coarser than anything in development. Candidate comparisons are secondary and descriptive.
```

### ## B1. Sites (spec §3.1–§3.3)
```diff
--- 
+++ 
@@ -3,4 +3,10 @@
-| set | pairs | role |
-|---|---|---|
-| confirmatory | KN182L03__MGDS_30193 (nu01a), RC2511__MGDS_32239 (nu01b), SUM1004__MGDS_33090 (nu03) | primary |
-| lockbox | AT42-03__MGDS_32007, TN159__MGDS_21981 | supporting only |
+| unit | site | high resolution | ship partner | pair id |
+|---|---|---|---|---|
+| p5u01 | Galápagos Spreading Center | MGDS:18210, 1 m product only (MGDS:31811 not used) | TN188, Simrad EM300, 2005 | `TN188__MGDS_18210` |
+| p5u02 | Escanaba Trough | MGDS:30368 | Escanaba, Simrad EM300, 1998 | `Escanaba__MGDS_30368` |
+| p5u03 | Monterey Canyon | MGDS:24756 | Monterey, Simrad EM300, 1998 | `Monterey__MGDS_24756` |
+| p5u04 | Mid-Atlantic Ridge 16.5°N | MGDS:22436 | KN161L04, SeaBeam 2112, 2000 | `KN161L04__MGDS_22436` |
+| p5u05 | La Jolla Canyon | MGDS:21415 | MV1209, Kongsberg EM122, 2012 | `MV1209__MGDS_21415` |
+| p5u06 | Bermuda | MGDS:31675 | KN173L02, SeaBeam 2112, 2003 | `KN173L02__MGDS_31675` |
+| p5u07 | Chile margin | MGDS:31860 | MV1003, Simrad EM122, 2010 | `MV1003__MGDS_31860` |
+| p5u08 | Dorado Outcrop | MGDS:31954 | AT26-24, Kongsberg EM122, 2014 | `AT26-24__MGDS_31954` |
@@ -8 +14 @@
-**Lockbox history (R09 §B1, CONFIRMED with file):** the submitted manuscript's '10 never-trained units' table (S5_external_test; REPORT_57 maps it to dir55/external_test/results.json) scores AT42-03__MGDS_32007 (lu03, 237 patches, within-pair R2 0.654) and TN159__MGDS_21981 (lu17, 113 patches, 0.202) with the frozen all-native model labelled 'het_production_v1_repro (all-native)' in the results file, i.e. the §C reproduction of het_production_v1 that REPORT_55 §C labels het_retrain_nonrepro (the production weights were never persisted; REPORT_55 §A). Earlier, the first het training (report 22, 2026-06-30) used TN159 as a test pair and AT42-03 as a validation pair, and report 26 reports calibration on both; REPORT_55 then declared them the 'never-trained reserve' and asserted them absent from every train / held set of the expanded retrain. Source: `/oak/stanford/groups/hilley/auv_ship_colocated_bathy/analysis_archive_2026-09-28/dir55/external_test/results.json` (sha256 `7df91a6387d966de160581d668f96b235927ef0f59e3899a44ae28aff35e10a9`); reports 22, 26, 55, 57 (sha256 in `scores/lockbox_history.json`). Lockbox results are therefore reported as supporting and labelled "scored by the submitted pipeline's het_production_v1_repro model (2026-07-08) and used as validation / test splits of the first het training (2026-06-30)".
+Each site is one unit and one pair: one high-resolution mosaic, one ship cruise. The sites come from ACQ-R08b; no other site may be added and no site is replaced. Pair ids follow `RECORD_CONTRACT_phase5.md` (`<ship cruise>__<hr id with ':' → '_'>`); units follow the acquisition site list (`inputs/phase5_sites.csv`) order. The Phase 4 sealed pairs are not used, and no Phase 4 sealed pair joins the development set.
@@ -9,0 +16,23 @@
+**Mechanical product rules (applied by script at step 1, without viewing):**
+
+1. **Resolution group:** the finest group of grids with cell size 0.985 to 4.99 m whose union covers at least 80 % of the largest such group (the ACQ-R08b rule).
+2. **Repeat surveys** (overlapping grids of the same seafloor, for example Monterey): for each overlap, keep the grid whose acquisition date is closest to the ship cruise's year; ties go to the larger valid area. Dates come from file names or metadata. Non-overlapping grids are mosaicked. Ruling S2: two grids overlap when their validity masks share ≥ 50 % of the smaller grid's valid area; below that they are mosaicked with the rule order as priority; an undated grid counts as a tie.
+3. **Dispersed grids (ruling S1):** after rules 1 and 2 the kept grids go to the unchanged development mosaic logic (`grid_lr.grid_region`, `harmonize_new.hr_mosaic`).
+4. **Ship side:** the named cruise's raw swath, gridded by the frozen development chain with contract v2.1 products; harmonized, co-registered and k-swept exactly as the development pairs were. Ruling S3: the current `grid_lr` region rule is used.
+
+**Eligibility (fixed now; decided at step 1 from metadata only).** A site enters the C1 set only if all hold:
+
+1. co-registration passes the development QA thresholds (ACQ-R03: offset ≤ 30 m, overlap ≥ 1 km², ≥ 1 256-px tile with ≥ 50 % joint-valid cells);
+2. the frozen sweep recovers a k;
+3. the gridded ship spacing gives 3.88 to 51.2 ship cells across a 256-px window (256 × unrounded HR pixel / lr_native), and the HR spacing is 0.985 to 4.99 m;
+4. at scoring, the B4 view rules give a scorable view (10 tiles in at least 8 blocks).
+
+A site failing 1 to 3 is **not testable** at step 1, with its reason (supplement column `eligibility_reason`); it has no build, figure page, prediction or score. A site failing 4 is **not scorable** at step 5. Then the artifact rule applies at the blind call. No site is replaced or added.
+
+**Notes (v3):**
+
+- **The 3.88 lower bound and FK151121.** The bound is FK151121's development value, 256 × 1.000274 / 65.99 = 3.8804 cells. The frozen Phase 5 chain, with today's region rule, gives FK151121 lr_native 66.09 m and 3.8746 cells, which would be "not testable". The bound is kept as written (ruling ACQ-R09 H-1, Steve, 2026-10-13). Nothing changes unless a site lands within about 0.3 % of 3.88; the lowest ACQ-R08b estimate is 4.32 cells (Mid-Atlantic Ridge).
+- **The gridding region rule.** Step 1 grids the ship swath with the current `grid_lr` rule (`grid_region`: the ship grid on the HR patch cluster; ruling S3). FK151121's development grid predates that rule (whole footprint, commit `ce0ebb3`). ACQ-R09 reproduced NA080 and KM0923 bit-identically under the current rule, and FK151121 bit-identically only under its recorded region in a separate harness.
+- **`vertical_offset` fallback.** The contract-v2.1 `vertical_offset` flag uses its 0.5 m fallback threshold for every Phase 5 site, because no CNN s_lr exists for them. This is the frozen builder's behaviour, as it was for the Phase 4 sealed pairs.
+- **No person views candidate data at step 1.** Co-registration passes or fails by the automated development thresholds only; no QC figure is produced, and the step-1 report carries only status, eligibility fields, spacings, cells across, k, product status and usability, file counts and record sha256 values (ACQ-R09 §2.10). The only viewing in Phase 5 is Steve's blind call on the Q-coded pages of step 2 (B3 step 3).
+
```

### ## Prior exposure
```diff
--- 
+++ 
@@ -3,4 +3,3 @@
-Before this version was frozen, Steve had seen the sealed pairs' bathymetry three times, and no score:
-- **The original co-registration sign-off,** which covered the lockbox pairs.
-- **The ACQ-R03 co-registration review** of nu01a and nu03: hillshades before and after the shift, contours and the search surface (ruling R6, 2026-10-01).
-- **A re-view on 2026-10-10** of the harmonized AUV and ship bathymetry of the sealed pairs (depth and hillshade). He noted that nu03 shows very low relief, with survey track lines as the only visible structure. No fine-band, difference or roughness layer was viewed, and no rule was changed as a result.
+- Steve has seen none of the eight sites' bathymetry. That includes the June QC figure for Bermuda and the La Jolla Zephyr survey (stated 2026-10-13).
+- Everyone has seen the ACQ-R08 and ACQ-R08b metadata (listed in REPORT_ACQ-R08b §5). That includes one median depth per site, rounded to 100 m.
+- The executor that ran ACQ-R08b saw only those numbers.
```

### ## B2. Frozen candidates (artifact paths and sha256)
```diff
--- 
+++ 
@@ -8,0 +9,2 @@
+B2 applies unchanged at Phase 5 (spec §2): no retraining, refitting or tuning; the development set is unchanged.
+
```

### ## B3. Order of operations at Phase 5 (each step a separate directive or job; nothing out of order)
```diff
--- 
+++ 
@@ -3,6 +3,6 @@
-1. **Acquisition repo:** The acquisition step is the frozen `src/phase4/sealed_step1.py --mode phase4` (tag `phase4-acq-freeze`; `freeze_record.json` sha256 `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb`; script sha256 `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1`), rehearsed in ACQ-R06 and ACQ-R06b.
-   - Confirmatory pairs get `hr_native_res_m`, `lr_native_res_m` and `max_recoverable_k` from the ACQ-R02 chain: the C2a rule, with depth = median of the harmonized `lr.tif`, and the C2c sweep `stage_c2c_recalibrate.sweep`, where k is the largest k with the verdict `recoverable_signal`.
-   - The lockbox pairs keep their June values (0.989 / 19.7 m and 0.991 / 41.0 m; k = 4 from the June sweep record). Their swath is fetched from the June plan; if the fetch is incomplete, their products are unavailable.
-   - `ship_products_usable` is true if and only if the products were built and carry no contract-v2.1 flag.
-   - A data-tied builder failure makes the products unavailable. A pair with no k gets nothing built. Any other error stops the step before the supplement is written.
-   - The step writes `manifest/pairs_phase4_sealed.parquet` once, with the per-k verdicts and the source of k.
+1. **Acquisition repo (ACQ-R10):** the frozen `src/phase5/sealed_step1_p5.py --mode phase5` (tag `phase5-acq-freeze`; code commit `4d93c8a`; freeze record `reports_post_grl_review/PHASE5/freeze_record_phase5.json` sha256 `6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1`; script sha256 `9f143e0c3f000163d9da48001cbd7578805c8c1042141f9994a972df73b1b582`), rehearsed in ACQ-R09.
+   - It runs locks 1 to 4: the sign-off holding this document's sha256, the freeze record, exactly the eight listed sites, and no second completed run.
+   - It fetches the named cruise's raw swath and grids it with the frozen development chain. It selects and mosaics the HR grids by the B1 product rules, then harmonizes and co-registers. It runs the frozen C2c sweep (k = the largest k with verdict `recoverable_signal`) and builds the contract-v2.1 products.
+   - `ship_products_usable` is true if and only if the products were built and carry no contract-v2.1 flag (frozen Phase 4 rule C1).
+   - It decides eligibility items 1 to 3 from metadata only.
+   - It writes `manifest/pairs_phase5_sealed.parquet` once: the 64 columns of the Phase 4 supplement, then `coreg_pass, k_recoverable, cells_across, hr_res_in_range, eligible, eligibility_reason`. It also writes the run record `reports_post_grl_review/PHASE5/step1_run_phase5_<UTC>.json` with the keys of `RECORD_CONTRACT_phase5.md`. No record key holds a depth statistic, amplitude or residual.
@@ -10,3 +10,7 @@
-   Its sha256 is recorded in `CHANGE_LEDGER.md`. No context grid is built.
-2. **CNN repo:** `phase4_build.py --mode phase4`. It also writes each pair's amplitude statistic (B4). `max_recoverable_k` is not set here: it comes from the supplement of step 1 (the acquisition repo owns the rule). A pair with no recoverable k is not built. It verifies the acquisition run record (complete; cites the freeze record and the supplement's sha256) before building. A pair's product features are used only if its supplement row has `ship_products_usable = true`; otherwise they are masked as for a development pair whose products are not used. **No prediction or score yet.**
-3. **Blind artifact judgement:** `phase4_figures.py --mode phase4` produces figure set A for the 5 sealed pairs, pages shuffled with seed 20261011 and coded Q01 to Q05, key in a separate file. Steve judges each page against the R08 §3.1 criteria and fills one row per page:
+   The supplement's sha256 is recorded in `CHANGE_LEDGER.md`.
+2. **CNN repo, build and blind pages:**
+   - `phase5_build.py --mode phase5` verifies the acquisition run record and builds each eligible site. It writes each site's amplitude statistic (B4). `max_recoverable_k` comes from the supplement; the acquisition repo owns the rule. A site's product features are used only if its supplement row has `ship_products_usable = true`; otherwise they are masked.
+   - A not-testable site is not opened; only its status is written.
+   - `phase5_figures.py --mode phase5` then produces figure set A for the eligible sites. Pages are shuffled with seed 20261011 and coded Q01 onward; the key is in a separate file.
+   - **No prediction or score yet.**
+3. **Blind artifact judgement:** Steve judges each page against the R08 §3.1 criteria and fills one row per page:
@@ -16 +20 @@
-   | `code` | Q01 to Q05 |
+   | `code` | Q01 onward, one per page |
@@ -23,3 +27,3 @@
-   The calls file is committed and its sha256 recorded in `CHANGE_LEDGER.md` **before** any prediction is computed.
-4. **Predict:** `phase4_predict.py --mode phase4`, on CPU. The script first reproduces the EW0207 reference predictions (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and refuses to continue otherwise.
-5. **Score once:** `phase4_score.py --mode phase4`. Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.
+   The calls file stays keyed by Q-code. It is committed and its sha256 recorded in `CHANGE_LEDGER.md` **before** any prediction is computed. The Q-code to pair join is part of this procedure: `phase5_score.py` joins the committed file to the pairs through the figure key, asserting every value unchanged, and writes no separate joined file (Phase 4 deviation note 04).
+4. **Predict:** `phase5_predict.py --mode phase5`, on CPU. The script first reproduces the EW0207 reference predictions (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and refuses to continue otherwise. Not-testable sites are not predicted.
+5. **Score once:** `phase5_score.py --mode phase5`. It assembles its own scoring inputs (pinned, R16b): it reads only the build and prediction outputs under `grl_review_archive/phase5/` (per eligible site the build statistics, tiles_v1.1, the D₁ interior table, and the predictions, whose sha256 must equal the prediction record's), and it refuses any external inputs table or statistics file. Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.
```

### ## B4. Scoring rules
```diff
--- 
+++ 
@@ -6 +6 @@
-  0. A pair with no recoverable k has no fine band. It is **not scorable (no k)**, has no figure page, and is outside C1.
+  0. A site that failed eligibility at step 1 (B1 items 1–3) is **not testable**, with its reason; it has no build, figure page, prediction or score, and is outside C1. A pair with no recoverable k has no fine band. It is **not scorable (no k)**, has no figure page, and is outside C1 (at Phase 5 this cannot occur for an eligible site: k is eligibility item 2).
@@ -11 +11 @@
-- **Cells per window:** `256 × hr_res / lr_native` is reported for each sealed pair beside the development minimum of 3.88. A pair below it is flagged "window thinner than any development pair". No rule is attached. From the ACQ-R02 records, the three confirmatory pairs are expected at about 2.5, 2.1 and 1.7, all below the development minimum.
+- **Cells per window:** `256 × hr_res / lr_native` is eligibility item 3 at step 1 (3.88 to 51.2; B1). It is reported for each site, and the score output keeps v2.1's flag "window thinner than any development pair" (below 3.88), with no rule attached. From the ACQ-R08b metadata the eight sites are expected at about 4.3 to 24.5 cells.
```

### ## B5. Criteria (fixed now)
```diff
--- 
+++ 
@@ -3,5 +3,5 @@
-- **C1, primary: ranking transfers.** The **C1 set** is the confirmatory units that are scorable, not artifact-dominated and not low-amplitude artifact-affected. Its size is stated.
-  - **Confirmed** if s_lr's block-bootstrap CI lower bound is above zero on every unit of the C1 set.
-  - **Partially confirmed** if on some but not all.
-  - **Not confirmed** if on none.
-  - **Not testable** if the C1 set is empty.
+- **C1 set:** the eligible sites (B1 items 1–3) that are scorable (item 4), not artifact-dominated and not low-amplitude artifact-affected. Its size is stated.
+- **C1, primary: pooled ranking transfers (spec §3.4).** The mean within-site s_lr ρ over the C1-set sites, with a 95 % CI from a block bootstrap stratified by unit (non-empty blocks resampled within each site, 2,000 resamples, one `numpy.random.default_rng(20261013)`, sites in ascending pair-id order; a resample with any undefined site ρ is dropped and counted, > 5 % → not estimable).
+  - **Confirmed** if the CI lower bound is above zero.
+  - **Not confirmed** otherwise.
+  - **Not testable** if fewer than 3 sites are in the C1 set.
@@ -9 +9,6 @@
-  The same reading is also stated with the low-amplitude artifact-affected units included.
+  The pooled row is also reported with the low-amplitude artifact-affected sites included. That row does not change the label.
+- **C1-per-site, secondary.** Each site's s_lr CI as in v2.1 (B4), with the count of sites whose lower bound is above zero and the v2.1 all-sites label, reported beside the primary:
+  - **confirmed** if the lower bound is above zero on every site of the C1 set;
+  - **partially confirmed** if on some;
+  - **not confirmed** if on none;
+  - **not testable** if the set is empty.
@@ -11,4 +16,10 @@
-  Expectation from development (REPORT_R10 §6.3; each pair as its own unit, held-out predictions): on the pairs of the ten real-relief units in the C1 set, s_lr met the C1 condition on 9 of 14 at D₁ and 8 of 12 at the 125 m distance. Pair ρ at D₁ ranged from -0.09 to +0.82, median +0.41. Without AR26, which the 2026-10-09 ruling keeps out of development medians: 8 of 13, range -0.09 to +0.63, median +0.39. If three independent units each met the condition at the 9-of-14 rate, all three would do so with probability 0.27, some but not all with 0.69, and none with 0.05. If the C1 set has two units, both pass with probability 0.41, one with 0.46, and neither with 0.13, at the same rate. At sealed-like coarseness (ship grid block-averaged to about 125 m, target unchanged; REPORT_R11 §5), s_lr met the condition on 9 of 14 pairs, pooled mean +0.37 [+0.29, +0.41]. Report each sealed unit's ρ and CI against these.
-- **C1 summary, secondary: pooled ranking.** The mean of s_lr's ρ over the units of the C1 set, with a 95 % CI from a block bootstrap stratified by unit (non-empty blocks resampled within each unit, 2,000 resamples, one `numpy.random.default_rng(20261013)`, units in ascending pair-id order). It is reported with a statement of whether its lower bound is above zero, and again with the low-amplitude artifact-affected units included. It does not change the C1 label. Expectation from development (REPORT_R10b §4.4): over all triples of the 14 development pairs at D₁, the lower bound was above zero in 346 of 364; the pooled mean ranged from +0.08 to +0.68, median +0.39. At the 125 m interior distance, the lower bound was above zero in 199 of 220 triples.
-- **C2, secondary and descriptive: model complexity.** Per sealed unit, report GBT_ship − s_lr, P-V1 − s_lr and P-V1 − GBT_ship. One sentence per comparison stating its direction across units. No superiority claim is made: with at most 3 units, even 3 of 3 gives a one-sided sign test of p = 0.125. State this in the result.
-- **C3, supporting:** C1, the C1 summary and C2 on the lockbox pairs, reported separately with their history label.
+  It is also stated with the low-amplitude artifact-affected sites included.
+- **Why the change, for the manuscript (Steve's condition):** with n sites and a per-site pass rate of 9 in 14 at development, "every site passes" occurs with probability about 0.64^n even if the method works as in development (about 3 % at n = 8). So the v2.1 per-site label cannot confirm transfer with more than two or three sites. The pooled mean estimates average transfer directly and has adequate power. The change is made after Phase 4 was known. Under it, Phase 4 would still read "not confirmed" (pooled −0.20 [−0.34, −0.02]), so the change does not alter that result. Both readings are reported for Phase 5.
+- **Expectation (spec §3.6):** at development, pooled s_lr is +0.392 [+0.317, +0.430] over 14 pairs, and +0.35 at 1.7 cells per tile (R14). Phase 4 was −0.20 on two sites. Phase 5 tests which of these the new sites resemble. For the secondary reading: development s_lr met the per-site condition on 9 of 14 pairs at D₁ (REPORT_R10 §6.3), pair ρ −0.09 to +0.82, median +0.41.
+- **C2, descriptive: model complexity (spec §3.5).**
+  - GBT_ship and P-V1 against s_lr, per site and pooled over the C1 set. Each candidate's ρ and each difference from s_lr (GBT_ship − s_lr, P-V1 − s_lr) gets a block-bootstrap 95 % CI.
+  - The CIs use the same resamples as s_lr: per site, the block draws of `default_rng(20261009)`; pooled, the draws of `default_rng(20261013)`, sites in ascending pair-id order. The same undefined-resample rule applies.
+  - `phase5_score.py` (pinned in B6) computes them. The frozen Phase 4 score gives no CI for GBT_ship or P-V1.
+  - P-V1 − GBT_ship is reported as a point value.
+  - One sentence per comparison states its direction across sites. No superiority claim is made.
+- C3 (the lockbox role) has no counterpart at Phase 5.
```

### ## B6. Provenance requirements at Phase 5
```diff
--- 
+++ 
@@ -3 +3,19 @@
-Every Phase 4 script runs in `phase4` mode only if `SIGNOFF_phase4_prereg_v2.md` exists and holds this document's sha256. Before running, it verifies that sha256, the candidate artifacts (B2), the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`) and the five Phase 4 scripts against this document:
+Every Phase 5 script runs in `phase5` mode only if `R16/SIGNOFF_phase5_prereg_v3.md` exists and holds this document's sha256. Before running, it verifies the following against this document:
+
+- that sha256;
+- the candidate artifacts (B2);
+- the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`);
+- the five Phase 5 scripts;
+- the five Phase 4 scripts, which the Phase 5 scripts import unchanged for every frozen function (targets, tiles, interior masks, features, model inputs, the three models, the EW0207 control, the artifact rule, the single-run lock and the bootstrap seeds).
+
+It also requires the acquisition freeze record sha256 `6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1` to be stated here.
+
+| script | sha256 |
+|---|---|
+| `phase5_common.py` | `767dec60213ac9bd9a8ec05c82a4ec95656fdfc3aeb2cbd7c6f5f47074802ea6` |
+| `phase5_build.py` | `973b775dda073fa3b4def8073fc7ce71b22a64df20d84a30e4c1b2b892d45633` |
+| `phase5_figures.py` | `47921a51699d370f90f7082590f03f6a69477d83c25978c154b3312535d811a0` |
+| `phase5_predict.py` | `96797b85309f360b971c15f49a4c02a3fdf4ce7d2e276af33c31e621ee8c6429` |
+| `phase5_score.py` | `7eb7c1cc3fc0acbfebf9b4f2c517c145b035699612f6e8ee31865c263787d2a5` |
+
+Imported unchanged (v2.1 B6, tag `phase4-prereg-v2.1`):
@@ -13 +31 @@
-In `phase4` mode the five sealed manifest rows are read only from `manifest/pairs_phase4_sealed.parquet` (beside manifest v2.1), which must hold exactly the five pre-registered pair ids with the v2.1 columns and whose sha256 must be recorded in `CHANGE_LEDGER.md`; development rows come from the frozen v2.1 file. After the supplement lock, every script also verifies the acquisition run record: the acquisition repo's `reports_post_grl_review/ACQ-R06b/step1_run_phase4_*.json` must have status complete in `phase4` mode, cite the freeze record sha256 `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb` (the `freeze_record.json` on disk must hash to it and list `src/phase4/sealed_step1.py` at `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1`) and cite the supplement's sha256; otherwise it refuses. `phase4_predict.py` first reproduces the EW0207 reference predictions from the R10 inputs on OAK (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and writes the comparison to its provenance record. Outputs go to `grl_review_archive/phase4/` with `SHA256SUMS`. `phase4_score.py` refuses to run a second time.
+**Supplement lock.** In `phase5` mode the eight site rows are read only from `manifest/pairs_phase5_sealed.parquet`, beside manifest v2.1. The file must meet all of these:
@@ -14,0 +33,23 @@
+- it holds exactly the eight pre-registered pair ids;
+- its first 64 columns are the Phase 4 supplement's names, order and arrow types (`manifest/pairs_phase4_sealed.parquet`, sha256 `6968e045…9206`);
+- they are followed by exactly the six eligibility columns;
+- `eligible` agrees with B1 items 1–3 on every row;
+- its sha256 is recorded in `CHANGE_LEDGER.md`.
+
+Development rows come from the frozen v2.1 file. The eligible rows are the admitted sites; a row with `eligible = False` is "not testable (<eligibility_reason>)".
+
+**Acquisition run-record lock.** After the supplement lock, every script verifies the acquisition run record. The acquisition repo's `reports_post_grl_review/PHASE5/step1_run_phase5_*.json` must:
+
+- have status complete in `phase5` mode;
+- cite the freeze record sha256 `6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1` in `locks.lock2_freeze_record`;
+- cite the supplement's sha256.
+
+The `freeze_record_phase5.json` on disk must hash to that value and list `src/phase5/sealed_step1_p5.py` at `9f143e0c3f000163d9da48001cbd7578805c8c1042141f9994a972df73b1b582`. Otherwise the script refuses.
+
+**Guard.**
+- The R09 guard of Phase 4 is installed unchanged and self-tests 702 of 702 probes.
+- The Phase 5 extension seals the roots `sealed_candidates_v3` and `auv_ship_colocated_bathy/phase5`, every site's pair id, ship cruise and HR id, and MGDS:31811. It self-tests on the real Phase 5 names before any lock.
+- After every lock, a path is admitted only if it contains an eligible site's full pair id. Every other path holding a Phase 5 token stays refused, and the Phase 4 tokens are never admitted.
+
+**Calls and single run.** `phase5_predict.py` and `phase5_score.py` require the committed code-keyed calls file, with its sha256 in `CHANGE_LEDGER.md`. `phase5_predict.py` first reproduces the EW0207 reference predictions from the R10 inputs on OAK and writes the comparison to its provenance record. Outputs go to `grl_review_archive/phase5/` with `SHA256SUMS`. `phase5_score.py` refuses to run a second time.
+
```

### ## Changes from v1
```diff
```

### ## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)
```diff
--- 
+++ 
@@ -8,0 +9 @@
+
```

### ## Changes in v3 (Phase 5; directive R16; spec of 2026-10-13)
```diff
--- 
+++ 
@@ -0,0 +1,12 @@
+
+
+- **Name and separation:** Phase 5 has its own scripts (`phase5/scripts/`), locks, records and outputs (`grl_review_archive/phase5/`). Phase 4, its files, records, tags and outputs are unchanged.
+- **Sites and product rules** (B1; spec §3.1, §3.2; rulings S1–S3), replacing the five Phase 4 sealed pairs, the lockbox role and its history.
+- **Eligibility** (B1; spec §3.3): items 1–3 at step 1 from metadata (`eligible` / `eligibility_reason` in the supplement), item 4 at scoring. A failing site is "not testable" or "not scorable" and is never replaced. Cells per window becomes eligibility item 3 (B4).
+- **C1:** the primary criterion is the pooled row; the v2.1 per-site label is secondary (B5; spec §3.4, with the manuscript reason).
+- **C2** gets block-bootstrap CIs for GBT_ship, P-V1 and their differences from s_lr, per site and pooled, from the same resamples (B5; spec §3.5).
+- **Expectation** (B5; spec §3.6) and **prior exposure** (spec §4) restated for Phase 5.
+- **Order** (B3; spec §2): step 2 is build and blind pages; the calls stay keyed by Q-code and the join to pairs is done inside `phase5_score.py` (Phase 4 deviation note 04).
+- **Locks** (B6): the sign-off `R16/SIGNOFF_phase5_prereg_v3.md`; the five Phase 5 scripts and the five imported Phase 4 scripts; the supplement `manifest/pairs_phase5_sealed.parquet` with 64 + 6 columns; the run record of `RECORD_CONTRACT_phase5.md` in `reports_post_grl_review/PHASE5/`; the acquisition freeze record sha256 `6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1` (tag `phase5-acq-freeze`); the Phase 5 guard extension and allow-list.
+- **Unchanged:** the models and artifacts (B2), targets, tiles, interior masks, features, model inputs, views, B_min = 8, the 10-tile minimum, the bootstrap (2,000 resamples, seeds 20261009 and 20261013), the amplitude statistic, the artifact rule (0.04 m floor), the EW0207 control, the figure seed 20261011 and codes, the calls fields, CPU inference, the single-run lock.
+- **Rehearsal (R16 Part C):** before signature, the four Phase 5 scripts ran end to end in rehearsal mode on the ACQ-R09 rehearsal outputs copied into a layout named for the real Phase 5 roots, through the same `start`, data locks, allow-list and guard as phase5 mode; the R10 reproduction and the refusal-test suite passed (REPORT_R16).
```

