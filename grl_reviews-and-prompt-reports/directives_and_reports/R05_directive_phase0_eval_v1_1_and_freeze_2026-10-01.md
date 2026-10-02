# DIRECTIVE R05 (Phase 0 close, CNN repo): rulings on R04's HOLDs, eval_v1.1, bar_v1.1, and the Phase 0 freeze

**Date:** 2026-10-01
**Author:** steering instance, for Claude Code on Sherlock (CNN repo)
**Predecessor:** `directives_and_reports/R04/REPORT_R04_2026-10-01.md` (R04 commit `8b1d52db2ed0de661c606a561fbafa61ce6cd6a6`, tag withheld)
**Inputs:**
- manifest v2.1, acquisition repo `manifest/pairs_v2_1.parquet`, sha256 `f57f3d8f98b320b136549de300ccac58787129af658a1c62296bab16b6c2ac35`;
- contract v2.1, sha256 `0b4aadaf61fb04213ae18771d2a58532f39e9d49b5152ab64738d188a7e18fcb`;
- R04 frozen code and tables as hashed in R04 §8.3.

**Report to:** `directives_and_reports/R05/REPORT_R05_<date>.md`
**Compute:** CPU only. No network training of any kind.
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK. Meters unless marked.

---

## 0. Purpose

R04 built the targets, the evaluation and the bar, and withheld the `phase0-freeze` tag pending three HOLDs. R05 applies
Steve's rulings on those HOLDs, makes two changes to the evaluation, refits the bar under the changed evaluation, and
completes the Phase 0 freeze. After R05, Phase 2 (CNN development) scores every model through the code frozen here.

**Why the evaluation changes (for the record in the report and the ledger):**

1. **Tile scale becomes primary.** A 32-px block is smaller than one LR cell on most units and shorter than the longest
   wavelengths in the fine band (up to `lr_native`). Its RMS therefore carries large sampling noise that no predictor can
   rank. R04 is consistent with this: every model scores higher at tile than at block scale (GBT_ship A_fine 0.550 vs
   0.423; T_fine 0.626 vs 0.428).
2. **Ranking is done within pairs, then summarized per unit.** Pooling a unit's pairs into one ranking mixes
   between-site level into a within-site statistic, and where a unit's pairs have different k it mixes different
   physical bands. R04 shows the symptom on exactly the two mixed-k units: lu15 (k 16 and 8) GBT_noship ρ −0.44 block /
   −0.40 tile, GBT_ship −0.15 block; lu07 (k 2 and 16) GBT_noship −0.01 block.

**Scope discipline.** Do the items below and nothing else. If a result suggests a new analysis, list it in one line under
"Not pursued". Do not run it.

**Challenge clause.** Report any wrong or underspecified instruction in §1 of the report, with evidence. Items that are
Steve's call become HOLDs (at most 3).

## 1. Rulings in force

| HOLD | ruling | action in R05 |
|---|---|---|
| R04 H1 (consistency check below 0.9) | **(a) accept `targets_v1` as built.** R04 §3.3 shows R03's A_fine was dominated by window-edge ringing; the R04 definition is the intended one. R03's §3–§4 results are not re-scored. | Tag the R04 commit (§5). `targets_v1.py` is not changed. |
| R04 H2 (EX1402L2 `ship_xtrack_frac` all NaN) | **Treat that one raster as unavailable for vu02 on the consumer side.** No acquisition rebuild is requested. | Add a consumer-side availability override (§2.3). The pair's other ship products stay usable. |
| R04 H3 (tiles below 50 % joint-valid) | **Apply a 50 % joint-validity floor to tiles**, matching the window rule. | Build `tiles_v1.1` (§2.1). |
| Venere (pu00) | **Removed from the Phase 0 plan.** No unit is pending. The unit-set versioning rule in R04 stays available if a unit is added later. | Record in the ledger. `units-v1` (16 units / 38 pairs) is the Phase 2 unit set. |

## 2. Changes

### 2.1 `tiles_v1.1`

- Keep a tile only if ≥ 50 % of its central 128 × 128 px are joint-valid (the `joint_v1` mask). Blocks inside a dropped
  tile are dropped. The block rule (≥ 75 % joint-valid) is unchanged.
- Implement this as a selection on `tiles_v1.parquet` / `blocks_v1.parquet`. `targets_v1.py` and the fields are not
  changed or recomputed.
- Report removed tiles and blocks per pair and per unit (R04 reported 242 tiles below 50 %, none below 64 px).
- Write `tiles_v1_1.parquet` and `blocks_v1_1.parquet`.

### 2.2 `eval_v1.1.py`

A copy of `eval_v1.py` with only the changes below. Everything not listed is unchanged (bootstrap over units, 2,000
resamples, seed, 95 % percentile CI, units-better count, exact one-sided sign test reported but not gated, the
`assert_units` and sealed-pair refusals, EW0207 as a separate table).

1. **Primary endpoint: A_fine at tile scale**, within-pair Spearman ρ, summarized as below.
2. **Secondary endpoints:** A_fine at block scale; T_fine at tile and block scale; E_1m at block scale (AUPRC for
   E_1m > 0 and Spearman ρ on E_1m).
3. **Within-pair, then unit, then median over units.** For every metric in the metric set (Spearman ρ, log–log R²,
   top-20 % capture, AUPRC):
   - compute the metric **within each pair** (log–log R² demeaned within the pair);
   - the **unit value** is the median over that unit's eligible pairs;
   - the endpoint value is the **median of unit values** over the development units.
   The bootstrap resamples units, using unit values. The units-better count and sign test compare unit values.
4. **Pair eligibility, fixed now:** a pair contributes to an endpoint only if it has **≥ 10 tiles** in `tiles_v1.1`
   with a finite target and a finite prediction. The same pair set is used at tile and block scale. A unit with no
   eligible pair is dropped from that endpoint. List every ineligible pair and any dropped unit.
5. **Detection rule, moved to the pair level:** a pair enters the E_1m metrics only if its block-scale prevalence of
   E_1m > 0 lies within [5 %, 95 %]. The E_1m threshold stays at 1.0 m. List the pairs and units this excludes.
6. **Pooled metrics** are kept as in v1 (pooled over all rows) and labelled secondary.
7. Run `eval_v1.1` self-tests on a synthetic table that includes a multi-pair unit whose pairs differ in level, and
   show that a predictor carrying only the pair level scores near zero under v1.1 but positive under v1.

### 2.3 Consumer-side availability override

- Add an explicit override list to the feature code: `(EX1402L2__MGDS_31073, ship_xtrack_frac) → unavailable`, with
  the reason `all-NaN where count ≥ 1 (R04 H2)`.
- The feature is NaN for that pair, as R04 already produced it. Record the override in the availability report.
- The validator result for this pair is reported as "fails on one raster, overridden by R05 ruling H2".

### 2.4 `bar_v1.1`

Refit with the frozen bar code, changing only:

- the tile and block set (`tiles_v1.1`, `blocks_v1.1`);
- the **inner tuning criterion**, which becomes the eval_v1.1 statistic (within-pair Spearman ρ, median over pairs
  within each inner unit, median over inner units; AUPRC the same way for the E_1m classifier).

Everything else is unchanged: the features, the 48-point grid, nested leave-one-unit-out, the 200-row cap per unit per
fold, the seed, the log targets, the reference rows.

**The Phase 2 bar is GBT_ship on A_fine at tile scale under eval_v1.1.** Report its value and CI, and GBT_ship −
GBT_noship with its CI and unit count.

## 3. Reporting items

1. **The bar under eval_v1.1:** all endpoints, all predictors, in the R04 `BAR_v1.md` table format.
2. **v1 vs v1.1 side by side** for every predictor on A_fine at tile and at block scale: median ρ and CI under each.
3. **Per-pair ρ table** for every multi-pair unit (lu05, lu07, lu12, lu13, lu15), for GBT_ship, GBT_noship, ref_C and
   ref_s_lr at tile scale.
4. **R02 σ_m provenance:** state, from the R02 code and records, whether the σ_m mosaic values on each of the native 7
   units were produced by a model that did **not** train on that unit (leave-one-unit-out) or in-sample. Label
   CONFIRMED or COULD NOT CHECK per unit. Score σ_m under eval_v1.1 as in R04, labelled with that provenance.
5. **EW0207** under eval_v1.1 as a separate one-unit table.

## 4. Preservation

Copy the R05 tables, the bar predictions and the scripts to OAK `grl_review_archive/R05/` with `SHA256SUMS` verified.

## 5. Freeze

1. Tag the R04 commit `8b1d52db2ed0de661c606a561fbafa61ce6cd6a6` as `r04-as-run`.
2. Commit the R05 code and tables. Tag that commit **`phase0-freeze`**.
3. In the report, record:
   - the sha256 of `targets_v1.py` (unchanged from R04), `eval_v1_1.py`, the feature code with the override, the bar
     code, `tiles_v1_1.parquet` and `blocks_v1_1.parquet`;
   - the manifest sha256 and contract version;
   - the unit set (`units-v1`, 16 units / 38 pairs; EW0207 separate);
   - the pair-eligibility rule, the detection rule and the E_1m threshold in force.
4. Ledger rows in `CHANGE_LEDGER.md`:
   - R04 H1 ruling (targets_v1 accepted; R03 A_fine superseded, not re-scored);
   - R04 H2 ruling (vu02 xtrack override);
   - R04 H3 ruling (tile floor; tiles_v1.1);
   - eval_v1.1 (tile primary; within-pair aggregation; pair eligibility; detection rule per pair) with the reasons in §0;
   - bar_v1.1 (refit under the new criterion);
   - Venere removed from the Phase 0 plan; units-v1 is the Phase 2 set.

## 6. Report

- **§1 Bottom line**, at most 6 points: challenge findings; tiles_v1.1 counts; the bar under eval_v1.1 (value, CI,
  GBT_ship − GBT_noship); what changed from v1 (side-by-side, and the multi-pair units); R02 σ_m provenance; freeze.
- **HOLDs:** at most 3.
- **"Not pursued":** one line per item, no analysis.
- Scripts as run under `directives_and_reports/R05/scripts/`.

## 7. Out of scope

- Any network training or GPU use.
- Recomputing the fields or changing `targets_v1.py`.
- New features, or removing features (including the edge and validity features).
- Re-scoring R03.
- Any acquisition-repo change, including rebuilding EX1402L2's products.
- Lockbox or confirmatory access.
- Adoption criteria, architecture proposals, pretraining work, manuscript text.
