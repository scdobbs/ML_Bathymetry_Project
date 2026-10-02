# DIRECTIVE R04 rev 1 (Phase 0, CNN repo): frozen targets, frozen evaluation, the bar, and two checks

**Date:** 2026-10-01
**Supersedes:** `R04_directive_phase0_targets_bar_and_checks_2026-09-29.md`, which was never run. Commit both files; run
only this one.
**Author:** steering instance, for Claude Code on Sherlock (CNN repo)
**Predecessors:** R02, R03 reports (CNN repo); ACQ-R01–R03 reports and `rulings_ACQ-R03.md` (acquisition repo).
**Inputs from acquisition:**
- manifest `manifest/pairs_v2_1.parquet` on OAK, sha256 `f57f3d8f98b320b136549de300ccac58787129af658a1c62296bab16b6c2ac35`;
- `INTERFACE_CONTRACT_v2.1.md`, sha256 `0b4aadaf61fb04213ae18771d2a58532f39e9d49b5152ab64738d188a7e18fcb`.

**Report to:** `directives_and_reports/R04/REPORT_R04_<date>.md`
**Compute:** CPU only. **No model training of any kind.**
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK. Meters unless marked.

---

## 0. Purpose and plan context

| phase | what | status |
|---|---|---|
| 0 | Freeze targets, evaluation and the free-baseline bar | **this directive** |
| 1 | Acquisition: products, corpus expansion, confirmatory set | closed (ACQ-R03); Venere (pu00) pending as v2.1.1 |
| 2 | CNN development on the development units | next |
| 3 | Self-supervised pretraining on ship-only multibeam, tested as a pre-registered ablation | corpus built in the acquisition repo after this directive |
| 4 | Pre-registered freeze and a single confirmatory scoring | — |

R04 builds the measuring instruments that Phases 2–4 use unchanged. Everything built here is **frozen** at the end of
R04. A later change needs a new version and a `CHANGE_LEDGER.md` row.

**Scope discipline.** Do the items below and nothing else. If a result suggests a new analysis, list it in one line
under "Not pursued". Do not run it.

**Challenge clause.** Report any wrong or underspecified instruction in §1 of the report, with evidence. Items that are
Steve's call become HOLDs (at most 3). The acquisition repo's conventions govern the manifest and products; follow them
where this directive misnames something.

### 0.1 Decisions in force

**Data source.** All pair paths, unit membership, `lr_native`, k and designations come from manifest v2.1, opened
read-only.
- Record its sha256 at start and at freeze. Stop if it differs from the hash above.
- Resolve harmonized directories from the manifest's `harmonized_path` (and `harmonized_lr` where present), never from
  `pair_id`.

**Development units (16): every development pair with a recoverable k.**

| unit | pairs |
|---|---|
| lu00 | RR1506__MGDS_29779 |
| lu02 | discol_so242_1 |
| lu04 | AT37-13__MGDS_31199 |
| lu05 | TN399__MGDS_30373, TN399__MGDS_33068, AT42-06__MGDS_5174 |
| lu06 | ccz_so268_1 |
| lu07 | EW9801__MGDS_31425, KM0923__MGDS_31429 |
| lu08 | FK181031__MGDS_24618 |
| lu09 | tag_m127 |
| lu10 | NA090__MGDS_31212 |
| lu11 | NA076__MGDS_32317 |
| lu12 | Cal DIG, 18 pairs, one unit |
| lu13 | NA080__MGDS_31290, NA080__MGDS_31289 |
| lu14 | AR26__MGDS_31838 |
| lu15 | TN299__MGDS_27339, TN299__MGDS_31253 |
| nu00 | FK151121__MGDS_22383 |
| vu02 | EX1402L2__MGDS_31073 |

That is 38 pairs in total. Verify this list against the manifest (`designation` starts with `development` and k is
recoverable), and report any difference as a challenge finding before proceeding.

**Pairs handled outside the development set:**
- **EW0207 (lu16):** reported separately, never pooled into the primary endpoint.
- **Not used (no-k):** KIWI10RR__MGDS_24499, TN268__MGDS_30466, FK181031__MGDS_24367, EX1202L2__MGDS_20815,
  2009_Amundsen__MGDS_30046, 2009_Amundsen__MGDS_30047.
- **Sealed:** lockbox AT42-03__MGDS_32007, TN159__MGDS_21981; confirmatory KN182L03__MGDS_30193 (nu01a),
  RC2511__MGDS_32239 (nu01b), SUM1004__MGDS_33090 (nu03).

**Guard.** Extend `r03_common.install_lockbox_guard()` so that it blocks:
- the 5 sealed pair ids;
- any path under `.../auv_ship_colocated_bathy/harmonized_confirmatory/`.

Every R04 script installs it. Self-test it at start: opening a lockbox path and a confirmatory path must both raise.

**Unit-set versioning.** The **code** is frozen, not the unit list. A unit added later (Venere, `pu00`, via manifest
v2.1.1) enters by:
1. running the frozen extraction, target, feature and bar code on it unchanged;
2. recomputing the bar on the enlarged set;
3. tagging `phase0-freeze+units-v<n>` and adding a ledger row.

Nothing else may change in that step.

**Cal DIG.** One unit. Evaluated on all its tiles, weighted as one unit.

**Weighting.** Unit-equal (median over units) is **primary**; pooled is secondary. In every fitted baseline, the
training rows from each unit are capped at **200** per fold by a seeded subsample, applied identically to all units.
State the per-unit counts used.

**Plane.** Excluded from every survey-planning target. The resolved-band tilt is a diagnostic only.

**Ship products.** Use contract v2.1 only, resolved per the contract from the manifest.
- A ship feature is used for a pair only where manifest `ship_products_usable` is true.
- Elsewhere it is NaN (never zero).
- Under that rule ship features are unavailable on lu02, lu06, lu09 (rulings R4) and lu12 (no swath).
- `ship_mean_regrid` and `ship_median_regrid` are never features.

---

## 1. Frozen targets: `targets_v1.py`

Implement once and use everywhere. All targets are computed from HR − δ (and the bicubic, where stated) on the
harmonized HR grid. They are computed at **unit-mosaic level**, not per tile, so that block and tile values do not carry
tile-edge filter artifacts.

### 1.1 Band-passed fields

For each development pair, on overlapping windows of 256 × 256 HR px with stride 128:

1. Remove a least-squares plane over joint-valid px and zero the invalid px.
2. Apply a Tukey (α = 0.25) taper.
3. Apply an FFT band-pass over [f_lr, f_hi], with f_lr and f_hi defined as in R02 T32B. Compute them from each pair's
   manifest v2.1 `lr_native` and HR resolution, in HR cycles per pixel. Use **raised-cosine transitions** of
   ±0.25 octave around each edge, not a hard annulus.
4. Keep the central 128 × 128 of each window.
5. Re-mask to joint-valid.

Produce two per-pixel fields:

- **`F_hr`**: the band-passed HR − δ. This is the true unresolved relief field.
- **`F_res`**: the band-passed (HR − δ − bicubic). This is the fine-band misfit; the plane is excluded by construction.

Store both per pair as float32 GeoTIFFs under the R04 scratch root.

**Consistency check, at matched support.** ACQ-R01 found that the R02/R03 tiles are 256 × 256 HR px windows, not the
128 × 128 the earlier R04 stated.

1. Establish from the R03 code which support R03's A_fine used, and report it.
2. Compare R03 A_fine with the RMS of `F_hr` computed over **the same support** (the same 256 windows, or their central
   128 if that is what R03 used).
3. Report the per-unit log-correlation and the median ratio on the native 7 and expansion units R03 covered.
   - Expect log-correlation ≥ 0.9. The filters differ (raised-cosine vs hard annulus), so the ratio need not be 1.
   - If any unit falls below 0.9, stop and report.

### 1.2 Summary targets (per block and per tile)

| target | definition |
|---|---|
| **A_fine** | RMS of `F_hr` over joint-valid px |
| **T_fine** | RMS of `F_res` over joint-valid px |
| **E_1m** (detection) | fraction of joint-valid px with \|`F_hr`\| > h, with **h = 1.0 m** |

**The E_1m threshold is a default.** Steve may change it before the §6 freeze; after that it is fixed.

Report the per-unit prevalence of E_1m > 0 at block scale. Units where fewer than 5 % or more than 95 % of blocks have
E_1m > 0 are excluded from the detection metrics. That exclusion rule is fixed now, and the excluded units must be
listed.

### 1.3 Spatial support: `tiles_v1`

- **Windows.** Every 256 × 256 HR px window at stride 128 with ≥ 50 % joint validity on every development pair. Use
  the R02 `extract_clean_patches.py` window logic, unchanged.
  - Pairs that already have R02/R03 windows reuse them; check the window grid is identical.
  - The 6 pairs new since R03 (TN399__MGDS_33068, AT42-06__MGDS_5174, KM0923__MGDS_31429, NA080__MGDS_31289,
    FK151121__MGDS_22383, EX1402L2__MGDS_31073) and any other pair without windows get them from the same code.
- **Tiles.** The central 128 × 128 of each window. At stride 128 these tile the pair without overlap and match the
  §1.1 field support.
- **Blocks.** The 16 non-overlapping 32 × 32 blocks inside each tile. A block is kept if ≥ 75 % of its px are
  joint-valid.

Evaluation uses all tiles and blocks. The R03 200-per-pair subsample no longer defines the evaluation set; the
training-row cap in §0.1 handles balance.

Report n windows, tiles and blocks per pair and per unit. Write `tiles_v1.parquet`: unit, pair, window id, tile bounds
in HR px and in CRS coordinates.

---

## 2. Frozen evaluation: `eval_v1.py`

- **Primary endpoint:** **A_fine at block scale, within-unit Spearman ρ, median over the 16 development units.**
- **Secondary endpoints:**
  - A_fine at tile scale;
  - T_fine at block and tile scale;
  - E_1m at block scale: within-unit AUPRC for E_1m > 0 and within-unit Spearman ρ on E_1m.
- **Metric set for every endpoint:**
  - within-unit Spearman ρ;
  - within-unit log–log R² (A_fine and T_fine only);
  - top-20 % capture;
  - each reported per unit, as the median over units, and pooled.
- **Differences:** bootstrap over units, 2,000 resamples, seed fixed in code, 95 % percentile CI, and the count of
  units where the first predictor is better. Also report the exact one-sided sign-test p-value of that count; it is
  reported, not used as a gate.
- **Interface:** input is a table of (unit, pair, tile_id, block_id, predictor columns); output is the tables.
  Predictors are plugged in by column, so Phases 2–4 score models through the same code.
  - `eval_v1` asserts that the unit set equals the frozen list (or a versioned extension of it), and refuses rows for
    sealed pairs.
- **EW0207** is scored through the same code as a separate one-unit table.

---

## 3. The bar: strongest free predictor at block and tile scale

### 3.1 Features

All features are ship-only. Re-run the R03 feature leak test on 100 tiles and 100 blocks.

**Block features.** Computed on ship windows centred on the **block** at two sizes, 128 and 224 HR px (the R03 (a)
and (b) spans measured in HR px). Where a spectral feature is undefined (too few cells), set it to NaN; the GBT handles
NaN natively.

**Features included:**
- the R03 expanded bank;
- the R02 free-combined features;
- roughness C;
- s_lr;
- metadata (depth, lr_native, k, hr_res);
- **contract-v2.1 ship features**, where `ship_products_usable` (§0.1). Over each window size compute:
  - median and 90th percentile of `ship_rsd`;
  - median of `ship_sd`;
  - median of log `ship_count`;
  - mean of `ship_xtrack_frac`;
  - mean of `ship_beam_angle`;
  - median of `ship_rsd / ship_sd`.

### 3.2 Models

1. **GBT_ship, nested leave-one-unit-out over the 16 units, with inner leave-one-unit-out tuning.** All features.
   Grid:
   - max_depth ∈ {2, 3, 4, 6};
   - n_estimators ∈ {100, 300, 600};
   - learning_rate ∈ {0.03, 0.1};
   - min_child_samples ∈ {20, 100}.

   The inner selection criterion is the median-over-inner-units Spearman ρ on the same target. Fit one model per
   target and scale; the target is log of the value. For E_1m, use P(E_1m > 0) via a classifier plus log E_1m where
   positive.
2. **GBT_noship:** identical, without the contract-v2.1 features. This separates the value of the ship products from
   the rest of the bank.
3. **Reference rows, no fitting:**
   - roughness C alone;
   - s_lr alone;
   - ship std at 128 px alone;
   - **median `ship_rsd` at 128 px alone** (on units where usable; per-unit availability reported).
4. **R02 σ_m** (native 7 units only, from the R02 mosaics): mean σ over the block or tile. This scores the old model on
   the new targets, **for reference only**.

### 3.3 Deliverable

`BAR_v1.md` plus the per-block and per-tile prediction tables for every model, target and scale.

**The bar for Phase 2 is GBT_ship at block scale on A_fine.** Report its primary-endpoint value and CI, and the
GBT_ship − GBT_noship difference with its CI and unit count.

---

## 4. Check A: differential control, re-measured

The submitted paper claims that the same model reaches R² ≈ 0.53 on total error but only ≈ 0.005–0.06 on fine-band
amplitude.

1. Locate the code and outputs that produced those numbers, and **reproduce them first** from the submitted pipeline.
   State what the reproduction matched or failed to match.
2. Fill the **2 × 2** table below on the native 7 units (NA090, NA080, TN399, RR1506, AT37-13, FK181031, TN299), in
   the paper's own metric and also in the §2 metrics:

   | | fine-band amplitude as the paper defined it (AUV-normalized) | A_fine at tile scale, in meters |
   |---|---|---|
   | submitted σ (het_production_v1) | | |
   | R02 clean σ_m | | |

   Add a row for total error (T-ship, R02 definition) against each model.
3. State in one sentence whether the gap between total-error skill and fine-amplitude skill survives in meters with the
   clean model. If a column cannot be computed, mark it COULD NOT CHECK and say why.

---

## 5. Check B: what σ_norm learned

Native 7 units, R02 mosaic σ, T-ship tile set. Within-unit partial Spearman correlations, pooled median over units
with bootstrap CI:

1. log σ_norm vs log slope (R02 bicubic tile slope), given log s_lr.
2. log σ_norm vs log T_res, given log s_lr — without and with log slope as an additional control.
3. log σ_norm vs log A_fine, given log s_lr and log slope.

Report the numbers with no interpretation beyond one sentence each.

---

## 6. Preservation and freeze

1. **Copy to OAK** (`.../auv_ship_colocated_bathy/grl_review_archive/`), with `SHA256SUMS` verified:
   - R02 `patches_clean/`, `mosaics/`, `het_clean_r02/`;
   - R03 `patches_clean_expanded/`, `s3/`, `s4/`, `s6/pilot/raw/`;
   - R04 fields, `tiles_v1.parquet`, feature tables and prediction tables.

   Do this **first**, before any other R04 compute. Report sizes. Scratch copies may remain.
2. **Commit** `INTERFACE_CONTRACT_v2.1.md` verbatim (its sha256 must match the hash in the header), plus a validator
   `contract_v2_1_validate.py`.
   - The validator checks schema, grid identity with `lr.tif`, NaN/count conditions, and the
     `ship_products_usable` gate.
   - Unit-test it on a synthetic conforming directory and a synthetic non-conforming one.
   - Run it on all development pairs and report the result.
   - Use `ship_products_v2_1/` everywhere; the v2 products of NA080__MGDS_31289 are deficient.
3. **Freeze.** Tag the commit `phase0-freeze`. In the report, record:
   - the sha256 of `targets_v1.py`, `eval_v1.py`, the feature code, the bar code and `tiles_v1.parquet`;
   - the manifest sha256;
   - the contract version;
   - the frozen unit list;
   - the E_1m threshold in force.

---

## 7. Input-representation record (report only, no decisions)

Phase 2 and the Phase 3 pretraining corpus must share one input specification. Record exactly what the current CNN
pipeline (R02 clean model and its patch extractor) feeds the network:
- LR representation (native grid or resampled to HR px, interpolation, window size in LR cells and HR px);
- normalization (per patch or global, which statistics);
- channels;
- augmentation;
- how `lr_native`, k and depth enter, if at all.

Also report, per development unit:
- LR cell size in metres;
- LR cells per 256-px window;
- the range of the ratio HR px / LR cell.

Make no change and no proposal; this is the factual basis for the Phase 2 input spec.

---

## 8. Report

- **§1 Bottom line**, at most 8 points:
  - challenge findings, including the unit-list verification;
  - target consistency check at matched support;
  - the bar: GBT_ship primary endpoint, value and CI, and GBT_ship − GBT_noship;
  - where the reference rows and R02 σ_m sit against the bar;
  - differential-control 2 × 2 verdict;
  - Check B;
  - preservation and contract validation;
  - freeze hashes.
- **HOLDs:** at most 3.
- **"Not pursued":** one line per item, no analysis.
- Scripts as run under `directives_and_reports/R04/scripts/`.

## 9. Out of scope

- Training of any network.
- GPU use.
- Architecture proposals.
- Adoption criteria.
- Pretraining-corpus work.
- Manuscript text.
- Any lockbox or confirmatory access.
- Writing to the acquisition repo or to OAK outside `grl_review_archive/` and the R04 scratch root.
- Any analysis not listed above.
