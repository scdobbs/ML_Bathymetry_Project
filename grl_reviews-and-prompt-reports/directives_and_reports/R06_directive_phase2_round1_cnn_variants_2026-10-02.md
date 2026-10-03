# DIRECTIVE R06 (Phase 2, round 1, CNN repo): retrain the CNN on the frozen target, four pre-registered variants, scored against the bar

**Date:** 2026-10-02
**Author:** steering instance, for Claude Code on Sherlock (CNN repo)
**Predecessors:** R04 and R05 reports; tag `phase0-freeze` (commit `866320b7e0917460b63060e86b4a8be8b409c64e`).
**Report to:** `directives_and_reports/R06/REPORT_R06_<date>.md`
**Compute:** GPU allowed for training. Budget in §5.
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK. Meters unless marked.

---

## 0. Purpose and plan context

| phase | what | status |
|---|---|---|
| 0 | Frozen targets, evaluation and bar | closed (`phase0-freeze`) |
| 1 | Acquisition and confirmatory set | closed |
| 2 | CNN development on the 16 development units | **round 1: this directive** |
| 3 | Self-supervised pretraining, tested as a pre-registered ablation | later |
| 4 | Pre-registered freeze and one confirmatory scoring | later |

**The question for round 1:** retrained on the frozen target, does a CNN beat the bar, and which of three design
changes (a level-free loss, ship-product channels, spatial context) moves it?

**What we know going in (R05):**
- The bar is GBT_ship, A_fine, tile scale, eval_v1.1: median ρ **0.510** [0.362, 0.604].
- The R02 CNN, trained leave-one-unit-out on a different target, already scores 0.489 on its 7 native units, level
  with GBT_ship on those units (−0.003, GBT better in 2 of 7).
- Ship products add nothing measurable to a GBT (+0.030, 9 of 16 units).
- A one-sided sign test at p < 0.05 needs at least 12 of 16 units better.

**This round does not adopt a model.** It produces scored, comparable candidates. Adoption criteria are written
separately, before Phase 4.

**Scope discipline.** Do the items below and nothing else. If a result suggests a new analysis, list it in one line
under "Not pursued". Do not run it.

**Challenge clause.** Report any wrong or underspecified instruction in §1 of the report, with evidence. Items that are
Steve's call become HOLDs (at most 3).

### 0.1 Fixed inputs (all read-only)

- Frozen code and tables at `phase0-freeze`: `targets_v1.py`, `eval_v1_1.py`, `features_v1_1.py`,
  `tiles_v1_1.parquet`, `blocks_v1_1.parquet`, the R04 field mosaics (`F_hr`, `joint_v1`), the R05 bar prediction
  tables. Record their sha256 at start; stop if any differs from R05 §6.
- Manifest v2.1 (sha256 `f57f3d8f…6ac35`) and contract v2.1, as in R05.
- Unit set `units-v1`: 16 units / 38 pairs. EW0207 is scored separately and never trained on. No-k pairs are not used.
- The lockbox and confirmatory guard from R04, installed in every script, including data loaders. Self-test it at
  start.

### 0.2 Rules that bind every variant

1. **Leave-one-unit-out.** 16 outer folds. A fold's model never sees any pair of the held-out unit, in training,
   validation, normalization statistics or any fitted quantity.
2. **No tuning on held-out data.** No early stopping, checkpoint choice or hyperparameter choice may use the held-out
   unit. Train for a fixed schedule (§2.4) and score the final checkpoint.
3. **Ship-side inputs only.** No HR-derived quantity may enter any input, including the joint-validity mask (it uses
   HR validity). Re-run the R04 leak test on the input pipeline: inputs must be byte-identical when HR is replaced
   by noise.
4. **Scoring only through `eval_v1_1.py`.** Predictions are written as tile and block tables in the bar's format and
   scored by the frozen code, against the R05 bar predictions.

---

## 1. Shared design (all variants)

### 1.1 Target and output

- Target per block: **log₁₀ A_fine** of the 16 blocks of each tile (from `blocks_v1_1`; non-kept blocks masked in
  the loss).
- The network outputs a **4 × 4 map of predicted log₁₀ A_fine per tile** (one value per 32-px block).
- **Tile prediction** = √(mean over all 16 blocks of (10^pred)²), i.e. the RMS combination with equal weights (no
  HR-derived weighting).
- Block predictions are scored at block scale; tile predictions at tile scale.

### 1.2 Input window and representation

- Input = the 256 × 256 HR-px window whose central 128 px is the tile (R04 window rule), except V3 (§2).
- LR representation, normalization, the validity channel and FiLM scale conditioning **as recorded in R04 §7**
  (area-average to native LR cells, bicubic to HR px, per-window normalization by mu_lr and s_lr, LR-validity
  channel).
- **Add to the FiLM conditioning vector:** log₂ s_lr and log₂ |mu_lr| of the window. Per-window normalization removes
  s_lr from the image, and s_lr alone scores 0.465 on the primary endpoint, so the network must be given it.
  Conditioning vector = [log₂ lr_native, log₂ hr_res, log₂ s_lr, log₂ |mu_lr|].

### 1.3 Architecture

- The R02 HetUNetClean encoder and FiLM blocks, unchanged in depth and width. Replace the per-pixel
  (mean, log-variance) head with a head that pools the decoder output to the 4 × 4 block grid of the central tile and
  regresses log₁₀ A_fine per block. No variance head in this round.
- Report the parameter count. Do not enlarge the network.

### 1.4 Sampling and augmentation

- **Unit-balanced sampling.** Draw a unit uniformly from the training units, then a pair uniformly within that unit,
  then 8 windows from that pair. One such draw is a **group**. A batch is 4 groups (32 windows).
- **Augmentation:** the 8 symmetries of the square (rotations by 90° and flips), applied identically to all input
  channels and to the 4 × 4 target map. Nothing else.
- Training windows are the `tiles_v1_1` windows of the training units (stride 128).

### 1.5 Training schedule

- Optimizer, learning rate, schedule and epoch count **as in the R02 recipe (`het_louo.py`)**, converted to the same
  number of optimizer steps under the §1.4 sampler. Report the values used.
- One seed per variant per fold (seed 20261002), except §2's seed check.

---

## 2. Variants (pre-registered; run exactly these)

| id | change from the one above it | what the comparison tests |
|---|---|---|
| **V0** | Shared design (§1), with an **absolute** loss: MSE on log₁₀ A_fine over kept blocks | Control: the R02 network retrained on the frozen target |
| **V1** | V0 with a **within-pair centred** loss: within each group, MSE between (pred − group mean of pred) and (target − group mean of target) over kept blocks | Does training on what is scored (rank within a pair, not level) help? V1 − V0 |
| **V2** | V1 plus a **ship-product branch**: `ship_rsd`, log `ship_count`, `ship_xtrack_frac`, `ship_beam_angle` on the native LR grid, bicubic to HR px, each with its own availability mask; a separate encoder branch fused before the head; channel dropout p = 0.3 per product during training | Do the contract-v2.1 products help a CNN? V2 − V1 |
| **V3** | V1 with the input cropped to the **central 128 px** (no context ring); the head is unchanged | Does spatial context beyond the tile help? V1 − V3 |

Notes:
- V2's products follow the contract consumer rules: used only where `ship_products_usable` is true and the R05
  override list allows; elsewhere the channel is zero **and** its availability mask is zero. Never fill without the
  mask. Report per-unit availability.
- Group means in V1–V3 use each group's kept blocks only. V0 uses the same sampler so that V1 − V0 isolates the loss.
- **Seed check:** train V1 with two further seeds (20261003, 20261004) on all 16 folds. Report the spread of the
  primary endpoint across the three seeds. This sets the noise floor for reading the V-differences.

---

## 3. Scoring and comparisons

For each variant (V1 with its three seeds reported separately and as a seed-averaged prediction):

1. Score every endpoint in `eval_v1_1.py` at tile and block scale, as in R05 §4.1.
2. **Against the bar:** for each variant, the difference to GBT_ship on A_fine/tile (median, 95 % CI, units better of
   16, sign-test p), and the same against ref_s_lr and ref_C.
3. **Between variants:** V1 − V0, V2 − V1, V1 − V3 on A_fine/tile, same statistics.
4. **On the 7 native units:** each variant against R02 σ_m (LOUO) under eval_v1.1.
5. **EW0207:** for each variant, train one additional model on all 16 units (same schedule and seed) and score EW0207
   with it as the separate one-unit table, as the bar did.
6. **Per-unit table** of A_fine/tile ρ for every variant, GBT_ship, ref_s_lr and ref_C, with each unit's median A_fine.

**Reading rule, stated now:** the best round-1 variant is the one with the highest median ρ on A_fine/tile; when two
differ by less than the seed spread from §2, the simpler one (fewer inputs, then smaller window) is named. This picks a
candidate for round 2; it is not adoption. Report plainly that choosing among four variants on the same 16 units
biases the winner's development score upward.

---

## 4. Diagnostic: low-amplitude units (report only, no decision)

Six development units have median A_fine below 3 cm (lu02, lu06, lu10, lu11, lu12, vu02). Where skill there comes from
proximity to data edges or holes rather than relief, it inflates every model's median.

1. Score `dist_edge` and `frac_valid` (from the frozen feature table, 128-px window) as reference rows under
   eval_v1.1, per unit, at tile scale.
2. For GBT_ship and each variant, report the within-pair partial Spearman ρ between prediction and A_fine given
   `dist_edge` and `frac_valid`, per unit, separately for the six low-amplitude units and the other ten.
3. One sentence per item. No interpretation beyond that.

---

## 5. Compute

1. **Pilot first:** train V1 on one fold (held-out unit lu00) and record wall time and GPU-hours.
2. Extrapolate to the full run: 4 variants × 16 folds, plus 2 seeds × 16 folds for V1, plus 4 all-unit models for
   EW0207 (100 trainings).
3. **Budget cap: 400 GPU-hours.** If the extrapolation exceeds it, stop after the pilot and report the estimate as a
   HOLD with options (fewer seeds; V3 dropped; shorter schedule). Do not cut anything on your own.

---

## 6. Preservation and freeze of the round

1. Commit the training, model and inference code. Record its sha256.
2. Save every fold's final checkpoint, the per-fold training logs and the prediction tables. Copy them to OAK
   `grl_review_archive/R06/` with `SHA256SUMS` verified.
3. Tag the commit `phase2-r1`. Add one `CHANGE_LEDGER.md` row describing round 1.

---

## 7. Report

- **§1 Bottom line**, at most 8 points:
  - challenge findings and the guard / leak-test results;
  - compute used against the budget;
  - each variant on the primary endpoint (value, CI) and against the bar (difference, CI, units better, sign-test p);
  - V1 − V0, V2 − V1, V1 − V3;
  - the seed spread;
  - the variants against R02 σ_m on the native 7;
  - EW0207;
  - the low-amplitude diagnostic, one sentence.
- **HOLDs:** at most 3.
- **"Not pursued":** one line per item, no analysis.
- Scripts as run under `directives_and_reports/R06/scripts/`.

## 8. Out of scope

- Any change to the frozen targets, tiles, evaluation, bar or features.
- Hyperparameter search, architecture changes beyond §1.3 and §2, or larger networks.
- Uncertainty heads, calibration, or a Laplace likelihood.
- Pretraining or any pretraining-corpus work.
- Adoption criteria.
- Lockbox or confirmatory access.
- Manuscript text.
- Any analysis not listed above.
