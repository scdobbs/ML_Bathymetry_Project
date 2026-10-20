# PREREG_phase4_v1 — Phase 4 pre-registration (frozen; tag `phase4-prereg`)

**Written:** 2026-10-09 by directive R09 Part B, from verified records; not executed in R09. **Rulings in force (Steve, 2026-10-09):** CNN development is closed for this paper; the CNN enters Phase 4 as one arm of a model-complexity comparison; the frozen candidates are s_lr, GBT_ship and P-V1 (seed-averaged); development scoring uses interior tiles at D₁ on the primary ten real-relief units (lu14 shown on all tiles beside the interior rows and kept out of every median); Phase 4's main question is whether within-site ranking of unresolved relief transfers to the sealed units, whose ship cells (103–151 m) are coarser than anything in development. Candidate comparisons are secondary and descriptive.

**Fallback status from R09 Part A:** {{FALLBACK_STATUS}}

## B1. Sealed units

| set | pairs | role |
|---|---|---|
| confirmatory | KN182L03__MGDS_30193 (nu01a), RC2511__MGDS_32239 (nu01b), SUM1004__MGDS_33090 (nu03) | primary |
| lockbox | AT42-03__MGDS_32007, TN159__MGDS_21981 | supporting only |

**Lockbox history (R09 §B1, CONFIRMED with file):** {{LOCKBOX_HISTORY}} Lockbox results are therefore reported as supporting and labelled "scored by the submitted pipeline's het_production_v1_repro model (2026-07-08) and used as validation / test splits of the first het training (2026-06-30)".

## B2. Frozen candidates (artifact paths and sha256)

1. **s_lr:** the `ref_s_lr` definition: `s_lr = max(SD of the valid LR cells of the 256-px window, 0.5 m)` from `preprocessing/extract_clean_patches.py:build_inputs` (the tiles_v1.1 `s_lr` column), computed from the sealed pairs' LR only. No fitting. Code sha256 (`extract_clean_patches.py` at this commit): `{{ECP_SHA}}`.
2. **GBT_ship:** no persisted all-16 model existed (R05 fitted it in memory to score EW0207, configuration = the most frequent per-fold choice, {{R05_MODE_COUNT}} of 16). Refit once in R09 with the frozen bar code (`bar_v1_1.py`, features `bar_v1_1.ALL`, target log₁₀ A_fine at tile scale, cap 200 rows per unit, seed 20261001), configuration chosen by the frozen inner criterion over all 16 units (leave-one-unit-out, eval_v1_1 within-pair ρ, median over units): cfg {{GBT_CFG_IDX}} = {{GBT_CFG}}; artifact `grl_review_archive/R09/artifacts/gbt_ship_all16_A_fine_tile.joblib`, sha256 `{{GBT_SHA}}`; record `gbt_ship_all16_record.json`. {{GBT_R05_NOTE}} It is not refit after this.
3. **P-V1:** the three R07 all-unit fine-tuned models (`R07/models/P-V1/seed<seed>/fold_ALL/ckpt.pt`, OAK `grl_review_archive/R07/models/`), seed-averaged as in R07 (mean of the three seeds' log₁₀ block predictions; tile = RMS over the 16 blocks): seed 20261002 `{{PV1_2002}}`, seed 20261003 `{{PV1_2003}}`, seed 20261004 `{{PV1_2004}}`.

No other predictor is scored at Phase 4.

## B3. Order of operations at Phase 4 (each step a separate directive or job; nothing out of order)

1. **Acquisition repo:** build the sealed pairs' contract-v2.1 ship products with the frozen code (needed for GBT_ship features). Context grids are not needed.
2. **CNN repo:** with the frozen code unchanged, build the sealed pairs' windows, tiles, fields (`F_hr`, `joint_v1`), `targets_v1` and `targets_v1_int` (sha256 `{{TVI_SHA}}`), the interior masks at D₁, and the feature table. **No prediction or score yet.**
3. **Blind artifact judgement:** produce figure set A for the 5 sealed pairs exactly as in R08 §3.3 (hillshade and `F_hr`, shuffled and coded, no names or scores; key held in a separate file). Steve judges each page against the R08 §3.1 criteria. The calls are committed with their sha256 **before** any prediction is computed. The scoring script refuses to run unless that calls file exists, with its hash recorded.
4. **Predict** with the three frozen candidates.
5. **Score once** (B4). Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.

## B4. Scoring rules

- **Statistic:** the `eval_v1_1` within-pair Spearman ρ on A_fine at tile scale. Each sealed pair is its own unit (nu01a, nu01b and nu03 are separate units under the leakage policy).
- **Tiles:**
  1. Interior tiles at D₁ (≥ 95 % of joint-valid px at ≥ D₁ = ⌈lr_native / hr_res⌉ px from any invalid px; R08b rule).
  2. If a pair has fewer than 10 interior tiles **and** the fallback was validated in R09 Part A ({{VALIDATED_SHORT}}), score it instead on tiles with a defined A_fine_int (≥ 2,048 interior px).
  3. If it still has fewer than 10, the pair is **not scorable** and is reported as such. There is no further fallback.
- **Uncertainty per pair:** a spatial-block bootstrap of ρ over 4 × 4-tile blocks (tile-grid coordinates r0 / 128, c0 / 128), 2,000 resamples, seed 20261009, 95 % percentile CI.
- **Secondary:** all-tile scores against A_fine, reported beside the primary.
- **Artifact handling:** a pair judged **artifact-dominated** in B3.3 is reported separately and excluded from C1; a pair with whole-survey stripes or ribbing is flagged, but stays in.

## B5. Criteria (fixed now)

- **C1, primary: ranking transfers.** **Confirmed** if, on every scorable, non-artifact-dominated confirmatory unit, s_lr's ρ has a block-bootstrap CI lower bound > 0. **Partially confirmed** if this holds on some but not all. **Not confirmed** if on none. Expectation from development (interior tiles at D₁, primary units; R08b `scores/scores.json:per_unit.D1.ref_s_lr`, 9 units): unit ρ for s_lr between {{EXP_MIN}} and {{EXP_MAX}}, median {{EXP_MED}}. Report each sealed unit's ρ against this range.
- **C2, secondary and descriptive: model complexity.** Per sealed unit, report GBT_ship − s_lr; P-V1 − s_lr; P-V1 − GBT_ship. Write one sentence per comparison stating its direction across units. No superiority claim is made: with at most 3 units, even 3 of 3 gives a one-sided sign test of p = 0.125. State this in the result.
- **C3, supporting:** C1 and C2 on the lockbox pairs, reported separately with their history label.

## B6. Provenance requirements at Phase 4

Every Phase 4 script verifies the sha256 of `PREREG_phase4_v1.md` (recorded in `CHANGE_LEDGER.md` and REPORT_R09 at the `phase4-prereg` tag), the candidate artifacts (B2) and the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash `{{R07_CODE_SHA}}`) against this document before running, and writes its outputs to `grl_review_archive/phase4/` with `SHA256SUMS`.
