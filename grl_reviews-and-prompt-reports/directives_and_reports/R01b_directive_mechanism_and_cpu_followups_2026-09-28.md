# R01b — Directive: confirm the blow-up mechanism, and CPU follow-ups from R01a

**Series:** post-review (R-series).
**Issued:** 2026-09-28
**Issued by:** steering/assessment instance, on Steve's ruling after REPORT_R01a
**Executor:** Claude Code on Sherlock (the R01a instance is fine; a fresh instance must read R01 and REPORT_R01a first)
**Repo:** `/home/users/scdobbs/GAN_Bathymetry_Project`, branch `rewrite/flow-matching-2026`
**Report:** `grl_reviews-and-prompt-reports/REPORT_R01b_<date>.md`, append-only
**Arbiter:** Steve.

---

## 0. Decisions of record (Steve, after R01a)

1. **The B1 leak is accepted as CONFIRMED.** Every σ result in the submitted paper (0.555, top-20% capture,
   §3.4, Figs 3–4) came from a model whose only input was normalized by the AUV patch's mean and std. The
   revision will disclose this and replace those results with a retrained model (R02).
2. **R01 Phase C is cancelled.** C2b (old weights, LR-only inputs) measures sensitivity, not skill, and C1 on
   leaked inputs is not worth the time. Nothing from Phase C runs.
3. **EW0207 is treated as defective** (R01a §A7): it is excluded from training in R02 and from headline
   numbers, and reported separately everywhere. Log this in `CHANGE_LEDGER.md` with the A7 evidence.
4. **R01b is CPU-only and uses existing artifacts.** Its purpose is to put the mechanism on record and to
   supply the two numbers R02 needs (the rim width and the NLL decision).

## 0.1 Standing constraints

Same as R01 §0.1: no training; no new weights; read-only on OAK canonical data, the production model,
the production parquet, the reproduction weights and all prior reports; append-only; escalate rather than
absorb; certainty labels (CONFIRMED / CONTRADICTED / COULD NOT CHECK); log any evaluation change in
`CHANGE_LEDGER.md`. Outputs to `/scratch/users/scdobbs/grl_review/R01b/`, with the report and small tables
copied to `grl_reviews-and-prompt-reports/R01b/` (committed) and rsync'd to the OAK
`analysis_archive_2026-09-28/grl_review/R01b/`. Use the restored patch set from R01, not the scratch original.
GPU: none, except that B2 may use one GPU for under 15 minutes if CPU probing is impractical.

---

## Task 1 — Mechanism (R01 B2–B5, unchanged in scope)

Use the reproduction model's persisted per-pixel μ/σ maps and weights (`dir55/repro/het_folds/`), each unit
scored by its held-out fold, the 8 native units, EW0207 reported both included and separately.

**B2. Effective receptive field.** Gradient probing on 20 patches (|∂μ(center)/∂input| footprint).
Report the radius holding 95% and 99% of the gradient mass. **Recommend a rim for R02** = the 99% radius,
rounded up to a multiple of 8, minimum 32 px.

**B3. Blow-up anatomy against inputs (the direct test of H-norm).** For the worst 1% of |HR − μ| pixels
versus all valid pixels: distance to tile border; distance to nearest invalid pixel; local |normalized input|;
patch HR std (m); patch |normalized offset|; whether the pixel lies within the LR-fill step created by filling
invalid LR cells with the HR mean (R01a B1 table, `P:312`). Report the per-patch fraction of |μ| > 5 pixels by
decile of HR std and by decile of max |normalized input|. **State plainly whether H-norm is CONFIRMED,
CONTRADICTED or partial**, and what fraction of the blow-ups each factor accounts for.

**B4. Error versus distance to border.** Mean and median |HR − μ| and mean σ vs distance to the tile border,
0–64 px, pooled and per unit; the distance at which each reaches its interior level.

**B5. σ behaviour and the NLL mean-abandonment test.** σ distribution; fraction of pixels at the clamp
(log_var = 6, σ = 20.09); σ vs distance to border. Then the signature test: within patches, is the mean's
error largest exactly where σ is inflated, beyond what input magnitude explains? Compute partial Spearman of
|HR − μ| on σ controlling for |normalized input| and distance to border, rim vs interior.
**Recommend for R02: standard Gaussian NLL, or β-NLL (Seitzer et al., 2022) with β = 0.5.** Recommend β-NLL
only if the partial correlation in the interior is materially positive (report the number; Steve decides).

## Task 2 — Which other paper numbers inherit the leak?

1. **§3.1 within-survey conditional (report 44 support A).** From code (file:line), confirm whether its tiles,
   features and target are computed in meters from HR and LR independently, or use any AUV-derived
   normalization of the ship-side features. One-line verdict.
2. **§3.2 transfer regression (support B).** Its features are computed from `lr_up` normalized by the AUV
   std. Confirm from code. Then re-run the review assessment's meters version (target and features converted
   back to meters with `stats`) with **EW0207 excluded**, and again with it included, for N = 7/8 native and
   N = 14/15 expanded. Report per fold: raw R², R² with per-unit mean removed, R² with mean and scale removed,
   Pearson r, Spearman ρ; plus medians and pooled values. Log-space only, as the frozen convention.
3. **Roughness heuristic provenance.** From code, confirm whether roughness C / roughness-3 is computed from
   the ship grid alone and in which units. If it uses any AUV-derived normalization, say so: R02 must
   recompute it.
4. **Metadata baseline provenance (report 44 Task 5).** Same check for every feature.

## Task 3 — Manifest vertical-offset discrepancy (R01a §A5)

For EW0207, FK181031 and TN299b (manifest offset −27.7, −12.6, −4.8 m; patch median ≈ 0) and the AT37-13
entry (−7936.62, a depth): read the manifest notes and the "C1 float recovery" code path. Determine whether
a vertical shift was applied to the harmonized grids after the manifest value was recorded. **Propose** the
corrected manifest values and a one-paragraph provenance note. **Do not write the manifest.** This must be
fixed before the corpus is deposited publicly; the write is a separate, explicitly authorized step.

---

## Report structure (`REPORT_R01b_<date>.md`)

1. Bottom line (≤ 10 lines): H-norm verdict; rim recommendation; NLL recommendation; which paper numbers are
   leak-affected; manifest finding.
2. Task 1 tables and figures (B3 decile plots; B4 profiles).
3. Task 2 verdicts with file:line, and the meters transfer table (EW0207 in and out).
4. Task 3 findings and proposed manifest corrections.
5. Change-ledger rows added.
6. Reproduction commands and paths.

## HOLD for Steve

- The rim and NLL recommendations (inputs to R02 §2).
- Any Task 2 finding that roughness or metadata baselines are leak-affected.
- The proposed manifest corrections.
