# R01 — Directive: data-integrity audit, model input/output audit, and no-retrain fixes

**Series:** post-review (R-series). R01 is the first directive after the GRL review of 2026GL125574.
The pre-review series ended at Directive 57a; R-numbers restart here and do not continue that sequence.
**Issued:** 2026-09-28
**Issued by:** steering/assessment instance, on Steve's instruction
**Executor:** Claude Code on Sherlock (fresh instance)
**Repo:** `/home/users/scdobbs/GAN_Bathymetry_Project`, branch `rewrite/flow-matching-2026`
**Directive/report folder:** `grl_reviews-and-prompt-reports/` in the repo
**Report:** `REPORT_R01_<date>.md` in that folder, append-only. Large phases may be reported in parts (R01a, R01b, …).
**Arbiter:** Steve.

---

## 0. Why this directive exists

The review assessment of 2026-09-28 (`REVIEW_ASSESSMENT_2026-09-28.md`, `HANDOFF_grl_review_2026-09-28_1.md`)
found that the CNN's mean output ranges from −51 to +65, in units where the patch HR has std 1. It also found
that 97% of the worst 1% of pixels lie within 8 px of the 256-px tile border, and that σ's headline R² of 0.555
is dominated by those pixels.

**Before the paper's claims are changed, Steve wants to know how the model actually performs once data defects
and edge artifacts are ruled out or removed.** This directive finds out. It does not decide framing, and it does
not retrain. A retrain, if warranted, is a separate directive (R02) issued after Steve reads this report.

Two hypotheses to test, not assume:

- **H-norm.** Patches are z-scored by their own HR mean and std. On low-relief patches, a meter-scale ship–AUV
  vertical offset then becomes a normalized input of order 10–100. Combined with zero padding, that produces
  the border blow-ups.
- **H-leak.** If HR-derived statistics (mean, std, or anything else computed from HR) are used to scale the
  model's inputs at evaluation, then the model sees AUV information at the "applied site." That contradicts
  the paper's claim of validation "with no AUV data at the applied site."

Neither is established. The handoff's description of the normalization has not yet been checked against the
training and evaluation code line by line. Doing that is Phase B1.

---

## 0.1 Standing constraints

- **No training.** No new model weights. GPU use is limited to *inference* with existing weights (Phase C).
  Gate: 2 GPU-hours total. Stop and report if the estimate exceeds that.
- **Read-only inputs.** Do not modify OAK canonical data, `models/het_production_v1/`, the production parquet,
  the reproduction weights, or any prior report.
- **Outputs** go to `/scratch/users/scdobbs/grl_review/R01/`. The report and small result tables are also
  copied to `grl_reviews-and-prompt-reports/R01/` in the repo and committed. At the end, rsync the R01 output
  directory to OAK `.../auv_ship_colocated_bathy/analysis_archive_2026-09-28/grl_review/R01/`. This is an
  additive write into the archive, not a canonical data write.
- **Append-only.** New dated files only. Never overwrite a prior report or result.
- **Escalate, don't absorb.** Stop that phase and report to Steve if you find any of: (i) sentinel or fill
  values inside pixels marked valid; (ii) HR-derived quantities entering model inputs at evaluation;
  (iii) a restore count mismatch; (iv) anything that contradicts this directive.
- **Change ledger.** Create `grl_reviews-and-prompt-reports/CHANGE_LEDGER.md` if it does not exist. Log every
  deviation from the submitted-paper conventions as one row: what changed, the defect evidence that motivated
  it (file:line or a result), the date, and the directive. R01 changes only evaluation, never the model, but
  log those evaluation changes too.
- **Holdouts.** Do not score the two permanent never-trained holdout units in R01. Scope is the 8 native LOUO
  units, each scored with the fold model that did not train on it. The holdouts stay untouched for a final
  evaluation. This is a default Steve may reverse.
- **Certainty labels** throughout: CONFIRMED (checked against code or data), CONTRADICTED, COULD NOT CHECK.

## 0.2 Environment and inputs

```
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate
```

| Input | Location |
|---|---|
| Frozen patch set (108,833 `.npz`) | OAK `.../auv_ship_colocated_bathy/patches_native_phase2_frozen_2026-07-01.tar` |
| Production model (code, seed, per-patch σ parquet; no weights) | repo `models/het_production_v1/` (commit 29633a6) |
| Reproduction weights and per-pixel μ/σ maps | `analysis_archive_2026-09-28/dir55/repro/het_folds/` on OAK (or scratch `dir55/` if still present) |
| Harmonized source grids and masks, manifest | OAK `.../auv_ship_colocated_bathy/{harmonized,manifest}` |
| Frozen conventions | `step0_bakeoff.py`, `task3_covariates.py`, `boundary_suite.py`, `het_louo.py` (archived copies authoritative) |
| Prior review scripts | `/scratch/users/scdobbs/grl_review/` and repo `grl_reviews-and-prompt-reports/scripts/` |

**Patch-set restore (do first).** The scratch copy
`/scratch/groups/hilley/bathy_diffusion_2026/patches_native_phase2` becomes purge-eligible from 2026-09-29 and
may be partially purged, which would fail silently. Do not read from it. Extract the OAK tarball to
`/scratch/users/scdobbs/grl_review/R01/patches_restored/`. Verify 108,833 files and per-split counts. Then
verify sha256 on a random sample of 200 files against the scratch originals where they still exist. Report
the result. A mismatch is an escalation.

---

## Phase A — Data integrity (no model)

Run over **every patch in the restored set** and over the **harmonized source grids** for the 8 native units.
Report per unit and overall.

**A1. Sentinels and non-finite values.** Count NaN, ±inf, values ≤ −9000, exact common fill values
(−9999, −32768, −32767, 3.4028235e38, 0 in contiguous runs ≥ 16 px), and constant runs ≥ 16 px. Do this in
HR, LR and every mask/validity array. Split every count into pixels marked valid and pixels marked invalid.
Any sentinel inside a valid pixel is an escalation. For the source grids, read with and without masking, and
report the nodata value in each file's metadata. This is the `masked=True` bug class the project has hit before.

**A2. Mask consistency.** Per patch: fraction valid under each mask (`hr_valid`, `lr_valid`, `joint_valid`).
Report pixels marked valid but non-finite or sentinel, and pixels marked invalid but holding plausible depths.
Also report the fraction of patches whose invalid region touches the tile border.

**A3. What `stats` holds.** Confirm from the patch-writing code (find it; report file:line) exactly what each
entry of `stats` is, which array it was computed from (HR or LR), and over which pixels (all or valid only).
Recompute each from the raw arrays for 500 random patches and report agreement.

**A4. Relief and normalization magnitude.** Per unit, the distribution (min, 5/25/50/75/95th percentile, max)
of: patch HR std in meters; patch LR std in meters; max |normalized LR input|; max |normalized HR|. Count
patches with HR std < 0.05, 0.1, 0.25 and 0.5 m, and patches with max |normalized input| > 5, 10 and 50.

**A5. Vertical offset.** Per patch, median(HR − bicubic LR) over joint-valid pixels, in meters and in
normalized units. Report the distribution per unit, the correlation of |normalized offset| with 1/HR std,
and whether the offset is spatially coherent within a unit (a datum offset) or varies patch to patch.

**A6. Physical plausibility and orientation.**
- Depth sign convention is consistent between HR and LR in every unit.
- Depth ranges match the manifest.
- Coarse-scale agreement: correlation between HR low-passed to the ship's scale and bicubic LR, per patch.
  Flag patches below 0.5. A low value suggests a flip, transpose, shift, or co-registration failure.
Report flagged patches by unit, with a few thumbnails.

**A7. Flagged units.** Summarize A1–A6 for EW0207 (degraded navigation) and TN399 (weakest within-survey unit).
Say whether either unit's data looks defective, not merely difficult.

---

## Phase B — Model inputs and outputs (code reading + existing outputs)

**B1. Pipeline table (code reading; file:line for every row).** For both **training** and **evaluation**
(`eval_fold`), record:
- every input channel and its provenance (LR-only, HR-derived, or mask);
- the normalization applied to each channel and to the target, and which statistics are used;
- how invalid pixels are filled before entering the network, and with what value;
- whether invalid pixels are excluded from the loss;
- the padding mode of every conv layer;
- the σ parameterization (log-variance? clamps? min/max?);
- the output activation;
- patch size and stride at training and at evaluation;
- augmentations.

Then state plainly: **does any HR-derived quantity reach the network's inputs at evaluation?** If yes, this is
an escalation. Stop B and report before continuing to C.

**B2. Receptive field.** Compute the network's theoretical receptive field. Measure the effective one by
gradient-based probing on 20 patches. This sets the minimum rim for Phase C.

**B3. Blow-up anatomy against inputs.** Using the reproduction model's per-pixel maps, for the worst 1% of
|HR − μ| pixels versus all valid pixels, compare: distance to tile border; distance to nearest invalid pixel;
local |normalized input|; patch HR std (m); patch |normalized offset|. Report:
- whether |μ| > 5 pixels are concentrated in low-HR-std or high-offset patches (test H-norm);
- the per-patch fraction of |μ| > 5 pixels as a function of HR std decile.

**B4. Error versus distance to border.** Mean and median |HR − μ|, and mean σ, as a function of distance to the
tile border, 0 to 64 px, pooled and per unit. Show where the edge effect decays to the interior level.

**B5. σ behaviour.** σ distribution, fraction of pixels at any clamp, and σ versus distance to border.
Test whether σ inflates where μ blows up (NLL mean-abandonment signature): Spearman(σ, |HR − μ|) in the rim
versus the interior.

---

## Phase C — Fixes that need no retraining

Use the **reproduction weights** (fold k scores only its held-out unit). Every result is reported under the
original evaluation **and** each fix, side by side, so nothing is replaced silently.

**C1. Overlapping-tile inference with center crop.** Re-run inference on each held-out unit's full harmonized
extent. Use overlapping 256-px windows with stride = 256 − 2·rim, where rim ≥ max(32 px, effective receptive
half-width from B2). Keep only each window's central region and assemble full-unit μ and σ mosaics. Then
re-cut these mosaics onto the same 256-px patch footprints the paper used, so patch-level comparisons are
like-for-like.

**C2. Sensitivity diagnostics (label these DIAGNOSTIC, not candidate fixes).** The weights were trained under
the original conditions, so these measure sensitivity, not a fair model:
- (a) reflect padding at inference instead of zero padding;
- (b) if B1 finds HR-derived normalization, the same inference with LR-only statistics substituted.

**C3. How well does the model actually predict the seafloor?** Answer this directly, per unit and pooled,
interior pixels only (rim from B2), **in meters and in normalized units**, after C1:

| Prediction | RMS error vs HR | median abs error | RMS after removing per-patch offset |
|---|---|---|---|
| flat (patch LR mean) | | | |
| bicubic ship grid | | | |
| CNN μ, original evaluation | | | |
| CNN μ, C1 | | | |

**C4. Does σ localize error? Score every predictor on every target.**

Predictors:
- σ (original evaluation)
- σ (C1)
- roughness C / roughness-3 (frozen conventions)
- the report-44 metadata baseline (linear and GBT)

Targets, each at patch level, interior pixels only unless stated, and each in normalized units **and** meters:
- T0 — the paper's target: CNN residual, all pixels (reproduces 0.555 as a check)
- T1 — CNN residual, interior
- T2 — CNN residual, interior, per-patch offset removed
- T3 — model-free: HR − bicubic LR, interior, offset removed

Metrics:
- within-unit R² (report-36 definition: log–log after subtracting each unit's mean of both)
- per-unit Spearman ρ
- top-20% error capture

Give per-unit values and medians. Add a bootstrap-over-units 95% CI (≥ 2,000 resamples) for each σ-minus-baseline
difference. **Designate no winner and draw no framing conclusion.** Report the table; Steve interprets it.

**C5. Visual QA for Steve.** For 3 units (NA090, TN299, RR1506), write GeoTIFFs of the full-unit μ, σ,
|HR − μ| and |HR − bicubic| mosaics (original and C1) in the harmonized CRS, for inspection in QGIS. Also
produce a PNG panel of the 12 worst original-evaluation patches, with LR input, HR, μ, σ and error side by
side, annotated with HR std and offset.

---

## Report structure (`REPORT_R01_<date>.md`)

1. **Bottom line** in ≤ 10 lines. Were data defects found? Is H-norm supported? Is H-leak supported? What do
   C3 and C4 show after C1?
2. Restore verification.
3. Phase A tables. A1 and A2 must be explicit even when every count is zero.
4. Phase B pipeline table with file:line references, and the explicit leakage answer.
5. B2–B5 results.
6. C3 and C4 tables, with the original-evaluation columns alongside.
7. **Candidate fixes for a retrain (R02)**, each tied to specific defect evidence from this report. List them;
   do not rank them against the headline score.
8. Change ledger rows added.
9. Reproduction commands and output paths (scratch, repo, OAK archive).

## HOLD for Steve

- Any escalation from §0.1.
- The B1 leakage answer.
- The C3/C4 tables and the candidate-fix list, before any R02 is drafted.
