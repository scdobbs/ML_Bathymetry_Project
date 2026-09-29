# R02 — Directive: clean-pipeline retrain of the heteroscedastic CNN, and a fair evaluation

**Series:** post-review (R-series).
**Issued:** 2026-09-28 (revised after REPORT_R01b; §2.0 values set by Steve)
**Issued by:** steering/assessment instance, on Steve's rulings after REPORT_R01b
**Executor:** Claude Code on Sherlock (fresh instance). Read, in order: R01, REPORT_R01a, R01b, REPORT_R01b,
`CHANGE_LEDGER.md`, then `het_louo.py` and `preprocessing/extract_colocated_patches.py`.
**Repo:** `/home/users/scdobbs/GAN_Bathymetry_Project`, branch `rewrite/flow-matching-2026`
**Report:** `grl_reviews-and-prompt-reports/REPORT_R02_<date>.md`, append-only (parts R02a, R02b… allowed)
**Arbiter:** Steve.

> **Start gate: satisfied.** REPORT_R01b exists and Steve has set the §2.0 values (rim 48 px, standard NLL,
> GroupNorm replaced).
>
> **Challenge clause.** Steve has asked that the executor push back if it finds an error in the steering
> instance's reasoning. Before building anything, read §2.0–2.1 against REPORT_R01b. If any ruling there
> rests on a wrong premise (for example, the claim that per-pixel channel LayerNorm removes spatial coupling,
> or that 48 px is a sufficient rim for the new architecture), **stop and write a short R02-pre note to
> Steve** stating the error, the evidence, and your proposed alternative. Do not silently substitute your own
> choice. If you find no error, say so in one line in the report and proceed.

---

## 0. Purpose

R01a confirmed that the submitted model's only input was the ship grid normalized by the **AUV** patch's mean
and standard deviation (`extract_colocated_patches.py:305–314`, read unchanged at `het_louo.py:200`). That
leaks AUV information at every "unseen" site. The same normalization turned meter-scale ship–AUV offsets
into inputs of order 10–300 on low-relief patches, which met zero padding at every tile border.

R02 retrains the same model family on a pipeline where **nothing derived from the AUV grid reaches the
network at evaluation**, fixes the defects R01a/R01b documented, and evaluates it fairly against free
baselines. This is an improved model, not a re-run. The revision will present it as the corrected model,
with the defects of the original disclosed.

## 0.1 Standing constraints

- **No AUV-derived quantity in any model input, at training or evaluation.** Enforced by the leak test (§1.3),
  which must pass before any training.
- **Read-only** on OAK canonical data, the original patch set and tarball, the production model, the
  reproduction weights and every prior report. New patches and outputs go to
  `/scratch/users/scdobbs/grl_review/R02/`.
- **GPU gate: 4 GPU-hours total** (preprocessing is CPU). Estimate before launching; stop and report if over.
- **One recipe, one run.** Train exactly the recipe in §2. Any second training run — a variant, an ablation,
  a tweak after seeing results — is a new directive (R03) with a ledger entry. Report everything computed;
  do not omit a metric or target because it looks unfavourable.
- **Holdouts untouched.** The two permanent never-trained holdout units are not trained on or scored in R02.
- **EW0207** is excluded from training and from headline numbers, and scored separately (§3.5).
- **Change ledger.** One row per change from the submitted pipeline, each citing its R01a/R01b evidence.
- Append-only; escalate rather than absorb; certainty labels (CONFIRMED / CONTRADICTED / COULD NOT CHECK).

---

## 1. Clean patch set

### 1.1 Construction

Write a new preprocessing path (a new function or flag; **do not modify the code path that produced
`patches_native_phase2`**). Build from the harmonized grids on OAK, same pairs, same 256-px HR tiles, same
strides (64 px train, 128 px eval), same valid-fraction rejections, same tile footprints wherever possible
so that the evaluation patches match the original ones. Per tile, working in **meters**:

| Quantity | Definition | Uses AUV? |
|---|---|---|
| `lr_raw_m` | ship grid on the tile's LR cell grid (area-average, as before) | no |
| `lr_valid` | ship validity on the LR cell grid | no |
| `mu_lr` | mean of `lr_raw_m` over LR-valid cells | no |
| `s_lr` | max(std of `lr_raw_m` over LR-valid cells, **0.5 m**) | no |
| LR fill | invalid LR cells ← `mu_lr` (**not** the HR mean) before upsampling | no |
| `lr_up_m` | bicubic of the filled LR grid onto the HR grid | no |
| **input ch. 1** | `x = (lr_up_m − mu_lr) / s_lr` | no |
| **input ch. 2** | `lr_valid`, nearest-upsampled to the HR grid | no |
| `delta` | per-tile offset = median(HR − `lr_up_m`) over joint-valid pixels | **label only** |
| **target** | `y = (HR − delta − mu_lr) / s_lr` on HR-valid pixels | **label only** |
| `stats` | store `mu_lr`, `s_lr`, `delta`, HR mean, HR std (meters), k, `lr_native`, `hr_res` | stored, not input |

`delta` removes the vertical ship–AUV offset (datum, tide, sound speed) from the target. The product is
seafloor **shape** relative to the ship surface, not absolute depth. Every evaluation below is offset-removed
for the same reason. Record the fraction of tiles where `s_lr` hit the 0.5 m floor, per unit.

Keep the `scale` conditioning vector unchanged (grid spacings from the manifest; not a leak, per R01a).

### 1.2 Integrity checks (report, as R01a Phase A)

Zero NaN/inf/sentinels in valid pixels; masks binary; tile counts per unit and split compared with the
original set; distribution of max |x| per unit, which should now be far below R01a's (report p50, p95, max).

### 1.3 Leak test (must pass before training)

For 100 random tiles across all units, regenerate the inputs twice: once from the true HR grid, once with
the HR grid replaced by random noise of the same shape and valid mask. **Input channels 1 and 2 and the
`scale` vector must be bit-identical between the two.** Also assert in code that the evaluation data loader
never reads `y`, `delta`, HR mean or HR std into the model's forward pass. Any failure is an escalation;
stop.

---

## 2. Model and training

### 2.0 Values set by Steve from REPORT_R01b

- **Rim width: `RIM = 48 px`** (stride 160). R01b measured the border decay on the old architecture at 28 px
  pooled and up to 40 px per unit (AT37-13). 48 px adds margin, because the normalization change below
  alters the receptive field. R02 must re-measure the border error profile on the new model (§3.1a).
- **Loss: standard Gaussian NLL.** R01b B5: interior partial ρ = 0.09 (not material); the rim signal was the
  zero-padding artifact.
- **Normalization: replace `GroupNorm` with per-pixel LayerNorm over the channel axis only** (normalizes
  each pixel's channel vector; no pooling over space). R01b B2: GroupNorm's per-tile statistics made the
  effective receptive field tile-wide (23–27% of gradient mass beyond 64 px; 99% radius 165 px), so a rim
  could not isolate the border and overlapping windows would disagree. With channel-only normalization the
  receptive field should be the convolutional one (theoretical 119 px, half-width ≈ 60 px).

### 2.1 Recipe

Same architecture, width and training schedule as `het_louo.py` (base_ch 32, FiLM scale conditioning,
residual mean `mean = x + head_mean(h)`, zero-initialized heads, 8,000 steps, batch 32, AdamW 2e-4,
wd 1e-4, cosine to 1e-6, grad-clip 1.0, bf16, per-pair cap 2,500, no augmentation), except:

| Change | Evidence |
|---|---|
| Inputs and target as §1.1 (ship-only normalization, LR-level fill, offset removed from target) | R01a B1, A4, A5 |
| `in_ch = 2` (validity channel added) | R01a B1 fill row |
| **Reflect padding** on every conv (`padding_mode='reflect'`) | R01a B1 padding row; R01b B3/B4 |
| **`GroupNorm` → per-pixel channel LayerNorm** in every block (FiLM conditioning unchanged, applied after the norm as before) | R01b B2 |
| log-variance clamp widened to [−14, 8] and the fraction of pixels at either clamp reported | R01a B1 σ row (σ at the clamp) |
| Loss: standard Gaussian NLL, masked as before | R01b B5 |
| EW0207 excluded from all training folds | R01a A7 |
| **Fixed seed** (record it), deterministic flags where available; **save every fold's weights** | report 55 §C (production un-seeded, weights discarded) |

### 2.2 Folds

Leave-one-unit-out over the native leakage units **excluding EW0207** (state the resulting fold count and
the unit list; TN299a/b stay together as one unit). Plus one additional model trained on all included native
units, used only to score EW0207 (§3.5). Each held-out unit is scored only by the fold model that did not
train on it.

### 2.3 Weights

Save to `/scratch/users/scdobbs/grl_review/R02/het_clean_r02/fold_*/` with config, seed and code hash. After
the report is written, copy to OAK `.../auv_ship_colocated_bathy/models/het_clean_r02/` (**authorized:
additive write of this new directory only**, sha256-verified) and commit to the repo under
`models/het_clean_r02/` if under 100 MB total; otherwise commit the config and manifest only.

---

## 3. Evaluation

### 3.1 Inference

For each held-out unit, infer over the **full harmonized extent** with overlapping 256-px windows, stride
256 − 2·RIM, keeping each window's central (256 − 2·RIM) region, and assemble full-unit μ and σ mosaics.
Convert to meters: μ_m = μ·s_lr + mu_lr (per window, before assembly), σ_m = σ·s_lr. Re-cut the mosaics onto
the original evaluation tile footprints (the same 177–200 tiles per pair as the paper) so patch-level
results are comparable. Report the old tile-by-tile inference as a secondary column so the effect of
mosaicking is visible.

**Pixels scored:** primary = all joint-valid pixels of the mosaic; sensitivity = joint-valid pixels ≥ 16 px
from any invalid pixel (coverage edges).

**§3.1a Border and receptive-field re-measurement (new model).** Before scoring, repeat R01b B2 (gradient
probing, 20 patches: radius holding 95% and 99% of gradient mass, mass beyond 64 px) and R01b B4 (mean
|error| and mean σ vs distance to the tile border, 0–64 px, single-tile inference) on the new model. Report
whether the effective field is now local and whether the border error reaches its interior level within
48 px. If it does not, still complete the evaluation with RIM = 48, flag it in the bottom line, and state the
rim the data would support; Steve decides whether a re-evaluation at a wider rim is warranted (no retrain).
Also report the seam check: on overlapping windows, the RMS difference between the two windows' μ and σ
in their shared central strip.

### 3.2 Does the model predict the seafloor? (meters, offset removed)

Per unit, median across units, and pooled:

| Prediction of (HR − delta) | RMS error | median abs error | fine-band RMS of prediction ÷ fine-band RMS of HR |
|---|---|---|---|
| flat (`mu_lr`) | | | |
| bicubic ship grid (`lr_up_m`) | | | |
| CNN μ_m | | | |

Also: radial power spectra of HR, `lr_up_m` and μ_m per unit (as Fig. 1b), to show whether μ adds power
beyond the ship's resolution and whether that power is correlated with HR (report the coherence in the
fine band). This speaks directly to the title claim.

### 3.3 Does σ localize ship-map error? (the capability test)

**Targets** (patch level, meters, offset removed):
- **T-ship** (model-free, primary): RMS over scored pixels of (HR − delta − `lr_up_m`) — where the ship map is wrong.
- **T-cnn**: RMS of (HR − delta − μ_m) — where the CNN is wrong.

**Predictors**, all computed from ship-side data only, in meters, on the identical patches:
- σ_m (patch mean of per-pixel σ)
- roughness C and roughness-3, **recomputed in meters from `lr_up_m` only** (R01b Task 2: the originals were
  divided by the AUV patch std; anisotropy is scale-free and may be reused)
- metadata baseline (report-44 features; linear and GBT), **recomputed** per R01b Task 2: slope from `lr_up_m`
  in meters; depth = `mu_lr` (ship mean, not the AUV patch mean); fraction valid from `lr_valid` only (not the
  joint mask); `dist_edge` and `lr_native` unchanged
- **free combined baseline**: GBT on roughness + metadata together (the strongest free predictor)
- **σ + free combined**: GBT on σ plus all free features, to measure σ's incremental value

**Metrics**, per unit, median, and pooled:
- within-unit R² (report-36 definition: log–log after subtracting each unit's mean of both)
- per-unit Spearman ρ
- top-20% error capture (fraction of total error in the top 20% of patches ranked by each predictor)
- for the GBT predictors, nested LOUO so no unit's data trains its own baseline

For every σ-involving comparison (σ vs each baseline, and σ + free vs free alone), give the difference with a
bootstrap-over-units 95% CI (≥ 2,000 resamples) and the count of units where σ is better. **Report the
table; designate no winner.**

### 3.4 Calibration (descriptive)

Per unit: median σ_m / |residual| on T-cnn; empirical coverage of ±1σ and ±2σ intervals at pixel level;
fraction of σ at the clamps.

### 3.5 EW0207 (separately)

Score EW0207 with the all-units model on §3.2–3.4. Report in its own table, never pooled into headline numbers.

### 3.6 Comparison with the submitted model

One table placing the submitted paper's headline numbers (σ 0.555 vs roughness-3 0.107; top-20% capture
0.502 vs 0.442) beside the R02 numbers on T-ship and T-cnn, with a column stating which pipeline and which
target each number came from. This table is for the response letter; label it plainly.

### 3.7 Visual QA

GeoTIFF mosaics (harmonized CRS) for NA090, TN299 and RR1506: μ_m, σ_m, |HR − delta − μ_m| and
|HR − delta − `lr_up_m`|, for Steve's inspection in QGIS. Plus a PNG hit/miss panel in the style of Fig. 4
for the same units on T-ship, for σ and for the free combined baseline.

---

## 4. Report structure (`REPORT_R02_<date>.md`)

1. Bottom line (≤ 12 lines): challenge-clause outcome; leak test result; §3.1a receptive-field and border
   result; does μ beat bicubic, and is μ's fine-band power coherent with HR; does σ beat the free baselines
   on T-ship and T-cnn, with CIs; does σ add incremental value over free features.
2. §1 clean patch set: construction, integrity, leak test.
3. §2 training: folds, seed, GPU time, clamp fractions, convergence curves per fold.
4. §3.2–3.5 tables and spectra.
5. §3.6 comparison table.
6. §3.7 visual QA paths.
7. Change-ledger rows.
8. Reproduction commands; all paths (scratch, repo, OAK).

## HOLD for Steve

- Any leak-test failure (stop before training).
- The §3.3 table and §3.6 comparison, before any manuscript text is drafted.
- Any proposed second run (R03).
