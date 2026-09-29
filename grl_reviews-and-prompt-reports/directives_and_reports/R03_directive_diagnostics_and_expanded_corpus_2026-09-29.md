# DIRECTIVE R03: Where σ's skill comes from, the strongest free baseline, and a clean expanded corpus

**Date:** 2026-09-29
**Author:** steering instance (conversation), for Claude Code on Sherlock
**Predecessor:** `directives_and_reports/R02/REPORT_R02_2026-09-29.md`
**Report to:** `directives_and_reports/R03/REPORT_R03_<date>.md` (R01/R01b/R02 convention)
**Compute:** CPU only. **0 GPU-h authorized.** No model training of any kind in R03.
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK, as in R02. Units in meters unless marked (norm).

---

## 0. Purpose, and what this directive is not

R02 showed that the clean-pipeline σ is statistically indistinguishable from free roughness / metadata baselines on
7 native units. R03 has three jobs, and none of them is making the CNN win.

1. Diagnose *where* σ's skill comes from and where it fails.
2. Build the strongest model-free baseline that a reviewer could build, so the bar for R04 is set before any retraining.
3. Put the expanded corpus through the R02 clean pipeline so R04 can train and evaluate on it.

Nothing here proposes adoption criteria or a winner. The pre-registration for R04 is written by the steering instance
and Steve after this report lands.

**Challenge clause.** If any instruction below is wrong, underspecified, or conflicts with the code or data, say so in
§1 of the report with evidence. Do not silently work around it. If a choice is Steve's to make, stop that item and list
it as a HOLD. Do not guess.

### 0.1 Decisions recorded (Steve, 2026-09-29)

- **D1.** The 7 novel-morphology expansion units may be used for training in R04. They are therefore *not* lockbox
  units. R03 builds clean patches for them (§1) and uses them in the model-free analyses (§4).
- **D2 (steering default, flag if you see a problem).** Cal DIG I (18 pairs → 1 leakage unit) is treated the same as
  the expansion units: clean patches built and included in the model-free analyses as **one** unit. Any per-pair cap
  in later training must be per *unit* for Cal DIG, so that 18 pairs do not dominate. Steve has not yet ruled on this;
  record it as an open HOLD in the report.
- **D3.** The **2 permanent-holdout pairs** are the lockbox. See §7.
- **D4.** EW0207 keeps its R02 status: it is excluded from folds and reported separately.

---

## 1. Clean patch set for the expanded corpus (CPU)

1. Run `preprocessing/extract_clean_patches.py` unchanged on every Cal DIG pair and every expansion-unit pair. Use the
   manifest `lr_native` (the hardcoded anchor values for Cal DIG/DISCOL/CCZ/TAG where applicable), `hr_res` and `k`.
   Keep the same rejection rules. If a pair's manifest lacks a field the extractor needs, stop that pair and list it.
   Do not infer the value.
2. Run the R02 §1.2 integrity checks and the §1.3 leak test (100 tiles, sampled in proportion to tile count across the
   new units) on the new patches. Report the R02 T-style table per unit: n tiles, rejections by reason, k, lr_px, stride,
   max |x| percentiles, `s_lr` p50, fraction at the 0.5 m floor, and `delta` p50/p95.
3. **Units to flag explicitly:**
   - the vent unit with the documented weak co-registration lock;
   - any pair carrying a CONDITIONAL fine-band verdict (report the restricted upper limit it carries);
   - any unit whose `delta` p95 exceeds 10 m (EW0207-like registration behaviour).
4. Output: `patches_clean_expanded/` plus `patch_index.parquet` and `leak_test.json` under the R03 scratch root. Leave
   the R02 `patches_clean/` untouched.

---

## 2. Decomposition of σ's skill (native units, from R02 outputs; no inference)

σ_m = σ_norm × s_lr, and s_lr is itself a roughness statistic. Quantify how much of σ_m's skill is s_lr.

For both **T-ship** and **T-cnn**, in both **mosaic** and **single-tile** modes, on the 7 native units, and with the
same within-unit log–log R², Spearman ρ and top-20% capture machinery as R02 T33 (bootstrap over units, 2,000
resamples), report:

| predictor | definition |
|---|---|
| log s_lr | per-tile `s_lr` from `stats` |
| log σ_norm | tile summary of σ in normalized units, using the same tile-summary statistic R02 used for σ_m |
| log σ_m | as R02 (reproduce R02's numbers as a check) |
| σ_norm \| s_lr | within-unit partial: residual of log σ_norm after within-unit regression on log s_lr, correlated with the residual of log T after the same regression |
| s_lr + roughness-3 | nested-LOUO linear, as R02 roughness-3 but adding log s_lr |

Report per unit and the median of units. The question to answer in one sentence in the bottom line is: **what fraction
of σ_m's within-unit R² is available from log s_lr alone, and is the partial contribution of σ_norm distinguishable
from zero?**

---

## 3. What T-ship is made of (band split)

T-ship mixes relief the ship never resolved with misfit inside the band the ship did resolve (residual misregistration,
ship noise, vertical artifacts). Survey-planning value lives in the former.

Using the R02 T32B spectral conventions (f_lr = ship Nyquist on the HR grid, f_hi = min(k·f_lr, 0.45) cyc/px, the same
taper and detrend), define per tile on joint-valid pixels, with offset `delta` removed:

- **T_res** = RMS of the residual (HR − δ − bicubic) low-passed below f_lr: misfit in the ship-resolved band.
- **T_fine** = RMS of the same residual band-passed over [f_lr, f_hi]: sub-ship misfit.
- **A_fine** = RMS of HR − δ itself band-passed over [f_lr, f_hi]: true unresolved relief. It is independent of the
  ship grid's own noise.

Specify and report the filter implementation (FFT mask with the R02 taper, or equivalent). Masked pixels must be
handled exactly as in R02 T32B. If that handling makes band-limited RMS ill-defined on low-validity tiles, report the
fraction affected and use interior-16 as the sensitivity.

Report:

1. Per unit: median T_res² / T-ship² and T_fine² / T-ship², to show which band dominates T-ship.
2. For **σ_m, roughness-3, free combined GBT, s_lr alone**, the T33 metric set against **T_res, T_fine and A_fine**
   separately, on native units in mosaic mode.
3. Per unit: rank correlation between T_res and T_fine. If they rank tiles the same way, the split is academic; if
   not, say which one each predictor tracks.
4. On NA080 and FK181031, where the bicubic has more fine-band power than HR (R02 T32B), report specifically whether
   σ_m and roughness track A_fine or the ship-noise excess (bicubic fine power − HR fine power).

---

## 4. The strongest free baseline (model-free; all units)

Run on **all units with clean patches**: 7 native + Cal DIG (1 unit) + 7 expansion, with EW0207 reported separately.
Targets: T-ship, T_res, T_fine, A_fine. Folds: leave-one-unit-out over the ~15 non-EW0207 units. Any fitted
predictor uses **nested** LOUO, exactly as R02's GBT baselines. Nothing here touches the CNN.

### 4.1 Context radius

Tiles contain only 4–13 ship cells across. Any spectral statistic estimated on the tile alone is starved. For every
ship-derived feature below, compute it on a square ship-grid window centred on the tile at **four context sizes**:

- (a) tile only;
- (b) tile + 48 HR px rim, which matches the CNN's measured effective field (R02 §4.2);
- (c) 32 × 32 ship cells;
- (d) 64 × 64 ship cells.

Clip windows to ship-valid data and report the valid fraction. **The context-size dependence is itself a primary
result:** if skill rises materially from (b) to (c)/(d), the CNN's receptive field is too small to see the statistics
that matter, and R04 needs to know that.

All features are ship-only. No HR-derived quantity may enter any feature. Re-run the R02 §1.3 leak logic on the
feature extractor: replace HR with noise and verify every feature is byte-identical on 100 tiles.

### 4.2 Spectral-extrapolation predictor (physics baseline)

Self-affine seafloor models (Goff & Jordan) imply that the resolved-band spectrum predicts unresolved variance.
Implement per tile and per context size:

1. Take the native-resolution ship window (not the bicubic upsample). Remove a plane, apply the R02 taper, and compute
   the radially averaged power spectrum.
2. Fit log P = log A − β log f by least squares over [f_min, 0.8 f_Nyq,ship], where f_min is the lowest radial bin
   with at least 8 Fourier coefficients. Exclude the top 20% near Nyquist because gridding rolls off power there.
   Record A, β and the fit R².
3. Clamp β to [1, 5] and report the fraction clamped per unit.
4. Extrapolate A f^−β over [f_lr, f_hi] (converted consistently to HR cyc/px) and integrate over the annulus to get
   predicted variance. **ŝ_fine** = its square root, in meters.
5. Report ŝ_fine as a **zero-parameter** predictor, with no fitting across tiles; its skill is purely rank / log–log R².
   Also report a nested-LOUO calibrated version (log T ~ log ŝ_fine).
6. **Anisotropic variant:** fit the 2D spectrum's principal-axis ratio and report it as a feature, not as a separate
   predictor.

### 4.3 Expanded free feature bank → nested-LOUO GBT

Compute per context size:

- every R02 roughness-3 and metadata feature;
- log s_lr;
- multi-scale detrended ship std (at each context size);
- slope mean / p90 and curvature std on the ship grid;
- β, log A, the fit R² and the anisotropy ratio from §4.2;
- a **noise indicator**: ship power in the top 20% of ship frequencies relative to the fitted power-law extrapolation
  there. Excess suggests ship noise rather than relief;
- depth (|mu_lr|) and log depth / lr_native, which is a beam-footprint proxy.

Fit a GBT with the R02 hyperparameter protocol, nested LOUO. Report its skill per context size, plus its skill using
only the best context size. Report permutation importance per fold, aggregated as median rank.

### 4.4 Tables

For each target in {T-ship, T_res, T_fine, A_fine}, produce R02 T33-style tables:

- **all ~15 units:** ŝ_fine (raw), ŝ_fine (calibrated), roughness-3, metadata GBT, R02 free combined GBT, expanded
  feature-bank GBT;
- **the 7 native units only:** the same rows plus σ_m (mosaic), for a like-for-like comparison with R02.

Include Δ tables with bootstrap-over-units CIs for:

- σ_m − expanded GBT (native units);
- expanded GBT − R02 free combined GBT (all units);
- ŝ_fine − roughness-3 (all units).

Report the CI width at N = 7 and at N ≈ 15 for the same comparison type. That shows how much power the expanded corpus
buys.

---

## 5. Where σ beats roughness, and where it loses (native units; QGIS export)

σ beats roughness substantially on NA090 (ρ 0.82 vs 0.67) and AT37-13 (0.67 vs 0.48) and loses on RR1506 (0.83 vs
0.86). Find out what is different about the tiles responsible.

For NA090, AT37-13 and RR1506, on T-ship in mosaic mode:

1. Compute within-unit ranks of T-ship, σ_m and roughness C.
2. Select the **15 tiles where σ is right and roughness is wrong**: smallest |rank σ − rank T| among the 15% of tiles
   with the largest |rank roughness − rank T|. Select the 15 tiles for the reverse case the same way.
3. For each selected tile, export:
   - a GeoPackage of footprints with ranks, T, σ_m, roughness C, s_lr, T_res/T_fine and ŝ_fine as attributes;
   - small GeoTIFFs, harmonized CRS as in R02 §7: HR − δ, ship bicubic, σ_m, |HR − δ − bicubic|.
4. Add a one-page summary table of feature means (from §4.3) for the "σ wins" versus "roughness wins" groups per unit.
   **No interpretation beyond the numbers.** Steve reviews these in QGIS.

---

## 6. Raw-swath availability (audit and a one-pair pilot)

Within-cell sounding scatter (`mbgrid -M` standard-deviation and count grids) may be the most informative ship-only
signal available. Before R04 can use it, establish what exists.

1. **Audit.** For every pair (native, Cal DIG, expansion, EW0207; excluding the lockbox, §7), record:
   - whether raw or processed swath files for the ship cruise are on OAK, or retrievable from MGDS/NCEI, and where;
   - the MB-System format id;
   - the number of files and their size;
   - whether the ship grid we use was produced from those files. If it was a provider grid, say so and state its
     provenance.
   Output a table. **Do not download anything larger than 20 GB total without a HOLD.**
2. **Pilot on one native pair** that already has raw swath on OAK. Choose the smallest; name it in the report.
   - Using the MB-System v5.8.2beta06 Apptainer sandbox, grid at the pair's `lr_native` with `mbgrid -M` (mean, count,
     SD).
   - Report the RMS difference between the regridded mean surface and the ship grid used in R02, on the common valid
     area. It must be small relative to `s_lr` to be usable.
   - For that pair's R02 evaluation tiles, report the within-unit rank correlation of tile-median SD-grid value with
     T-ship, T_fine and A_fine, next to roughness C.
   - This is a feasibility number, not evidence. Label it so.
3. Report the wall time for the pilot and extrapolate the time to regrid every pair that has raw data.

---

## 7. Lockbox (the 2 permanent-holdout pairs)

- Identify the two pairs from the manifest. Record only their pair IDs and the sha256 of their manifest rows in the
  report.
- **No R03 script may open, read, extract, grid or summarize their HR or LR rasters.** Add an explicit guard to every
  new script: a deny-list checked against the input path that raises before any read. Report `grep` evidence that the
  guard is present in every new script.
- If any R03 step would need them (for example, a manifest-wide loop), exclude them and say so.

---

## 8. Report requirements

- **§1 Bottom line**, in at most 10 numbered points, covering:
  - challenge-clause findings;
  - expanded-corpus integrity and leak test;
  - the s_lr fraction of σ's skill (§2);
  - which band dominates T-ship (§3);
  - ŝ_fine vs roughness-3 vs expanded GBT on all units (§4);
  - context-size dependence (§4.1);
  - σ_m vs expanded GBT on native units (§4.4);
  - CI width at N = 7 vs N ≈ 15;
  - raw-swath feasibility (§6).
- **HOLDs listed explicitly.** D2 is open by default.
- Every table's source file and job ID.
- Scripts and sbatch files as run are committed to the repo under `directives_and_reports/R03/scripts/`. Large outputs
  stay on scratch; list their paths.
- Append rows to `CHANGE_LEDGER.md` only if a pipeline quantity changes. §1 should change none: the extractor runs
  unchanged.

## 9. Out of scope for R03

- Any training, fine-tuning, or architecture change.
- Any GPU use.
- Any adoption criterion, winner designation, or manuscript text.
- Any touch of the lockbox pairs.
- Regridding more than the one pilot pair.
