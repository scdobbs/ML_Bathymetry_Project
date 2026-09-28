# Handoff report: GRL review assessment and data preservation (2026-09-28)

**From:** execution-side Claude Code instance (fresh), Sherlock login node, repo
`/home/users/scdobbs/GAN_Bathymetry_Project`, branch `rewrite/flow-matching-2026`.
**To:** the steering instance writing directives for Steve.
**Companion document:** `grl_reviews-and-prompt-reports/REVIEW_ASSESSMENT_2026-09-28.md` (full point-by-point
assessment with per-unit tables). Scripts and outputs: `grl_reviews-and-prompt-reports/scripts/` (copies) and
`/scratch/users/scdobbs/grl_review/` (originals, also archived on OAK, see §1).

Certainty labels: CONFIRMED (checked against code or data), CONTRADICTED, COULD NOT CHECK.
Nothing below was retrained; all computations were CPU jobs on persisted artifacts under frozen conventions.

---

## 1. Data preservation: what was at risk, what was saved, where it is now

### 1.1 The risk
Scratch and group scratch purge files untouched for 90 days. Every artifact for this paper on scratch dates
from 2026-07-01 to 2026-07-10, so the purge window opens 2026-09-29 for the oldest and about 2026-10-08 for
the newest. Nothing on scratch was touched, moved or refreshed; the purge clock there is unchanged.

### 1.2 Saved today (two copies, both on OAK, both verified)

**(a) The frozen patch set.** The exact 108,833 `.npz` files behind every number in the paper.
Source `/scratch/groups/hilley/bathy_diffusion_2026/patches_native_phase2` (46 GB, last modified 2026-07-01,
purge-eligible from 2026-09-29). Now one tarball:
```
/oak/stanford/groups/hilley/auv_ship_colocated_bathy/patches_native_phase2_frozen_2026-07-01.tar
```
48.7 GB. Verification inside Slurm job 45765506: 108,833 files in source, 108,833 in tar, printed
`COUNT MATCH`. Log: `/scratch/users/scdobbs/grl_review/backup_patches_45765506.out`.
To restore: `tar -xf <tar> -C <destination>` reproduces `patches_native_phase2/{train,val,test}`.

**(b) All post-July-3 analysis directories.** `/scratch/users/scdobbs/{dir44..dir57, step0_bakeoff,
reconciliation_29a, manuscript_package, manuscript_package_2026-07-09.zip}` plus today's `grl_review/`.
These hold every per-fold results JSON, every script, the reproduction model's fold weights and per-pixel
σ/mean maps (`dir55/repro/het_folds/`), the final figure package, and the transcribed manuscript tables.
Now at:
```
/oak/stanford/groups/hilley/auv_ship_colocated_bathy/analysis_archive_2026-09-28/
```
rsync, 3.2 GB, exit 0. Directory names are preserved, so any path of the form
`/scratch/users/scdobbs/dirNN/...` maps to `.../analysis_archive_2026-09-28/dirNN/...`.

### 1.3 Never at risk (already durable)
- Production model `het_production_v1` (code + seed + LOUO per-patch σ parquet; **no weights were ever saved**):
  repo `models/het_production_v1/` (commit 29633a6), OAK `.../auv_ship_colocated_bathy/models/het_production_v1/`.
- Harmonized corpus and manifest: OAK `.../auv_ship_colocated_bathy/{harmonized,manifest}` (3.8 GB, 22 pair dirs).
- The repo itself (home is not purged).

### 1.4 Not copied
Pre-July-3 scratch directories (`bathy_diffusion_v2..v4`, `lr_grids_v3`, `global_bathy`, `more_data_2026`,
the other patch sets `patches_512_s128`, `patches_global_s256*`, `patches_colocated_phase1`). They belong to
the retired flow-model and global-synthetic work, not to this paper; some may already be past purge age.
Archive on request only.

### 1.5 Public deposits
Zenodo 10.5281/zenodo.21300922 (ML code, 40 kB) and 10.5281/zenodo.21300925 (acquisition code, 421 kB)
both resolve; **both are software deposits. No data compilation deposit exists.**

### 1.6 Environment recipe (the module Python lacks pandas; the venv has it)
```
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate
```
Frozen helper conventions: `/scratch/users/scdobbs/step0_bakeoff/step0_bakeoff.py` and
`/scratch/users/scdobbs/dir44/task3_covariates.py` (both archived).

---

## 2. What was read
Manuscript (15 pp, main text + references + captions only), Supporting Information (10 pp: Text S1 to S3,
Table S1), reviewer attachment (2 pp), `working-with-steve.md`, reports 36, 44, 45, 50, 55, and the
production-model code (`het_louo.py`, `step0_bakeoff.py`, `task3_covariates.py`, `boundary_suite.py`).

Two facts about the submission itself:
- The submitted SI cites "Text S7" (S2.4), which does not exist in the file.
- The supplementary figure/table set S1 to S7 planned in `manuscript_package/figures/FIGURE_MANIFEST_FINAL.md`
  (transfer distribution + N=15 table, reproduction ledger, expansion tables, confounder small-multiples,
  coverage curves, cal_dig section) was prepared but never submitted.

---

## 3. Bottom line on the review
The review is short but well aimed; every substantive point is correct or points at a real gap. Executing
the reviewer's two main requests (within-fold correlation; an explicit definition of the error target)
undercuts both headline claims more than the reviewer anticipated. Recommendation: major revision with
re-scoping, not a rebuttal. The reviewer "ultimately agrees" with the conclusions, which is the opening.

---

## 4. Reviewer point 1: "negative transfer R² is a normalization result". CONFIRMED, and larger than stated.

**Convention actually used** (`step0_bakeoff.louo_r2`, `task3_covariates.louo`): log-linear fit of log10
fine-band RMS on three coarse features (log10 coarse RMS, curvature, anisotropy), fit on the other units,
scored on the held-out unit with SS_tot about the held-out unit's own mean, in per-patch z-scored units,
reported as the mean over folds. Exactly the convention the reviewer describes.

**Exact protocol re-run with the requested statistics** (`review_stats.py` Part 1; reproduces the paper's
pooled 0.196 / mean −0.744 native and 0.360 / −0.593 expanded before adding columns):

| corpus | paper's mean per-fold R² | median per-fold R² | Pearson r mean (units > 0) | Spearman ρ mean (units > 0) | pred/obs scale ratio, median |
|---|---|---|---|---|---|
| native, 8 units | −0.74 | −0.03 | 0.55 (8 of 8) | 0.56 (8 of 8) | 0.53 |
| expanded, 15 units | −0.59 | +0.29 | 0.58 (14 of 15) | 0.63 (15 of 15) | 0.53 |

The means are set by EW0207 alone (−4.68 native, −7.26 expanded), the unit the SI itself flags as the
degraded-navigation outlier (Text S1.8). Per-unit tables are in the assessment document.

**Units.** The transfer target is the fine-band RMS of the per-patch z-scored HR (patch HR mean 0.000,
std 1.000, checked directly; `stats[1]` holds the std in meters). It is a dimensionless ratio, not an
amplitude in meters, and the paper calls it "RMS amplitude". Section 3.1's within-survey R² comes from a
different support (report 44 support A: raster tiles of 160 to 197 m edge, planar detrend, meters).
Neither the paper nor the SI discloses that 3.1 and 3.2 use different supports, windows and units.

Same protocol with target and features multiplied by `stats[1]` (meters):

| corpus | pooled R² normalized (paper) | pooled R² meters | per-fold raw R² mean / median, meters | Spearman ρ mean, meters (units > 0) |
|---|---|---|---|---|
| native 8 | 0.196 | 0.48 | −0.18 / −0.05 | 0.68 (8 of 8) |
| expanded 15 | 0.360 | 0.72 | −0.01 / +0.33 | 0.71 (15 of 15) |

Fine-band RMS in meters spans 0.06 m (lu11) to 3.1 m (EW0207) across units, a factor of 55.

**Consequence.** The reviewer's proposed sentence is the correct one: the absolute level (and the scale;
predictions are under-dispersed by about half) does not transfer, relative spatial structure does, on every
unit. Section 3.2, Key Point 2 ("ship data cannot predict unseen fine relief"), the abstract's "the
conditional collapses", and the Discussion's "not recoverable from ship data" are not supportable as
written. The SI already concedes the point elsewhere: Text S3.2(ii) gives roughness-3 a cross-unit
amplitude R² of +0.32. Log-space: analyses are already log10 on both supports; the paper never says so.

---

## 5. Reviewer point 2: "define the error target". CONFIRMED, with a serious consequence.

**Definition** (from `het_louo.py`, `eval_fold`): per 256-px patch, `resid_rms` = RMS over valid pixels of
(normalized HR − CNN predicted mean), full band, normalized units; σ = mean over valid pixels of the
predicted per-pixel standard deviation. "Within-pair R²" = R² of log10 σ on log10 `resid_rms` after
subtracting each pair's mean of both (report 36). None of this is in the paper or SI.

**Is it the same regression as fine-band RMS? No.** From the production parquet
(`het_louo_perpatch.parquet`, 1,763 patches): true fine-band relief is a median 0.2% of the residual
variance (IQR 0.0 to 0.7%); log σ vs log fine-band RMS has within-pair R² 0.003 with mixed-sign
correlations. That answers the reviewer, and raises the next question: what is the other 99.8%?

**Anatomy of the residual** (reproduction model's persisted per-pixel maps, `dir55/repro/het_folds/`;
the production model's per-pixel maps were never saved; the reproduction reproduces within-unit R² at
0.553 vs 0.555, so it is representative), `review_stats.py` Part 2 and `edge_followup.py`:

| quantity (median over 1,763 patches) | value |
|---|---|
| residual RMS, all valid pixels (normalized units) | 3.12 |
| median absolute pixel error | 0.27 |
| share of squared error carried by the worst 1% of pixels | 0.54 |
| share carried by the worst 5% | 0.97 |
| worst-1% pixels within 8 px of the **patch border** | 0.97 |
| worst-1% pixels within 8 px of an interior invalid (coverage-hole) pixel | 0.00 |
| residual RMS, interior only (8-px rim excluded at borders and holes) | 0.73 |
| RMS of (HR − bicubic ship grid), model-free ship-vs-AUV disagreement, all pixels | 0.65 (0.39 offset removed) |
| RMS of (CNN mean − bicubic ship grid) | 3.15 |

The CNN mean ranges from −51 to +65 in units where the patch HR has std 1. The "total error" is a tail
statistic of a small population of blow-up pixels at the 256-px tile border (a CNN padding effect), not a
coverage-hole effect and not seafloor content. Only 41% of patches even contain an interior hole.

**Within-unit R² by target** (same 1,763 patches, same folds):

| predictor → target | within-unit R² |
|---|---|
| σ → CNN residual, all pixels (the paper's 0.555) | 0.55 |
| σ → CNN residual, errors capped at 5 | 0.85 |
| σ → median absolute error | 0.31 |
| σ → model-free ship-vs-AUV disagreement, offset removed, all pixels | 0.23 |
| roughness C → same model-free target | 0.58 |
| σ (interior) → CNN residual, interior only | 0.72 |
| roughness C → CNN residual, interior only | 0.27 |
| σ (interior) → model-free disagreement, interior, offset removed | 0.40 |
| roughness C → model-free disagreement, interior, offset removed | 0.57 |

**Reading.** The paper's "0.555 vs 0.107 for terrain heuristics" is specific to a target on which σ has
a structural advantage (same network, same border features) and which is not "where a ship map is wrong"
in a physical sense. On the model-free question, free roughness beats σ. A recoverable version of
Section 3.3 exists: with the rim excluded, σ still beats roughness on the CNN's own residual (0.72 vs
0.27), and that residual is then the same size as the ship grid's own disagreement with the AUV. Two
caveats travel with it: (i) the interior CNN residual still contains each patch's vertical ship-minus-AUV
offset (datum, tide, sound speed) in normalized units, and on the offset-removed target roughness wins
(0.57 vs 0.40), so part of σ's edge is a survey-quality signal in the reviewer's sense; (ii) these are
reproduction-model numbers, and Figures 3 and 4 cannot be recomputed on an interior target without
adopting the reproduction model (or a seeded re-run) as the model of record and disclosing that.

**One cheap computation remains before deciding:** σ vs roughness on the offset-removed CNN residual, so
both are judged on the same offset-free target with the CNN's own mean. If σ wins there, the capability
claim survives cleanly; if not, the honest claim is the two-stage one the paper already makes, with
roughness doing the within-region work as well.

---

## 6. Remaining reviewer points

- **Ship-survey-quality alternative.** Legitimate. Report 44 Task 5 metadata baseline (local slope at ship
  resolution, distance to coverage boundary, fraction valid, depth, native resolution; linear and GBT)
  scored 0.129 vs σ 0.555 but could not include sounding density, incidence angle, ping density or
  swath-overlap because the gridded products carry none of them (report 44 blind spot 5). The paper does not
  state that limitation. Given §5, the target itself is dominated by border behaviour and vertical offsets,
  which are survey-geometry effects routed through the CNN.
- **Goff (2010).** Never tested in any report. COULD NOT CHECK the paper's figures (publisher 403). Two
  answers exist without new data: (a) scale mismatch, Goff predicts kilometre-scale abyssal-hill RMS height,
  the fine band here is 2 to about 90 m wavelengths; (b) the SI's own +0.32 cross-unit roughness R² and the
  0.72 pooled R² in meters at N=15 show site-level amplitude is partly predictable from coarse data, which is
  consistent with Goff's premise. The wording, not the data, conflicts with Goff. A direct test (Goff's grid
  at 15 to 19 sites) has low power and needs the grid; offer it only if obtainable.
- **Missing processing details.** CONFIRMED. No band-pass/window definition, patch RMS recipe, spatial-block
  size (report 45 §Q1: block edge 373 to 480 m, about 1.9× the roughness-envelope correlation length),
  MB-System steps, vertical-offset handling beyond one sentence, or hyperparameters (in `het_louo.py`:
  base_ch 32, 8,000 steps, AdamW, cosine schedule, per-pair cap 2,500, no early stopping).
- **Dataset DOI.** CONFIRMED; see §1.5. Cheapest fix, most expected by the editor.
- **Line edits.** All CONFIRMED (period at line 154, "notyield" at 188, "deployment ," at 214; "The author
  declares" at 237 and "The authors declare" at 248; abstract "we", acknowledgments "I").

## 7. Problems the reviewer did not raise
1. Different supports and units for 3.1 vs 3.2 (§4).
2. `lr_native` is the grid posting; the coarse predictor band [f_lr/2, f_lr] is wavelengths at and beyond the
   ship grid's Nyquist (the Fig. 1 caption itself calls 64 m the limit at NA090), i.e. the bicubic roll-off band.
3. The production model was un-seeded and its weights discarded; the re-run reproduces aggregate skill but not
   the per-patch σ field (pooled Spearman 0.77, per-fold 0.12 to 0.99; report 55 §C). Not disclosed; the planned
   reproduction ledger (S5) was not submitted.
4. Fig. 4 hit/miss maps rank patches by a statistic set by a handful of border pixels.
5. Section 3.2 dismisses pooled R² as "shared-mean structure" while 3.3's 0.555 is itself a pooled within-pair
   statistic; one vocabulary, defined once, is needed.

---

## 8. Recommendation
Major revision. Rewrite 3.2 around the per-fold table (normalized and meters, N=8 and N=15), with median and
per-unit values and EW0207 marked; the title still holds (level and scale are site-specific and must be
measured; ordering transfers). For 3.3, recompute on an interior-only, offset-free target first (CPU only)
and withdraw the capability claim if σ loses there; options ranked in the assessment document §4 as
(a) interior/robust target, (b) model-free disagreement target, (c) edge-aware retrain. Deposit the corpus
with a DOI; submit the prepared S1 to S7 set; add the processing text and the reproduction ledger. Do not
answer the correlation request with a table while keeping the "collapse" language; the numbers contradict
the words on every unit.

## 9. Decisions reserved for Steve
1. Framing of 3.2 (accept the reviewer's sentence, or a stronger "predictable to within a site-specific factor").
2. Which 3.3 target and which model of record (production parquet vs reproduction weights).
3. Whether to archive the pre-July-3 scratch directories (§1.4).
4. Whether to run the Goff site-level test.

## 10. Reproduction
```
sbatch /scratch/users/scdobbs/grl_review/run.sbatch        # Part 1 + Part 2, ~4 min CPU
sbatch /scratch/users/scdobbs/grl_review/run_edge.sbatch   # edge follow-up, ~9 min CPU (slower if group scratch is busy)
```
Expected: Part 1 prints pooled 0.196 / mean −0.744 (native8/base/normalized) and 0.360 / −0.593
(expanded15/base/normalized); Part 2 prints within_unit_r2_logsigma_vs.resid_rms = 0.553. Outputs:
`/scratch/users/scdobbs/grl_review/{results.json,edge_followup.json}`, copies in
`grl_reviews-and-prompt-reports/scripts/` and on OAK.
