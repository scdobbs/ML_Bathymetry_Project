# Assessment of the GRL review of "Machines cannot invent the seafloor, but can tell us where to look"

**Prepared:** 2026-09-28, fresh Claude Code instance, for S.C. Dobbs.
**Read in full:** manuscript (15 pp), Supporting Information (10 pp), reviewer attachment (2 pp),
reports 36, 44, 45, 50, 55 and the production-model code. All new numbers below were computed today from
persisted artifacts on frozen conventions; scripts and outputs are in `grl_reviews-and-prompt-reports/scripts/`
and `/scratch/users/scdobbs/grl_review/`.

Certainty labels used: CONFIRMED (checked against code or data), CONTRADICTED, COULD NOT CHECK.

---

## 0. Bottom line

The review is short but well aimed. Every substantive point is either correct or points at a real gap.
Executing the reviewer's two requests (within-fold correlation, and an explicit definition of the error
target) produces results that weaken both headline claims more than the reviewer anticipated:

1. **The cross-site "collapse" of the fine-band conditional (Section 3.2, Key Point 2, abstract) is a
   statistic-choice result.** The per-fold R² is computed against the held-out unit's own mean, in
   per-patch-normalized units, and reported as a mean that one degraded-navigation unit (EW0207) dominates.
   Within-fold Spearman correlation is positive on 8 of 8 native units and 15 of 15 expanded units. In
   meters, the pooled cross-site log-log R² at N=15 is 0.72.
2. **The error that σ localizes (Section 3.3, Key Point 3, Figs 3 and 4) is dominated by extreme pixels
   at mask and patch edges, produced by the CNN's own mean.** On the reproduction model, which reproduces
   the paper's within-unit R² (0.553 versus 0.555), the CNN mean is about five times worse than the plain
   bicubic ship grid, and the worst 1% of pixels carry a median 54% of the squared error, 94% of them within
   8 px of an edge. On the model-free question "where does the ship grid disagree with the AUV grid", the
   free roughness heuristic beats σ (within-unit R² 0.58 versus 0.23).

Both findings are recoverable in a revision, but not by adding statistics to the current text. Section 3.2
needs re-scoping and Section 3.3 needs a redefined target. Details and numbers follow.

---

## 1. Where everything is, and what was at risk today

| Item | Location | Status |
|---|---|---|
| Production model (code + seed + LOUO σ parquet; no weights ever saved) | repo `models/het_production_v1/` (commit 29633a6); OAK `auv_ship_colocated_bathy/models/het_production_v1/`; scratch `dir54/protect/` | durable |
| Loadable reproduction weights + per-pixel σ and mean maps | `/scratch/users/scdobbs/dir55/repro/het_folds/` | copied to OAK today |
| Harmonized corpus (22 pair directories) + manifest | OAK `auv_ship_colocated_bathy/{harmonized,manifest}` (3.8 GB) | durable |
| Frozen patch set used by every number in the paper | `/scratch/groups/hilley/bathy_diffusion_2026/patches_native_phase2` (46 GB, 108,831 files, last modified 2026-07-01) | purge-eligible from about 2026-09-29; tarball to OAK running as job 45765506 |
| Analysis directories dir44 to dir57, step0_bakeoff, manuscript_package | `/scratch/users/scdobbs/` (modified Jul 3 to Jul 10) | copied to OAK `analysis_archive_2026-09-28/` today (3.2 GB, exit 0) |
| Public code | Zenodo 10.5281/zenodo.21300922 (ML code, 40 kB) and 10.5281/zenodo.21300925 (acquisition code, 421 kB) | both resolve; both are software deposits |
| Public data compilation | none | the reviewer is right that no compilation DOI exists |

This repository is the right one for the model: the paper's model is the heteroscedastic CNN in
`models/het_production_v1/step0_bakeoff/het_louo.py`, not the rectified-flow model that CLAUDE.md describes.
The flow model was retired as "not load-bearing" in reports 34 to 36.

Two things the submitted SI does not contain although they were prepared: the supplementary figure and table
set S1 to S7 listed in `manuscript_package/figures/FIGURE_MANIFEST_FINAL.md` (transfer distribution and N=15
table, reproduction ledger, expansion tables, confounder small-multiples, coverage curves), and any processing
text beyond Text S1 to S3. The SI's Section S2.4 cites "Text S7", which does not exist in the submitted file.

Verify the patch tarball before relying on it:

```
tail -3 /scratch/users/scdobbs/grl_review/backup_patches_45765506.out   # expect "COUNT MATCH"
```

---

## 2. The review, point by point

### 2.1 Dataset release and missing compilation DOI. CONFIRMED, actionable.
Open Research lists two Zenodo DOIs; both are code deposits. The harmonized corpus (3.8 GB on OAK) has never
been deposited. Table S1 has per-survey DOIs but no compilation DOI. This is the cheapest item to fix and
the one the editor will most expect fixed.

### 2.2 "Negative R² against the held-out unit's own mean is a normalization result." CONFIRMED, and larger than stated.

What the paper computes (`step0_bakeoff.louo_r2` and `dir44/task3_covariates.louo`): a log-linear fit of
log10 fine-band RMS on three coarse features, fit on the other units, scored on the held-out unit with
SS_tot about the held-out unit's own mean. That is exactly the convention the reviewer describes. The
paper reports the mean over folds.

Per-fold values behind the paper's "mean per-fold R² = −0.74" (native, 8 units) and "−0.59" (N=15):

| Native unit | raw R² | R² after removing per-unit mean | Pearson r | Spearman ρ | prediction scale / obs scale |
|---|---|---|---|---|---|
| NA090 | 0.47 | 0.49 | 0.83 | 0.82 | 0.38 |
| NA080 | −0.89 | −0.08 | 0.41 | 0.39 | 0.91 |
| TN399 | 0.11 | 0.25 | 0.52 | 0.53 | 0.67 |
| RR1506 | 0.06 | 0.25 | 0.61 | 0.69 | 0.26 |
| AT37-13 | −0.12 | 0.23 | 0.49 | 0.52 | 0.39 |
| EW0207 | −4.68 | −0.03 | 0.53 | 0.52 | 1.08 |
| FK181031 | −1.32 | −1.10 | 0.23 | 0.28 | 1.30 |
| TN299 | 0.42 | 0.44 | 0.78 | 0.75 | 0.37 |
| **mean / median** | **−0.74 / −0.03** | 0.06 / 0.24 | 0.55 / 0.52 | 0.56 / 0.53 | 0.53 |

Positive Pearson and Spearman on 8 of 8 units. The mean raw R² is set by EW0207, which the SI itself
flags as the degraded-navigation outlier (Text S1.8). The median raw R² is −0.03.

At N=15 (same protocol, expanded units), raw per-fold R² is positive on 9 of 15 units, median +0.29;
Spearman ρ is positive on 15 of 15, median 0.75; the mean of −0.59 is set by EW0207 at −7.26.

So the reviewer's proposed sentence is the correct one: the absolute level (and, per the last column,
the scale, predictions are under-dispersed by about half) does not transfer; relative spatial structure
does, on every unit. This is also what Sections 3.3 and 3.4 already say about σ ("ranking generalizes").
Key Point 2 ("ship data cannot predict unseen fine relief"), the abstract's "the conditional collapses",
and the Discussion's "for any practical predictor, fine-scale structure at an unvisited site is not
recoverable" are not supported in that form. The SI already concedes the point in a different place:
Text S3.2(ii) reports a cross-unit amplitude R² of +0.32 for the roughness heuristic.

### 2.3 Linear versus log space. Partly CONTRADICTED, partly a bigger problem.

The analyses are already in log10 space on both supports (report 44 §0.2). The paper never says so;
it should. The larger issue the reviewer's question exposes is the unit of the target. The transfer
regression (support B) uses the patch files, in which HR and LR are z-scored by each 256-px patch's own
HR mean and standard deviation (checked directly: patch HR mean 0.000, std 1.000; `stats[1]` holds the
std in meters). The "fine-band RMS amplitude" in Section 3.2 is therefore a dimensionless ratio, fine-band
RMS divided by the patch's total relief, not an amplitude in meters. Section 3.1's within-survey R² comes
from a different support (report 44 support A: raster tiles of 160 to 197 m edge, planar detrend, meters).
Sections 3.1 and 3.2 are compared across different supports, different windows and different units, and
neither the text nor the SI discloses this.

Re-running the exact transfer protocol with the target and features multiplied by `stats[1]` (meters):

| corpus | units | pooled R² normalized (paper) | pooled R² meters | per-fold raw R² mean / median, meters | Spearman ρ mean, meters (positive) |
|---|---|---|---|---|---|
| native | 8 | 0.196 | 0.48 | −0.18 / −0.05 | 0.68 (8 of 8) |
| expanded | 15 | 0.360 | 0.72 | −0.01 / +0.33 | 0.71 (15 of 15) |

Read carefully: the meters pooled R² is high partly because between-site amplitude differences are large
in meters and coarse-band amplitude tracks them (spectral continuity). That is cross-site skill for the
absolute amplitude, which is what Goff-style prediction and the Seabed 2030 framing care about. The
per-fold level still fails at some units (TN399, EW0207, NA080, FK181031). The honest statement is
"predictable to within a site-specific factor", not "collapses".

The fine-band RMS in meters spans 0.06 m (lu11 margin) to 3.1 m (EW0207), a factor of 55 across units,
which is why the units matter.

### 2.4 "I am confused about the error target." CONFIRMED. The definition, and what it turns out to contain.

Definition (from `het_louo.py`, `eval_fold`): per 256-px patch, `resid_rms` = RMS over valid pixels of
(normalized HR − CNN predicted mean), full band, in per-patch normalized units; σ = mean over valid pixels
of the predicted per-pixel standard deviation. "Within-pair R²" = R² of log10 σ on log10 `resid_rms` after
subtracting each pair's mean of both (report 36). None of this is in the paper or SI.

Is it the same regression as fine-band RMS? No. From the production parquet, the true fine-band relief
accounts for a median 0.2% of the residual variance (IQR 0.0 to 0.7%), and log σ against log fine-band
RMS has within-pair R² 0.003 with mixed-sign correlations across pairs. So the reviewer's worry is
answered, but the answer raises the next question: what is the other 99.8%?

Anatomy of the residual, computed from the persisted per-pixel maps of the reproduction model (dir55;
the production model's per-pixel maps were never saved; the reproduction reproduces the within-unit R²
at 0.553, so it is representative), 1,763 patches, 8 units:

| quantity (median over patches) | value |
|---|---|
| residual RMS (normalized units) | 3.12 |
| median absolute pixel error | 0.27 |
| residual RMS with pixel errors capped at 5 | 0.93 |
| share of squared error carried by the worst 1% of pixels | 0.54 |
| share carried by the worst 5% | 0.97 |
| distance of worst-1% pixels to nearest invalid pixel or patch edge | 4 px (all valid pixels: 38 px) |
| fraction of worst-1% pixels within 8 px of an edge | 0.94 |
| RMS of (HR − bicubic ship grid), the model-free ship-vs-AUV disagreement | 0.65 (0.39 after removing the patch's vertical offset) |
| RMS of (CNN mean − bicubic ship grid) | 3.15 |

The CNN mean output ranges from −51 to +65 in units where the patch's HR has standard deviation 1. The
"total error" is a tail statistic of a small population of blow-up pixels at coverage-mask and patch
borders. Consequences:

- σ's within-unit R² of 0.55 rises to 0.85 against the capped residual and falls to 0.31 against the
  median absolute error. σ is very good at predicting where its own network's mean blows up.
- Against the model-free ship-vs-AUV disagreement (offset removed), σ has within-unit R² 0.23, while
  coarse roughness alone has 0.58. Per unit, Spearman(σ, model-free disagreement) is positive on 4 of 8
  units and negative on 4 of 8.
- The paper's headline "0.555 versus 0.107 for terrain heuristics" is therefore specific to a target on
  which σ has a structural advantage (same network, same edge features) and which is not "where a ship
  map is wrong" in any physical sense.

A follow-up job (45767155) separates patch-border effects from interior coverage-hole effects and tests
whether σ retains skill when both rims are excluded; its output is appended in Section 6 when available.

### 2.5 Ship-survey quality as the alternative explanation. CONFIRMED as legitimate; partially pre-empted; now central.

Report 44 Task 5 built a metadata baseline (local slope at ship resolution, distance to coverage boundary,
fraction valid, depth, native resolution; linear and gradient-boosted) and found within-pair R² 0.129
against σ's 0.555. The paper cites this as "metadata". It could not include sounding density, incidence
angle, ping density or swath-overlap count because the gridded products carry none of them (report 44,
blind spot 5). That limitation is not stated in the paper. Given Section 2.4, the more important point is
that the target itself is dominated by coverage-edge behaviour, which is a survey-geometry effect in the
reviewer's sense, only routed through the CNN.

### 2.6 Goff (2010). Legitimate, never tested. COULD NOT CHECK the paper's figures directly (publisher 403).

No report tests gravity-derived roughness. Two answers are available without new data:
(a) Scale: Goff predicts abyssal-hill RMS height, a kilometre-scale statistic; the fine band here is
wavelengths of 2 to about 90 m, two orders of magnitude finer. The ship's own coarse band is a far closer
predictor of that band than gravity can be. (b) The paper's own SI (S3.2 ii) gives roughness-3 a cross-unit
amplitude R² of +0.32 over 8 units, and Section 2.3 above gives pooled log-log R² 0.72 in meters at N=15.
So site-level amplitude is partly predictable from coarse data, which is consistent with Goff's premise,
and the manuscript's current wording is what conflicts with Goff, not the data. A direct test (Goff's
predicted RMS at each of the 15 to 19 sites versus measured fine-band RMS in meters) would have low power
at that N and requires obtaining Goff's grid; I recommend answering with (a) and (b) and offering the test
as a supplement only if the grid is obtainable.

### 2.7 Missing processing details. CONFIRMED.
The SI has no band-pass or window definition, no patch RMS recipe, no spatial-block size (report 45 §Q1
has it: block edge 373 to 480 m, about 1.9 times the roughness-envelope correlation length), no MB-System
steps, and only one sentence on vertical offsets. Hyperparameters exist in `het_louo.py` (base_ch 32,
8,000 steps, AdamW, cosine schedule, per-pair cap 2,500, no early stopping) and are absent from the SI.

### 2.8 Line edits. CONFIRMED (period at line 154, "notyield" at 188, "deployment ," at 214; "The author
declares" at line 237 and "The authors declare" at line 248; abstract uses "we", acknowledgments "I").

---

## 3. Problems the reviewer did not raise

1. **Different supports for 3.1 and 3.2** (Section 2.3 above). The within-survey conditional (meters,
   tiles, planar detrend) and the transfer test (normalized 256-px patches) are not the same measurement.
   A referee who asks for the SI details will find this.
2. **Native resolution versus Nyquist.** `lr_native` is the ship grid posting (e.g. 32 m at NA090). The
   coarse predictor band [f_lr/2, f_lr] is wavelengths 64 to 32 m, at and beyond the ship grid's Nyquist
   wavelength of 64 m, which the Fig. 1 caption itself identifies as the ship's limit. The "coarse band"
   is the bicubic roll-off band, not "structure just within the ship's resolving power".
3. **Reproducibility of the production model.** The production run was un-seeded and its weights were
   discarded. The byte-faithful re-run reproduces aggregate skill but not the per-patch σ field (pooled
   Spearman 0.77, per-fold 0.12 to 0.99; report 55 §C). The planned reproduction ledger (S5) is not in the
   submitted SI. The paper does not disclose this.
4. **The tail-dominated target also drives Fig. 4.** Hit/miss maps at the top 20% rank patches by a
   statistic that a handful of edge pixels set. The maps look like content because they are at patch
   granularity.
5. **Pooled versus per-fold inconsistency across sections.** Section 3.2 dismisses pooled R² (0.20 to
   0.36) as "shared-mean structure" while Section 3.3's 0.555 is itself a pooled within-pair statistic.
   Both conventions are legitimate; the paper must use one vocabulary and define it.

---

## 4. Recommendation

Treat this as a major revision, not a rebuttal. Reasoning: items 2.2 and 2.4 are not matters of emphasis.
The reviewer "ultimately agrees" with the conclusions, which is the opening to re-scope rather than defend.

**Section 3.2 and the framing.** Replace "collapses" and "cannot predict" with the reviewer's sentence,
supported by the per-fold table above (normalized and meters, N=8 and N=15). Report median and per-unit
values, mark EW0207, and state the unit of the target. The title still holds: level and scale are
site-specific and must be measured; ordering transfers.

**Section 3.3.** Three options, in order of preference:
(a) Redefine the target as interior-only (exclude a rim at mask and patch edges) and robust (median or
capped), recompute σ versus roughness-3 versus metadata on identical patches, and report whichever wins.
Pending job 45767155 gives the first look. (b) Redefine the target as the model-free ship-vs-AUV
disagreement (bicubic residual, offset removed), on which roughness currently wins; the paper then becomes
"free roughness localizes ship-map disagreement; the CNN adds nothing", which is still publishable in
this letter's frame. (c) Retrain with edge-aware masking; this is a new model and a new set of numbers.
I recommend (a) first because it needs no GPU; if σ loses on (a), the capability claim should be withdrawn
rather than defended.

**SI and data.** Deposit the harmonized corpus (3.8 GB) with a DOI; add the S1 to S7 set already
prepared; add the processing text the reviewer lists; add the reproduction ledger.

**Do not** answer the correlation request with a table alone while keeping the "collapse" language;
the numbers contradict the words on every unit.

---

## 5. Concrete next steps (all CPU, no retraining)

1. Confirm the patch-set tarball on OAK verified (`COUNT MATCH`), then stop worrying about purge.
2. Read job 45767155's output (`/scratch/users/scdobbs/grl_review/edge_followup.json`): border versus
   hole attribution, and σ versus roughness on interior-only residuals.
3. Decide 3.3 option (a)/(b)/(c). This is the framing decision that is yours.
4. Recompute Fig. 3 and Fig. 4 on the chosen target; regenerate Fig. 2c with median and per-unit bars
   (machinery exists in dir50/dir56).
5. Draft the response letter only after step 3.

Reproduction of everything in this document:

```
sbatch /scratch/users/scdobbs/grl_review/run.sbatch        # Part 1 transfer decomposition + Part 2 anatomy, ~4 min
sbatch /scratch/users/scdobbs/grl_review/run_edge.sbatch   # edge follow-up
```
Expected: Part 1 reproduces pooled 0.196 / mean −0.744 (native) and 0.360 / −0.593 (N=15) before adding
the new columns; Part 2 reproduces within-unit R²(log σ, log resid_rms) = 0.553.

---

## 6. Follow-up job output (job 45767155, completed 14:26)

Attribution of the worst-1% pixels, 1,763 patches, reproduction model:

| quantity (median over patches) | value |
|---|---|
| patches containing any interior invalid pixel | 41% (728 of 1,763) |
| worst-1% pixels within 8 px of the patch border | 0.97 |
| worst-1% pixels within 8 px of an interior invalid pixel | 0.00 |
| residual RMS, all valid pixels | 3.12 |
| residual RMS, interior only (8-px rim excluded at borders and holes) | 0.73 |
| model-free ship-vs-AUV disagreement, interior, offset removed | 0.36 |

The blow-ups are a CNN patch-border effect (padding at the 256-px tile edge), not a coverage-hole effect.
Within-unit R² on interior-only targets:

| predictor → target | within-unit R² |
|---|---|
| σ (interior mean) → CNN residual (interior) | 0.72 |
| roughness C → CNN residual (interior) | 0.27 |
| σ (interior mean) → ship-vs-AUV disagreement (interior, offset removed) | 0.40 |
| roughness C → ship-vs-AUV disagreement (interior, offset removed) | 0.57 |

Reading. Option (a) in Section 4 is viable: with the rim excluded, σ still beats roughness by a wide
margin on the CNN's own residual (0.72 versus 0.27), and that residual is now of the same size as the
ship grid's own disagreement with the AUV, so it is a physically meaningful error. Two caveats must go in
the paper if this route is taken. First, the interior CNN residual still contains each patch's vertical
ship-minus-AUV offset (datum, tide, sound-speed residuals), expressed in normalized units; on the
offset-removed disagreement roughness beats σ (0.57 versus 0.40). The σ advantage therefore rests partly
on predicting where a constant vertical offset is large relative to local relief, which is a
survey-quality signal in exactly the reviewer's sense. Second, these are reproduction-model numbers; the
production model's per-pixel maps do not exist, so the paper's Figures 3 and 4 cannot be recomputed on
an interior target without adopting the reproduction model (or the seeded re-run) as the model of record,
which the reproduction ledger should then disclose.

Before choosing, one more cheap computation is worth running: the same table with the per-patch offset
also removed from the CNN residual, so that σ and roughness are compared on the same offset-free target
with the CNN's own mean. If σ wins there, the capability claim survives cleanly; if not, the honest
claim is the two-stage one the paper already makes, with roughness doing the within-region work too.
