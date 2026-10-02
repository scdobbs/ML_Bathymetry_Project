# Check A — differential control re-measured (native 7 units)

## 1. Reproduction of the submitted numbers (from het_production_v1 persisted outputs)

| set | n | cross-pair pooled R² σ~total error | cross-pair pooled R² σ~fine | within-pair R² σ~total (by pair / by unit) | within-pair R² σ~fine (by pair / by unit) | median unit ρ σ~total / σ~fine |
|---|---|---|---|---|---|---|
| 8 units (paper set) | 1763 | 0.460 | 0.00001 | 0.555 / 0.559 | 0.003 / 0.000 | 0.661 / -0.081 |
| native 7 (EW0207 excluded) | 1563 | 0.469 | 0.00255 | 0.582 / 0.585 | 0.001 / 0.002 | 0.618 / -0.200 |

Recorded in het_louo_summary.json: {'r2_sigma_vs_resid': 0.45979273967695955, 'r2_sigma_vs_hrfine': 1.2077276716704333e-05, 'n_patches': 1763, 'n_folds_done': 8, 'compare_to_regressor_r2': 0.06}. 'Within-site total-error localization reaches R2 = 0.555' (manuscript line 161-162); abstract 'error-localization R2 = 0.555 versus 0.107'; the directive's '0.53 / 0.005-0.06' are not verbatim in the manuscript: 0.555 is the within-pair number, 0.46 the pooled cross-pair number in het_louo_summary.json, 1.2e-5 (8-fold) / 0.005 (report-34 6-fold) the cross-pair sigma~fine, 0.003 the within-pair sigma~fine (REVIEW_ASSESSMENT §2.4), 0.06 the fine-band transfer regressor (summary compare_to_regressor_r2).

Alignment: 1763 rows aligned positionally; verification per pair (corr of log C x hr_std vs log C_R02; n_valid-mismatch rows; p99 relative mismatch): NA090__MGDS_31212 0.969 / 0 / 0.000; NA080__MGDS_31290 0.945 / 17 / 0.011; TN399__MGDS_30373 0.917 / 29 / 0.103; RR1506__MGDS_29779 0.981 / 10 / 0.006; AT37-13__MGDS_31199 0.931 / 77 / 0.025; EW0207__MGDS_32556 0.878 / 16 / 0.092; FK181031__MGDS_24618 0.933 / 0 / 0.000; TN299__MGDS_27339 0.910 / 116 / 0.123; TN299__MGDS_31253 0.939 / 4 / 0.023; failures {}; native-7 rows 1563, of which 662 fall on the stride-128 R04 grid.

## 2. Models x targets (native 7 units; paper metric = within-pair log-log R², unit-demeaned pooled; §2 metrics = median within-unit Spearman ρ / median within-unit log-log R² / median top-20 % capture; n rows)

| model | target | paper metric: within-pair R² | median ρ | median log-log R² | median capture20 | pooled ρ | n |
|---|---|---|---|---|---|---|---|
| submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | paper total error resid_rms (AUV-normalized, 256 patch) | 0.585 | 0.618 | nan | 0.527 | 0.637 | 1563 |
| submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | paper fine-band amplitude (AUV-normalized, 256 patch) | 0.002 | -0.200 | nan | 0.236 | -0.235 | 1563 |
| submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | A_fine tile 128 (m, R04) | 0.099 | -0.213 | nan | 0.014 | -0.129 | 662 |
| submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | A_fine F_hr RMS over the 256 window (m, R04) | 0.100 | -0.310 | nan | 0.018 | -0.142 | 662 |
| submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | A_fine R03 convention, 256 window (m) | 0.124 | -0.347 | nan | 0.052 | -0.163 | 662 |
| submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | T-ship (R02 definition, m, 256 mosaic tile) | 0.004 | -0.055 | nan | 0.347 | -0.109 | 1563 |
| submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | paper total error resid_rms (AUV-normalized, 256 patch) | 0.106 | 0.273 | nan | 0.295 | 0.206 | 1563 |
| submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | paper fine-band amplitude (AUV-normalized, 256 patch) | 0.013 | -0.234 | nan | 0.128 | -0.296 | 1563 |
| submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | A_fine tile 128 (m, R04) | 0.141 | 0.334 | nan | 0.487 | 0.664 | 662 |
| submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | A_fine F_hr RMS over the 256 window (m, R04) | 0.151 | 0.250 | nan | 0.557 | 0.679 | 662 |
| submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | A_fine R03 convention, 256 window (m) | 0.175 | 0.432 | nan | 0.470 | 0.698 | 662 |
| submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | T-ship (R02 definition, m, 256 mosaic tile) | 0.380 | 0.681 | nan | 0.521 | 0.688 | 1563 |
| R02 clean sigma_m (mosaic, m, per 256 tile) | paper total error resid_rms (AUV-normalized, 256 patch) | 0.002 | 0.048 | nan | 0.287 | 0.079 | 1563 |
| R02 clean sigma_m (mosaic, m, per 256 tile) | paper fine-band amplitude (AUV-normalized, 256 patch) | 0.029 | -0.201 | nan | 0.165 | -0.028 | 1563 |
| R02 clean sigma_m (mosaic, m, per 256 tile) | A_fine tile 128 (m, R04) | 0.406 | 0.468 | nan | 0.647 | 0.763 | 662 |
| R02 clean sigma_m (mosaic, m, per 256 tile) | A_fine F_hr RMS over the 256 window (m, R04) | 0.458 | 0.538 | nan | 0.761 | 0.787 | 662 |
| R02 clean sigma_m (mosaic, m, per 256 tile) | A_fine R03 convention, 256 window (m) | 0.466 | 0.439 | nan | 0.669 | 0.783 | 662 |
| R02 clean sigma_m (mosaic, m, per 256 tile) | T-ship (R02 definition, m, 256 mosaic tile) | 0.799 | 0.846 | nan | 0.755 | 0.911 | 1563 |
| R02 clean sigma_m (mosaic, m, per 128 tile) | paper total error resid_rms (AUV-normalized, 256 patch) | 0.000 | -0.057 | nan | 0.230 | 0.197 | 662 |
| R02 clean sigma_m (mosaic, m, per 128 tile) | paper fine-band amplitude (AUV-normalized, 256 patch) | 0.009 | -0.247 | nan | 0.202 | -0.171 | 662 |
| R02 clean sigma_m (mosaic, m, per 128 tile) | A_fine tile 128 (m, R04) | 0.419 | 0.496 | nan | 0.647 | 0.768 | 662 |
| R02 clean sigma_m (mosaic, m, per 128 tile) | A_fine F_hr RMS over the 256 window (m, R04) | 0.434 | 0.531 | nan | 0.761 | 0.781 | 662 |
| R02 clean sigma_m (mosaic, m, per 128 tile) | A_fine R03 convention, 256 window (m) | 0.442 | 0.460 | nan | 0.669 | 0.779 | 662 |
| R02 clean sigma_m (mosaic, m, per 128 tile) | T-ship (R02 definition, m, 256 mosaic tile) | 0.699 | 0.833 | nan | 0.706 | 0.903 | 662 |

## 3. Verdict (numbers; one sentence in the report)

Submitted σ (paper's units): within-pair R² 0.585 on its own total error vs 0.002 on the paper's fine-band amplitude, and 0.004 on T-ship (m) vs 0.099 on A_fine tile (m); submitted σ in meters: 0.380 on T-ship vs 0.141 on A_fine tile (0.151 on the 256 window); R02 clean σ_m: 0.799 on T-ship vs 0.406 on A_fine tile (0.458 on the 256 window); median within-unit ρ: R02 σ_m 0.85 (T-ship) vs 0.47 (A_fine tile); submitted σ in m 0.68 vs 0.33.
