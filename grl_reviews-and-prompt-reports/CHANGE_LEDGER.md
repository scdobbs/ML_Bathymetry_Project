# Change ledger (post-review, R-series)

One row per deviation from the submitted-paper conventions (2026GL125574). Evaluation-only changes are logged
too. Append-only; never edit a prior row.

| # | Date | Directive | What changed | Scope (model / evaluation / data / text) | Defect evidence motivating it | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-09-28 | R01b §0 (Steve's decision of record 3) | EW0207 excluded from training folds and from headline numbers; reported separately everywhere | evaluation, data (unit selection) | REPORT_R01a §A7: 3,601 of 10,917 EW0207 patches (33%) fail coarse-scale ship-vs-AUV agreement (corr < 0.5; p5 = −0.51); vertical offset IQR 13.6 m with no spatial structure; degraded USBL-era navigation (SI Text S1.8); it alone sets the paper's mean per-fold transfer R² (−4.68 native, −7.26 expanded) | decided |
| 2 | 2026-09-28 | R02 §1.1 | Inputs normalized by SHIP-side statistics only: x = (bicubic(LR) − mu_lr)/max(std_lr, 0.5 m); target y = (HR − delta − mu_lr)/s_lr with delta = per-tile median(HR − bicubic LR) (label only); invalid LR cells filled with mu_lr, not the HR mean | data (new patch set `R02/patches_clean`, new module `preprocessing/extract_clean_patches.py`; original path untouched) | REPORT_R01a B1 (P:305–314: AUV mean/std scaled the input), A4 (|input| > 10 on 3,275 native patches), A5 (meter-scale offsets), R01b B3 (fill step not causal but HR-mean fill is AUV-derived) | done |
| 3 | 2026-09-28 | R02 §2.1 | Second input channel = LR validity mask (in_ch = 2) | model | REPORT_R01a B1 fill row | done |
| 4 | 2026-09-28 | R02 §2.1 | Reflect padding on every conv (was zero padding) | model | REPORT_R01a B1 padding row; R01b B3/B4 (97% of worst pixels within 8 px of the tile border) | done |
| 5 | 2026-09-28 | R02 §2.0/§2.1 | GroupNorm → per-pixel LayerNorm over channels (no spatial statistics) | model | REPORT_R01b B2 (GroupNorm made the effective receptive field tile-wide: r99 165 px) | done |
| 6 | 2026-09-28 | R02 §2.1 | log-variance clamp widened from [−10, 6] to [−14, 8]; clamp fractions reported | model | REPORT_R01a B1 σ row; R01b B5 (2.9% of pixels at the upper clamp) | done |
| 7 | 2026-09-28 | R02 §2.1 | Fixed seed 20260928, cudnn deterministic, every fold's weights saved | model (training procedure) | report 55 §C (production un-seeded, weights discarded, σ field not reproducible) | done |
| 8 | 2026-09-28 | R02 §3.1 | Evaluation by overlapping-window inference over the full unit extent, RIM = 48 px, stride 160; single-tile inference kept as a secondary column | evaluation | REPORT_R01b B4 (border error decays to interior level by 28–40 px) | done |
| 9 | 2026-09-28 | R02 §3.3 | Roughness C / curvature and the metadata baseline (slope, depth, fraction valid) recomputed in meters from ship-side data only | evaluation (baselines) | REPORT_R01b Task 2 (the originals were divided by the AUV patch std; depth = AUV patch mean; fraction valid used the AUV mask) | done |
