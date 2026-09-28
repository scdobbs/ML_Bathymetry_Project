# Change ledger (post-review, R-series)

One row per deviation from the submitted-paper conventions (2026GL125574). Evaluation-only changes are logged
too. Append-only; never edit a prior row.

| # | Date | Directive | What changed | Scope (model / evaluation / data / text) | Defect evidence motivating it | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-09-28 | R01b §0 (Steve's decision of record 3) | EW0207 excluded from training folds and from headline numbers; reported separately everywhere | evaluation, data (unit selection) | REPORT_R01a §A7: 3,601 of 10,917 EW0207 patches (33%) fail coarse-scale ship-vs-AUV agreement (corr < 0.5; p5 = −0.51); vertical offset IQR 13.6 m with no spatial structure; degraded USBL-era navigation (SI Text S1.8); it alone sets the paper's mean per-fold transfer R² (−4.68 native, −7.26 expanded) | decided |
