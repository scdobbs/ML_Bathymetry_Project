# het_clean_r02 — R02 clean-pipeline heteroscedastic CNN (2026-09-28)

Eight `HetUNetClean` models (base_ch 32, in_ch 2, reflect padding, per-pixel channel LayerNorm, log-variance clamp
[−14, 8], standard Gaussian NLL), seed 20260928: seven leave-one-unit-out folds (`fold_<held-out unit>`) and
`fold_ALL7` (trained on all seven units; scores EW0207 only). Architecture and loader: 
`grl_reviews-and-prompt-reports/directives_and_reports/R02/scripts/het_clean.py`; data: the R02 clean patch set
(ship-only normalization; `preprocessing/extract_clean_patches.py`). Report:
`grl_reviews-and-prompt-reports/directives_and_reports/R02/REPORT_R02_2026-09-29.md`. Full directory incl. per-pixel
single-tile maps on OAK `auv_ship_colocated_bathy/models/het_clean_r02/`.
