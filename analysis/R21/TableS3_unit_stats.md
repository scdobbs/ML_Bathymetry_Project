# Table S3. Descriptive agreement per development unit on the frozen held-out predictions

Tiles: each pair's frozen scoring view (interior tiles against A_fine; the fallback view against A_fine_int for TN399__MGDS_33068, AR26__MGDS_31838 and cal_dig_morro_bay__basin_flank / 20190314m4_LuciaChica970m; the two not-scorable pairs AT42-06 and luciachica_2007 contribute no tiles). Statistics are computed within each pair; a multi-pair unit is summarized as the median over its scored pairs, as the frozen evaluation (eval_v1.1) does. rho is the Spearman correlation between the candidate and the target (equal to the frozen per-pair values). r is the Pearson correlation between log predicted and log observed amplitude after centering each on its pair mean. R2 (mean removed) = 1 - sum((obs_c - pred_c)^2) / sum(obs_c^2) on the pair-centered logs. R2 (mean and scale removed) is the same after standardizing both within the pair; it equals 2r - 1 exactly. No tile had a non-positive prediction.

| unit | class | pairs | tiles | s_lr rho | s_lr r | s_lr R2 (mean removed) | s_lr R2 (mean and scale removed) | GBT_ship rho | GBT_ship r | GBT_ship R2 (mean removed) | GBT_ship R2 (mean and scale removed) | P-V1 rho | P-V1 r | P-V1 R2 (mean removed) | P-V1 R2 (mean and scale removed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lu00 | ten | 1 | 2157 | +0.506 | +0.447 | -0.252 | -0.105 | +0.543 | +0.541 | -0.423 | +0.083 | +0.585 | +0.604 | +0.131 | +0.207 |
| lu04 | ten | 1 | 86 | +0.384 | +0.334 | -0.200 | -0.332 | +0.319 | +0.322 | -0.270 | -0.357 | +0.355 | +0.356 | -0.656 | -0.289 |
| lu05 | ten | 3 | 5186 | +0.184 | +0.162 | -1.906 | -0.677 | +0.149 | +0.187 | -2.740 | -0.626 | +0.274 | +0.286 | -0.613 | -0.429 |
| lu07 | ten | 2 | 79 | +0.488 | +0.481 | +0.078 | -0.039 | +0.535 | +0.503 | +0.027 | +0.006 | +0.557 | +0.534 | +0.160 | +0.068 |
| lu08 | ten | 1 | 969 | +0.631 | +0.630 | +0.284 | +0.259 | +0.527 | +0.607 | +0.276 | +0.214 | +0.576 | +0.632 | +0.391 | +0.265 |
| lu09 | ten | 1 | 631 | -0.085 | -0.044 | -1.617 | -1.089 | +0.100 | +0.018 | -0.330 | -0.965 | +0.190 | +0.116 | -0.386 | -0.768 |
| lu13 | ten | 2 | 4844 | +0.588 | +0.604 | +0.311 | +0.207 | +0.507 | +0.514 | +0.242 | +0.028 | +0.593 | +0.563 | +0.298 | +0.127 |
| lu14 | ten | 1 | 48 | +0.815 | +0.764 | +0.556 | +0.529 | +0.660 | +0.715 | +0.512 | +0.431 | +0.602 | +0.669 | +0.343 | +0.337 |
| lu15 | ten | 2 | 670 | +0.165 | +0.223 | -0.856 | -0.555 | +0.240 | +0.255 | -0.096 | -0.490 | +0.251 | +0.237 | -0.851 | -0.526 |
| nu00 | ten | 1 | 523 | +0.390 | +0.530 | -0.492 | +0.061 | +0.370 | +0.497 | -0.257 | -0.007 | +0.330 | +0.438 | -1.290 | -0.125 |
| lu02 | six | 1 | 214 | +0.594 | +0.668 | +0.428 | +0.335 | +0.509 | +0.650 | +0.064 | +0.300 | +0.083 | +0.152 | +0.022 | -0.696 |
| lu06 | six | 1 | 69 | +0.325 | +0.173 | -0.048 | -0.655 | +0.834 | +0.706 | +0.498 | +0.412 | +0.829 | +0.855 | +0.635 | +0.710 |
| lu10 | six | 1 | 119 | +0.343 | +0.340 | +0.012 | -0.321 | +0.169 | +0.091 | +0.007 | -0.818 | -0.094 | +0.026 | -0.045 | -0.948 |
| lu11 | six | 1 | 118 | -0.013 | +0.091 | -3.375 | -0.817 | +0.248 | +0.280 | +0.015 | -0.440 | +0.027 | +0.009 | -0.532 | -0.982 |
| lu12 | six | 18 | 1895 | +0.549 | +0.565 | +0.006 | +0.129 | +0.533 | +0.601 | +0.111 | +0.202 | +0.571 | +0.592 | -0.306 | +0.183 |
| vu02 | six | 1 | 57 | +0.654 | +0.657 | +0.358 | +0.314 | +0.690 | +0.775 | +0.594 | +0.550 | +0.658 | +0.659 | +0.289 | +0.318 |
| lu16 | EW0207 | 1 | 1621 | +0.511 | +0.552 | -0.192 | +0.103 | +0.439 | +0.499 | -0.232 | -0.002 | +0.536 | +0.548 | +0.255 | +0.095 |
