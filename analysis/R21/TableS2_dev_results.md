# Table S2. Development results per unit (frozen outputs only; nothing recomputed)

Column sources. class: R08b ruling_unit_set_2026-10-09.md. pairs, scored tiles, median A_fine: grl_review_archive/R08b/tables/interior_per_unit.md (tiles = tiles_v1.1 per unit; interior D1 count in Panel A). view: R10b/rehearsal/scores/per_pair_D1.csv (frozen scorer view per pair; the unit rho below is the R08b/R09 interior-D1 value, which uses the interior tiles of every pair). s_lr, GBT_ship, V2, P-V1 rho: R08b/tables/per_unit_ten.md and per_unit_six.md (rows ref_s_lr, GBT_ship_A_fine, V2_seedavg, P-V1_seedavg; 'all / D1'), EW0207 from R08b/tables/EW0207.md (V2 seed-averaged value not tabulated there). 95 % CI: a per-unit CI is not among the frozen outputs; for single-pair units the per-pair block-bootstrap CI of s_lr (R10b per_pair_D1.csv, seed 20261009, 2000 resamples) is given; multi-pair units: NOT FOUND. Spectral extrapolation: R08/tables/wide_per_unit.md column shat_fine_W64 (64-cell ship window, all tiles); no interior-tile value exists. Unit values are medians over the unit's eligible pairs of the within-pair Spearman rho (eval_v1.1).

## Panel A: interior tiles (D1)

| unit | class | pairs | scored tiles | view | s_lr rho [95% CI] | GBT_ship rho | V2 rho | P-V1 rho | spectral extrapolation rho | median A_fine (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| lu00 | real-relief | 1 | 2,157 | interior D1 (RR1506__MGDS_29779:interior) | +0.51 [+0.39, +0.61] | +0.54 | +0.53 | +0.59 | NOT FOUND (interior) | 0.2911 |
| lu04 | real-relief | 1 | 86 | interior D1 (AT37-13__MGDS_31199:interior) | +0.38 [+0.20, +0.59] | +0.32 | +0.37 | +0.36 | NOT FOUND (interior) | 0.5468 |
| lu05 | real-relief | 3 | 5,021 | interior D1 (TN399__MGDS_30373:interior; TN399__MGDS_33068:fallback; AT42-06__MGDS_5174:not_scorable) | +0.18 [CI NOT FOUND] | +0.22 | +0.19 | +0.25 | NOT FOUND (interior) | 0.3414 |
| lu07 | real-relief | 2 | 79 | interior D1 (EW9801__MGDS_31425:interior; KM0923__MGDS_31429:interior) | +0.49 [CI NOT FOUND] | +0.54 | +0.39 | +0.56 | NOT FOUND (interior) | 0.8644 |
| lu08 | real-relief | 1 | 969 | interior D1 (FK181031__MGDS_24618:interior) | +0.63 [+0.52, +0.72] | +0.53 | +0.64 | +0.58 | NOT FOUND (interior) | 0.0493 |
| lu09 | real-relief | 1 | 631 | interior D1 (tag_m127:interior) | -0.09 [-0.25, +0.10] | +0.10 | -0.03 | +0.19 | NOT FOUND (interior) | 0.4110 |
| lu13 | real-relief | 2 | 4,844 | interior D1 (NA080__MGDS_31290:interior; NA080__MGDS_31289:interior) | +0.59 [CI NOT FOUND] | +0.51 | +0.70 | +0.59 | NOT FOUND (interior) | 0.1055 |
| lu14 | real-relief | 1 | 0 | interior D1 (AR26__MGDS_31838:fallback) | — [+0.68, +0.90] | — | — | — | NOT FOUND (interior) | — |
| lu15 | real-relief | 2 | 670 | interior D1 (TN299__MGDS_27339:interior; TN299__MGDS_31253:interior) | +0.17 [CI NOT FOUND] | +0.24 | +0.23 | +0.25 | NOT FOUND (interior) | 0.7902 |
| nu00 | real-relief | 1 | 523 | interior D1 (FK151121__MGDS_22383:interior) | +0.39 [+0.27, +0.51] | +0.37 | +0.50 | +0.33 | NOT FOUND (interior) | 1.8572 |
| lu02 | artifact-affected | 1 | 214 | interior D1 (discol_so242_1:interior) | +0.59 [+0.27, +0.83] | +0.51 | +0.56 | +0.08 | NOT FOUND (interior) | 0.0065 |
| lu06 | artifact-affected | 1 | 44 | interior D1 (ccz_so268_1:fallback) | +0.60 [+0.17, +0.45] | +0.64 | +0.62 | +0.59 | NOT FOUND (interior) | 0.0013 |
| lu10 | artifact-affected | 1 | 119 | interior D1 (NA090__MGDS_31212:interior) | +0.34 [+0.07, +0.56] | +0.17 | +0.31 | -0.09 | NOT FOUND (interior) | 0.0289 |
| lu11 | artifact-affected | 1 | 118 | interior D1 (NA076__MGDS_32317:interior) | -0.01 [-0.33, +0.33] | +0.25 | +0.37 | +0.03 | NOT FOUND (interior) | 0.0150 |
| lu12 | artifact-affected | 18 | 1,863 | interior D1 (cal_dig_morro_bay__20180426m2_Channel1000:interior; cal_dig_morro_bay__20180427m1_Channel700:interior; cal_dig_morro_bay__20180427m3_PockmarkNorth:interior; cal_dig_morro_bay__20180428m1_Cable:interior; cal_dig_morro_bay__201804_LuciaChica2m:interior; cal_dig_morro_bay__20190314m4_LuciaChica970m:fallback; cal_dig_morro_bay__20190315m1_HeadlessCanyon:interior; cal_dig_morro_bay__20190316m1_BankTop:interior; cal_dig_morro_bay__20190317m1_1000mGully:interior; cal_dig_morro_bay__20190317m2_600mGully:interior; cal_dig_morro_bay__20190318m2_Transect601060m:interior; cal_dig_morro_bay__20190510m1_BankFlankHoles:interior; cal_dig_morro_bay__20190510m2_BankFlankIncipCh:interior; cal_dig_morro_bay__20190511m1_6thHeadlessCany:interior; cal_dig_morro_bay__20190511m2_BankTopEofCanyon3:interior; cal_dig_morro_bay__basin_flank:fallback; cal_dig_morro_bay__luciachica_2007:not_scorable; cal_dig_morro_bay__pockmark_south:interior) | +0.55 [CI NOT FOUND] | +0.53 | +0.53 | +0.62 | NOT FOUND (interior) | 0.0148 |
| vu02 | artifact-affected | 1 | 57 | interior D1 (EX1402L2__MGDS_31073:interior) | +0.65 [+0.28, +0.84] | +0.69 | +0.13 | +0.66 | NOT FOUND (interior) | 0.0265 |
| lu16 | EW0207 (scored separately) | 1 | 1,621 | interior D1 (EW0207__MGDS_32556:interior) | +0.511 [+0.44, +0.58] | +0.477 | +0.506 | +0.535 | NOT FOUND (interior) | 1.3342 |

## Panel B: all tiles

| unit | class | pairs | scored tiles | view | s_lr rho [95% CI] | GBT_ship rho | V2 rho | P-V1 rho | spectral extrapolation rho | median A_fine (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| lu00 | real-relief | 1 | 2,591 | all tiles | +0.55 [CI NOT FOUND] | +0.56 | +0.55 | +0.56 | +0.006 | 0.2829 |
| lu04 | real-relief | 1 | 184 | all tiles | +0.50 [CI NOT FOUND] | +0.53 | +0.55 | +0.56 | +0.180 | 0.7141 |
| lu05 | real-relief | 3 | 6,458 | all tiles | +0.27 [CI NOT FOUND] | +0.28 | +0.26 | +0.30 | -0.016 | 0.3546 |
| lu07 | real-relief | 2 | 351 | all tiles | +0.54 [CI NOT FOUND] | +0.47 | +0.55 | +0.56 | +0.350 | 1.3464 |
| lu08 | real-relief | 1 | 1,252 | all tiles | +0.65 [CI NOT FOUND] | +0.57 | +0.64 | +0.61 | -0.367 | 0.0592 |
| lu09 | real-relief | 1 | 772 | all tiles | +0.05 [CI NOT FOUND] | +0.18 | +0.10 | +0.27 | +0.362 | 0.4603 |
| lu13 | real-relief | 2 | 6,413 | all tiles | +0.54 [CI NOT FOUND] | +0.49 | +0.67 | +0.57 | -0.134 | 0.1015 |
| lu14 | real-relief | 1 | 52 | all tiles | +0.81 [CI NOT FOUND] | +0.66 | +0.77 | +0.68 | +0.111 | 1.3893 |
| lu15 | real-relief | 2 | 1,278 | all tiles | +0.25 [CI NOT FOUND] | +0.36 | +0.35 | +0.34 | +0.073 | 1.2574 |
| nu00 | real-relief | 1 | 732 | all tiles | +0.40 [CI NOT FOUND] | +0.36 | +0.52 | +0.29 | -0.031 | 1.9144 |
| lu02 | artifact-affected | 1 | 300 | all tiles | +0.62 [CI NOT FOUND] | +0.71 | +0.73 | +0.34 | -0.503 | 0.0082 |
| lu06 | artifact-affected | 1 | 69 | all tiles | +0.32 [CI NOT FOUND] | +0.82 | +0.81 | +0.85 | +0.248 | 0.0015 |
| lu10 | artifact-affected | 1 | 174 | all tiles | +0.43 [CI NOT FOUND] | +0.27 | +0.43 | +0.14 | +0.387 | 0.0361 |
| lu11 | artifact-affected | 1 | 206 | all tiles | -0.08 [CI NOT FOUND] | +0.27 | +0.47 | +0.30 | +0.192 | 0.0174 |
| lu12 | artifact-affected | 18 | 2,497 | all tiles | +0.34 [CI NOT FOUND] | +0.60 | +0.61 | +0.58 | -0.300 | 0.0182 |
| vu02 | artifact-affected | 1 | 109 | all tiles | +0.57 [CI NOT FOUND] | +0.73 | +0.26 | +0.60 | +0.246 | 0.0296 |
| lu16 | EW0207 (scored separately) | 1 | 2,716 | all tiles | +0.552 [CI NOT FOUND] | +0.467 | +0.522 | +0.564 | NOT FOUND | 1.5869 |

## Panel C: multi-pair units, pooled across pairs versus ranked within pairs (all tiles; R05 §4.2, bar_tables/report_tables_v1_1.md)

Each cell reads 'v1 pooled rho -> v1.1 unit value'. V2, P-V1, interior-view and spectral rows: NOT FOUND among the frozen outputs.

| unit | GBT_ship: pooled across pairs -> within-pair median | s_lr: pooled across pairs -> within-pair median |
|---|---|---|
| lu05 | +0.275 -> +0.279 | +0.264 -> +0.266 |
| lu07 | +0.445 -> +0.472 | +0.111 -> +0.544 |
| lu12 | +0.666 -> +0.604 | +0.489 -> +0.339 |
| lu13 | +0.484 -> +0.486 | +0.531 -> +0.536 |
| lu15 | +0.644 -> +0.364 | +0.547 -> +0.255 |

### Per-pair rows behind Panel C

| unit | pair | k | n tiles | GBT_ship rho (pair) | s_lr rho (pair) | unit value, median over pairs (GBT_ship / s_lr) |
|---|---|---|---|---|---|---|
| lu05 | TN399__MGDS_30373 | 16 | 6264 | +0.279 | +0.266 | +0.279 / +0.266 |
| lu05 | TN399__MGDS_33068 | 16 | 178 | +0.170 | +0.150 | +0.279 / +0.266 |
| lu05 | AT42-06__MGDS_5174 | 2 | 16 | +0.350 | +0.426 | +0.279 / +0.266 |
| lu07 | EW9801__MGDS_31425 | 2 | 128 | +0.469 | +0.481 | +0.472 / +0.544 |
| lu07 | KM0923__MGDS_31429 | 16 | 223 | +0.476 | +0.607 | +0.472 / +0.544 |
| lu12 | cal_dig_morro_bay__20180426m2_Channel1000 | 2 | 190 | +0.606 | +0.403 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20180427m1_Channel700 | 2 | 142 | +0.784 | +0.582 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20180427m3_PockmarkNorth | 2 | 85 | +0.460 | +0.112 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20180428m1_Cable | 2 | 68 | +0.602 | -0.130 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__201804_LuciaChica2m | 2 | 177 | +0.657 | +0.402 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190314m4_LuciaChica970m | 2 | 68 | +0.812 | +0.623 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190315m1_HeadlessCanyon | 2 | 149 | +0.579 | +0.109 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190316m1_BankTop | 2 | 214 | +0.612 | +0.490 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190317m1_1000mGully | 2 | 141 | +0.332 | +0.060 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190317m2_600mGully | 2 | 182 | +0.718 | +0.276 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190318m2_Transect601060m | 2 | 92 | +0.544 | -0.176 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190510m1_BankFlankHoles | 2 | 181 | +0.563 | +0.277 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190510m2_BankFlankIncipCh | 2 | 164 | +0.683 | +0.811 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190511m1_6thHeadlessCany | 2 | 205 | +0.870 | +0.675 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__20190511m2_BankTopEofCanyon3 | 2 | 204 | +0.509 | +0.782 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__basin_flank | 2 | 57 | +0.484 | +0.201 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__luciachica_2007 | 2 | 43 | +0.653 | +0.812 | +0.604 / +0.339 |
| lu12 | cal_dig_morro_bay__pockmark_south | 2 | 135 | +0.441 | -0.110 | +0.604 / +0.339 |
| lu13 | NA080__MGDS_31290 | 16 | 3245 | +0.481 | +0.539 | +0.486 / +0.536 |
| lu13 | NA080__MGDS_31289 | 16 | 3168 | +0.492 | +0.533 | +0.486 / +0.536 |
| lu15 | TN299__MGDS_27339 | 16 | 897 | +0.302 | +0.258 | +0.364 / +0.255 |
| lu15 | TN299__MGDS_31253 | 8 | 381 | +0.426 | +0.252 | +0.364 / +0.255 |
| lu05 | +0.275 -> +0.279 | +0.299 -> +0.320 | +0.302 -> +0.305 | +0.264 -> +0.266 |  |  |
| lu07 | +0.445 -> +0.472 | +0.103 -> +0.480 | +0.148 -> +0.549 | +0.111 -> +0.544 |  |  |
| lu12 | +0.666 -> +0.604 | +0.675 -> +0.564 | +0.498 -> +0.418 | +0.489 -> +0.339 |  |  |
| lu13 | +0.484 -> +0.486 | +0.256 -> +0.464 | +0.418 -> +0.421 | +0.531 -> +0.536 |  |  |
| lu15 | +0.644 -> +0.364 | -0.397 -> +0.215 | +0.456 -> +0.276 | +0.547 -> +0.255 |  |  |

