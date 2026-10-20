# BAR_v1 — strongest free predictor at block and tile scale (R04 §3)

Units: units-v1 (16 units); training cap 200 rows per unit per fold (seed 20261001); grid 48 configurations; E_1m threshold 1.0 m.


### A_fine/block (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.493 | 0.249 | 0.670 | 0.283 | 0.218 | +0.47 / +nan / +0.30 / +0.21 / +nan / +nan / +0.57 / +nan / +0.49 / +nan / +nan / +0.49 / +nan / +0.70 / +nan / +nan |
| ref_C | 16 | 0.384 | 0.163 | 0.566 | 0.421 | 0.140 | +0.43 / +0.34 / +0.16 / +0.24 / +0.27 / +0.03 / +0.42 / +0.18 / +0.44 / +0.02 / +0.41 / +0.42 / +0.69 / +0.46 / +0.35 / +0.45 |
| ref_s_lr | 16 | 0.421 | 0.183 | 0.570 | 0.345 | 0.212 | +0.44 / +0.62 / +0.30 / +0.22 / +0.21 / -0.03 / +0.59 / -0.05 / +0.43 / -0.11 / +0.45 / +0.55 / +0.69 / +0.51 / +0.31 / +0.41 |
| ref_shipstd128 | 16 | 0.402 | 0.190 | 0.529 | 0.302 | 0.189 | +0.45 / +0.64 / +0.37 / +0.21 / +0.24 / +0.07 / +0.56 / +0.04 / +0.40 / -0.09 / +0.46 / +0.47 / +0.74 / +0.51 / +0.34 / +0.41 |
| ref_rsd128 | 12 | 0.322 | 0.137 | 0.466 | 0.159 | 0.185 | +0.57 / +nan / +0.24 / +0.19 / +nan / +0.30 / +0.15 / +nan / +0.21 / +0.16 / +nan / +0.65 / +0.35 / +0.55 / +0.41 / +0.37 |
| GBT_ship_A_fine | 16 | 0.423 | 0.199 | 0.603 | 0.325 | 0.158 | +0.45 / +0.41 / +0.33 / +0.25 / +0.43 / +0.20 / +0.52 / +0.09 / +0.46 / +0.24 / +0.46 / +0.50 / +0.64 / -0.15 / +0.35 / +0.46 |
| GBT_noship_A_fine | 16 | 0.406 | 0.195 | 0.609 | 0.245 | 0.133 | +0.48 / +0.49 / +0.36 / +0.25 / +0.48 / -0.01 / +0.53 / +0.03 / +0.18 / +0.22 / +0.49 / +0.48 / +0.74 / -0.44 / +0.28 / +0.46 |

GBT_ship median rho = 0.423 [95% CI 0.251, 0.458] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/block | rho | +0.017 [-0.057, +0.176] | 8 / 8 / 0 of 16 | 0.598 |
| GBT_ship_A_fine − ref_C | A_fine/block | rho | +0.040 [-0.074, +0.127] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_A_fine − ref_s_lr | A_fine/block | rho | +0.003 [-0.118, +0.146] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/block | rho | +0.021 [-0.113, +0.117] | 9 / 7 / 0 of 16 | 0.402 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/block | rho | +0.078 [-0.098, +0.247] | 7 / 5 / 0 of 12 | 0.387 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/block | rho | -0.042 [-0.315, +0.036] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/block | capture20 | -0.006 [-0.134, +0.076] | 5 / 11 / 0 of 16 | 0.962 |
| GBT_ship_A_fine − ref_C | A_fine/block | capture20 | +0.038 [-0.100, +0.196] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_s_lr | A_fine/block | capture20 | +0.033 [-0.120, +0.223] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/block | capture20 | +0.074 [-0.044, +0.194] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/block | capture20 | +0.090 [+0.010, +0.160] | 9 / 3 / 0 of 12 | 0.073 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/block | capture20 | +0.010 [-0.059, +0.041] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/block | r2_loglog | +0.004 [-0.102, +0.100] | 7 / 9 / 0 of 16 | 0.773 |
| GBT_ship_A_fine − ref_C | A_fine/block | r2_loglog | +0.036 [-0.064, +0.118] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_A_fine − ref_s_lr | A_fine/block | r2_loglog | +0.016 [-0.107, +0.073] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/block | r2_loglog | +0.009 [-0.106, +0.048] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/block | r2_loglog | +0.062 [-0.096, +0.189] | 7 / 5 / 0 of 12 | 0.387 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/block | r2_loglog | -0.010 [-0.356, +0.037] | 4 / 3 / 0 of 7 | 0.500 |

### T_fine/block (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.493 | 0.424 | 0.692 | 0.367 | 0.324 | +0.47 / +nan / +0.32 / +0.24 / +nan / +nan / +0.70 / +nan / +0.49 / +nan / +nan / +0.80 / +nan / +0.69 / +nan / +nan |
| ref_C | 16 | 0.385 | 0.196 | 0.568 | 0.492 | 0.259 | +0.43 / +0.33 / +0.18 / +0.26 / +0.21 / +0.04 / +0.56 / +0.18 / +0.44 / +0.02 / +0.42 / +0.71 / +0.74 / +0.45 / +0.35 / +0.44 |
| ref_s_lr | 16 | 0.409 | 0.168 | 0.587 | 0.397 | 0.287 | +0.44 / +0.61 / +0.30 / +0.24 / +0.17 / -0.02 / +0.73 / -0.05 / +0.43 / -0.12 / +0.44 / +0.76 / +0.73 / +0.50 / +0.30 / +0.39 |
| ref_shipstd128 | 16 | 0.399 | 0.181 | 0.515 | 0.363 | 0.282 | +0.45 / +0.64 / +0.37 / +0.23 / +0.19 / +0.08 / +0.67 / +0.04 / +0.41 / -0.09 / +0.46 / +0.71 / +0.79 / +0.51 / +0.33 / +0.39 |
| ref_rsd128 | 12 | 0.339 | 0.181 | 0.438 | 0.175 | 0.155 | +0.57 / +nan / +0.24 / +0.20 / +nan / +0.31 / +0.17 / +nan / +0.23 / +0.16 / +nan / +0.50 / +0.43 / +0.55 / +0.40 / +0.37 |
| GBT_ship_T_fine | 16 | 0.428 | 0.202 | 0.643 | 0.473 | 0.263 | +0.48 / +0.43 / +0.32 / +0.28 / +0.35 / +0.27 / +0.66 / +0.09 / +0.45 / +0.08 / +0.50 / +0.75 / +0.80 / +0.51 / +0.33 / +0.43 |
| GBT_noship_T_fine | 16 | 0.365 | 0.213 | 0.645 | 0.321 | 0.277 | +0.49 / +0.23 / +0.34 / +0.27 / +0.39 / +0.11 / +0.68 / +0.05 / +0.41 / +0.08 / +0.50 / +0.76 / +0.81 / +0.32 / +0.34 / +0.40 |

GBT_ship median rho = 0.428 [95% CI 0.325, 0.500] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/block | rho | +0.063 [-0.028, +0.138] | 7 / 9 / 0 of 16 | 0.773 |
| GBT_ship_T_fine − ref_C | T_fine/block | rho | +0.043 [-0.014, +0.117] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_s_lr | T_fine/block | rho | +0.019 [-0.062, +0.109] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/block | rho | +0.029 [-0.071, +0.095] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/block | rho | +0.098 [-0.030, +0.264] | 7 / 5 / 0 of 12 | 0.387 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/block | rho | -0.013 [-0.180, +0.012] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/block | capture20 | -0.002 [-0.045, +0.014] | 7 / 9 / 0 of 16 | 0.773 |
| GBT_ship_T_fine − ref_C | T_fine/block | capture20 | +0.075 [+0.003, +0.214] | 14 / 2 / 0 of 16 | 0.002 |
| GBT_ship_T_fine − ref_s_lr | T_fine/block | capture20 | +0.056 [-0.018, +0.231] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/block | capture20 | +0.129 [-0.023, +0.266] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/block | capture20 | +0.174 [+0.018, +0.245] | 10 / 2 / 0 of 12 | 0.019 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/block | capture20 | -0.021 [-0.061, +0.019] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/block | r2_loglog | -0.011 [-0.057, +0.118] | 4 / 12 / 0 of 16 | 0.989 |
| GBT_ship_T_fine − ref_C | T_fine/block | r2_loglog | +0.006 [-0.018, +0.141] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_s_lr | T_fine/block | r2_loglog | +0.034 [+0.004, +0.104] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/block | r2_loglog | +0.020 [-0.022, +0.086] | 14 / 2 / 0 of 16 | 0.002 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/block | r2_loglog | +0.020 [-0.042, +0.275] | 8 / 4 / 0 of 12 | 0.194 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/block | r2_loglog | -0.123 [-0.289, +0.020] | 4 / 3 / 0 of 7 | 0.500 |

### E_1m/block (10 units)

| predictor | n units | median rho | median capture20 | median auprc | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu04 / lu05 / lu07 / lu08 / lu09 / lu10 / lu13 / lu14 / lu15 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.360 | 0.746 | 0.607 | 0.264 | 0.513 | +0.40 / +0.29 / +0.17 / +nan / +0.34 / +nan / +0.36 / +0.38 / +nan / +0.68 |
| ref_C | 10 | 0.308 | 0.462 | 0.738 | 0.377 | 0.529 | +0.37 / +0.16 / +0.19 / +0.01 / +0.30 / +0.19 / +0.36 / +0.31 / +0.70 / +0.43 |
| ref_s_lr | 10 | 0.319 | 0.467 | 0.728 | 0.309 | 0.537 | +0.38 / +0.29 / +0.17 / -0.05 / +0.35 / -0.04 / +0.29 / +0.40 / +0.71 / +0.50 |
| ref_shipstd128 | 10 | 0.345 | 0.509 | 0.746 | 0.269 | 0.486 | +0.39 / +0.35 / +0.16 / +0.05 / +0.34 / +0.05 / +0.30 / +0.35 / +0.77 / +0.50 |
| ref_rsd128 | 9 | 0.289 | 0.381 | 0.779 | 0.165 | 0.481 | +0.48 / +0.23 / +0.14 / +0.29 / +0.17 / +nan / +0.24 / +0.43 / +0.40 / +0.54 |
| P_GBT_ship_E_1m | 10 | 0.327 | 0.471 | 0.719 | 0.197 | 0.443 | +0.40 / +0.31 / +0.18 / +0.29 / +0.24 / -0.02 / +0.34 / +0.35 / +0.73 / +0.37 |
| GBT_ship_E_1m | 10 | 0.320 | 0.483 | 0.754 | 0.163 | 0.382 | +0.45 / +0.30 / +0.18 / +0.06 / +0.27 / +0.09 / +0.34 / +0.35 / +0.74 / +0.66 |
| P_GBT_noship_E_1m | 10 | 0.265 | 0.479 | 0.739 | 0.062 | 0.367 | +0.42 / +0.32 / +0.21 / +0.02 / +0.33 / +0.09 / +0.20 / +0.37 / +0.64 / +0.11 |
| GBT_noship_E_1m | 10 | 0.338 | 0.509 | 0.764 | 0.130 | 0.355 | +0.45 / +0.31 / +0.21 / +0.24 / +0.33 / +0.17 / +0.35 / +0.35 / +0.76 / +0.66 |

GBT_ship median rho = 0.320 [95% CI 0.178, 0.507] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_E_1m − GBT_noship_E_1m | E_1m/block | rho | -0.018 [-0.084, +0.009] | 3 / 7 / 0 of 10 | 0.945 |
| GBT_ship_E_1m − ref_C | E_1m/block | rho | +0.012 [-0.034, +0.135] | 6 / 4 / 0 of 10 | 0.377 |
| GBT_ship_E_1m − ref_s_lr | E_1m/block | rho | +0.001 [-0.076, +0.109] | 8 / 2 / 0 of 10 | 0.055 |
| GBT_ship_E_1m − ref_shipstd128 | E_1m/block | rho | -0.026 [-0.072, +0.081] | 7 / 3 / 0 of 10 | 0.172 |
| GBT_ship_E_1m − ref_rsd128 | E_1m/block | rho | +0.046 [-0.074, +0.231] | 6 / 3 / 0 of 9 | 0.254 |
| GBT_ship_E_1m − sigma_m_R02 | E_1m/block | rho | -0.024 [-0.070, +0.054] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_E_1m − GBT_noship_E_1m | E_1m/block | capture20 | -0.026 [-0.071, +0.017] | 3 / 7 / 0 of 10 | 0.945 |
| GBT_ship_E_1m − ref_C | E_1m/block | capture20 | +0.021 [+0.003, +0.154] | 9 / 1 / 0 of 10 | 0.011 |
| GBT_ship_E_1m − ref_s_lr | E_1m/block | capture20 | +0.016 [-0.112, +0.053] | 8 / 2 / 0 of 10 | 0.055 |
| GBT_ship_E_1m − ref_shipstd128 | E_1m/block | capture20 | -0.027 [-0.089, +0.021] | 4 / 6 / 0 of 10 | 0.828 |
| GBT_ship_E_1m − ref_rsd128 | E_1m/block | capture20 | +0.116 [-0.172, +0.131] | 7 / 2 / 0 of 9 | 0.090 |
| GBT_ship_E_1m − sigma_m_R02 | E_1m/block | capture20 | -0.092 [-0.100, +0.034] | 3 / 4 / 0 of 7 | 0.773 |
| P_GBT_ship_E_1m − P_GBT_noship_E_1m | E_1m/block | auprc | -0.020 [-0.048, +0.099] | 4 / 6 / 0 of 10 | 0.828 |

Units excluded from detection metrics (prevalence of E_1m > 0 at block scale outside [5 %, 95 %]): {'lu02': 0.00977891156462585, 'lu06': 0.010526315789473684, 'lu11': 0.0028373266078184113, 'lu12': 0.021135992722797443, 'nu00': 0.9851593190746399, 'vu02': 0.030285035629453682}; prevalence per unit: lu00 0.60, lu02 0.01, lu04 0.79, lu05 0.50, lu06 0.01, lu07 0.91, lu08 0.08, lu09 0.69, lu10 0.08, lu11 0.00, lu12 0.02, lu13 0.11, lu14 0.75, lu15 0.67, nu00 0.99, vu02 0.03

Training rows per unit after the 200-row cap (block): lu00 200/40396, lu02 200/4704, lu04 200/2739, lu05 200/99902, lu06 200/1045, lu07 200/4747, lu08 200/19615, lu09 200/12100, lu10 200/2654, lu11 200/3172, lu12 200/37377, lu13 200/100200, lu14 200/631, lu15 200/19472, nu00 200/11455, vu02 200/1684

### A_fine/tile (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.519 | 0.308 | 0.621 | 0.339 | 0.277 | +0.57 / +nan / +0.48 / +0.28 / +nan / +nan / +0.63 / +nan / +0.52 / +nan / +nan / +0.49 / +nan / +0.76 / +nan / +nan |
| ref_C | 16 | 0.406 | 0.198 | 0.564 | 0.455 | 0.166 | +0.51 / +0.36 / +0.34 / +0.30 / +0.31 / +0.15 / +0.47 / +0.26 / +0.46 / +0.05 / +0.50 / +0.42 / +0.71 / +0.46 / +0.39 / +0.52 |
| ref_s_lr | 16 | 0.496 | 0.293 | 0.553 | 0.383 | 0.244 | +0.55 / +0.62 / +0.50 / +0.26 / +0.32 / +0.11 / +0.65 / +0.05 / +0.43 / -0.09 / +0.49 / +0.53 / +0.78 / +0.55 / +0.40 / +0.56 |
| ref_shipstd128 | 16 | 0.477 | 0.274 | 0.553 | 0.336 | 0.209 | +0.55 / +0.61 / +0.56 / +0.25 / +0.36 / +0.16 / +0.59 / +0.13 / +0.35 / -0.06 / +0.50 / +0.46 / +0.70 / +0.53 / +0.39 / +0.56 |
| ref_rsd128 | 12 | 0.362 | 0.183 | 0.483 | 0.178 | 0.204 | +0.63 / +nan / +0.37 / +0.22 / +nan / +0.35 / +0.17 / +nan / +0.14 / +0.30 / +nan / +0.65 / +0.17 / +0.56 / +0.50 / +0.48 |
| GBT_ship_A_fine | 16 | 0.550 | 0.348 | 0.674 | 0.417 | 0.169 | +0.57 / +0.70 / +0.54 / +0.28 / +0.84 / +0.44 / +0.59 / +0.15 / +0.13 / +0.40 / +0.67 / +0.48 / +0.56 / +0.64 / +0.38 / +0.68 |
| GBT_noship_A_fine | 16 | 0.482 | 0.279 | 0.583 | 0.221 | 0.131 | +0.58 / +0.65 / +0.51 / +0.30 / +0.79 / +0.10 / +0.61 / +0.06 / +0.46 / +0.10 / +0.68 / +0.26 / +0.61 / -0.40 / +0.23 / +0.59 |

GBT_ship median rho = 0.550 [95% CI 0.397, 0.644] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/tile | rho | +0.068 [-0.037, +0.293] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_C | A_fine/tile | rho | +0.144 [+0.009, +0.251] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_s_lr | A_fine/tile | rho | +0.054 [-0.032, +0.194] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/tile | rho | +0.073 [-0.001, +0.193] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/tile | rho | +0.151 [-0.050, +0.339] | 8 / 4 / 0 of 12 | 0.194 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/tile | rho | +0.022 [-0.244, +0.063] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/tile | capture20 | +0.091 [-0.036, +0.254] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_C | A_fine/tile | capture20 | +0.110 [-0.025, +0.242] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_s_lr | A_fine/tile | capture20 | +0.121 [-0.041, +0.308] | 8 / 8 / 0 of 16 | 0.598 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/tile | capture20 | +0.121 [-0.005, +0.219] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/tile | capture20 | +0.162 [-0.013, +0.219] | 8 / 4 / 0 of 12 | 0.194 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/tile | capture20 | +0.065 [-0.045, +0.212] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/tile | r2_loglog | +0.068 [-0.089, +0.157] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_A_fine − ref_C | A_fine/tile | r2_loglog | +0.150 [+0.004, +0.266] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_s_lr | A_fine/tile | r2_loglog | +0.055 [-0.046, +0.248] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/tile | r2_loglog | +0.074 [-0.024, +0.244] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/tile | r2_loglog | +0.121 [-0.065, +0.277] | 8 / 4 / 0 of 12 | 0.194 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/tile | r2_loglog | -0.025 [-0.328, +0.061] | 4 / 3 / 0 of 7 | 0.500 |

### T_fine/tile (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.575 | 0.502 | 0.687 | 0.441 | 0.418 | +0.57 / +nan / +0.52 / +0.34 / +nan / +nan / +0.79 / +nan / +0.53 / +nan / +nan / +0.82 / +nan / +0.75 / +nan / +nan |
| ref_C | 16 | 0.420 | 0.243 | 0.603 | 0.527 | 0.294 | +0.53 / +0.33 / +0.37 / +0.34 / +0.28 / +0.19 / +0.62 / +0.25 / +0.47 / +0.03 / +0.46 / +0.71 / +0.88 / +0.45 / +0.39 / +0.49 |
| ref_s_lr | 16 | 0.483 | 0.261 | 0.583 | 0.439 | 0.325 | +0.54 / +0.61 / +0.51 / +0.29 / +0.31 / +0.16 / +0.80 / +0.03 / +0.46 / -0.16 / +0.45 / +0.75 / +0.87 / +0.54 / +0.38 / +0.50 |
| ref_shipstd128 | 16 | 0.493 | 0.276 | 0.592 | 0.401 | 0.305 | +0.54 / +0.59 / +0.56 / +0.28 / +0.35 / +0.22 / +0.73 / +0.12 / +0.37 / -0.11 / +0.46 / +0.69 / +0.81 / +0.53 / +0.39 / +0.52 |
| ref_rsd128 | 12 | 0.405 | 0.228 | 0.472 | 0.204 | 0.173 | +0.64 / +nan / +0.38 / +0.23 / +nan / +0.39 / +0.17 / +nan / +0.19 / +0.30 / +nan / +0.48 / +0.42 / +0.56 / +0.49 / +0.50 |
| GBT_ship_T_fine | 16 | 0.626 | 0.446 | 0.650 | 0.430 | 0.271 | +0.59 / +0.73 / +0.46 / +0.36 / +0.83 / +0.49 / +0.71 / +0.16 / +0.48 / +0.53 / +0.73 / +0.74 / +0.81 / +0.66 / +0.36 / +0.67 |
| GBT_noship_T_fine | 16 | 0.548 | 0.427 | 0.682 | 0.331 | 0.284 | +0.61 / +0.58 / +0.52 / +0.40 / +0.83 / +0.43 / +0.70 / +0.12 / +0.52 / +0.09 / +0.74 / +0.79 / +0.88 / +0.17 / +0.39 / +0.60 |

GBT_ship median rho = 0.626 [95% CI 0.480, 0.730] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/tile | rho | +0.079 [-0.030, +0.183] | 7 / 9 / 0 of 16 | 0.773 |
| GBT_ship_T_fine − ref_C | T_fine/tile | rho | +0.206 [+0.060, +0.349] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_s_lr | T_fine/tile | rho | +0.144 [+0.014, +0.284] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/tile | rho | +0.133 [+0.007, +0.282] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/tile | rho | +0.155 [+0.029, +0.315] | 10 / 2 / 0 of 12 | 0.019 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/tile | rho | +0.014 [-0.084, +0.014] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/tile | capture20 | -0.032 [-0.070, +0.048] | 7 / 9 / 0 of 16 | 0.773 |
| GBT_ship_T_fine − ref_C | T_fine/tile | capture20 | +0.048 [-0.072, +0.136] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_s_lr | T_fine/tile | capture20 | +0.067 [-0.132, +0.322] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/tile | capture20 | +0.058 [-0.120, +0.291] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/tile | capture20 | +0.093 [-0.065, +0.284] | 9 / 3 / 0 of 12 | 0.073 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/tile | capture20 | -0.079 [-0.097, +0.043] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/tile | r2_loglog | +0.019 [-0.054, +0.202] | 8 / 8 / 0 of 16 | 0.598 |
| GBT_ship_T_fine − ref_C | T_fine/tile | r2_loglog | +0.203 [+0.067, +0.347] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_s_lr | T_fine/tile | r2_loglog | +0.185 [+0.018, +0.334] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/tile | r2_loglog | +0.170 [+0.011, +0.329] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/tile | r2_loglog | +0.194 [+0.026, +0.308] | 10 / 2 / 0 of 12 | 0.019 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/tile | r2_loglog | -0.084 [-0.220, +0.053] | 2 / 5 / 0 of 7 | 0.938 |

Training rows per unit after the 200-row cap (tile): lu00 200/2599, lu02 200/302, lu04 186/186, lu05 200/6508, lu06 69/69, lu07 200/361, lu08 200/1254, lu09 200/778, lu10 177/177, lu11 200/207, lu12 200/2549, lu13 200/6476, lu14 59/59, lu15 200/1284, nu00 200/738, vu02 111/111

### EW0207 (lu16), block scale, A_fine — separate one-unit table (41606 rows; models refit on the 16 development units)

| predictor | rho | log-log R² | capture20 |
|---|---|---|---|
| ref_C | 0.406 | 0.174 | 0.400 |
| ref_s_lr | 0.447 | 0.217 | 0.434 |
| ref_shipstd128 | 0.421 | 0.184 | 0.415 |
| ref_rsd128 | nan | nan | nan |
| GBT_ship_A_fine | 0.458 | 0.176 | 0.444 |
| GBT_noship_A_fine | 0.430 | 0.171 | 0.321 |

### EW0207 (lu16), tile scale, A_fine — separate one-unit table (2738 rows; models refit on the 16 development units)

| predictor | rho | log-log R² | capture20 |
|---|---|---|---|
| ref_C | 0.480 | 0.243 | 0.413 |
| ref_s_lr | 0.554 | 0.332 | 0.442 |
| ref_shipstd128 | 0.509 | 0.277 | 0.412 |
| ref_rsd128 | nan | nan | nan |
| GBT_ship_A_fine | 0.546 | 0.290 | 0.411 |
| GBT_noship_A_fine | 0.533 | 0.283 | 0.382 |
