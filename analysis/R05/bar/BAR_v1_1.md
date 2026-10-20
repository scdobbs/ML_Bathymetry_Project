# BAR_v1_1 — strongest free predictor under eval_v1.1 (R05 §2.4)

Units: units-v1 (16 units); tiles_v1.1 / blocks_v1.1; training cap 200 rows per unit per fold (seed 20261001); grid 48 configurations; inner criterion = eval_v1.1 statistic; E_1m threshold 1.0 m; pair eligibility >= 10 tiles; detection rule per pair [0.05, 0.95].


### A_fine/tile (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.489 | 0.250 | 0.622 | 0.336 | 0.277 | +0.57 / +nan / +0.48 / +0.28 / +nan / +nan / +0.63 / +nan / +0.52 / +nan / +nan / +0.49 / +nan / +0.40 / +nan / +nan |
| ref_C | 16 | 0.405 | 0.193 | 0.453 | 0.453 | 0.166 | +0.51 / +0.35 / +0.33 / +0.30 / +0.31 / +0.55 / +0.46 / +0.26 / +0.45 / +0.05 / +0.42 / +0.42 / +0.71 / +0.28 / +0.39 / +0.52 |
| ref_s_lr | 16 | 0.465 | 0.266 | 0.521 | 0.381 | 0.246 | +0.55 / +0.62 / +0.50 / +0.27 / +0.32 / +0.54 / +0.65 / +0.05 / +0.43 / -0.08 / +0.34 / +0.54 / +0.81 / +0.25 / +0.40 / +0.57 |
| ref_shipstd128 | 16 | 0.425 | 0.260 | 0.532 | 0.334 | 0.212 | +0.55 / +0.60 / +0.56 / +0.25 / +0.36 / +0.50 / +0.59 / +0.13 / +0.36 / -0.05 / +0.34 / +0.46 / +0.77 / +0.25 / +0.39 / +0.56 |
| ref_rsd128 | 12 | 0.317 | 0.143 | 0.485 | 0.178 | 0.207 | +0.63 / +nan / +0.36 / +0.32 / +nan / +0.27 / +0.17 / +nan / +0.14 / +0.31 / +nan / +0.66 / +0.28 / +0.18 / +0.50 / +0.48 |
| GBT_ship_A_fine | 16 | 0.510 | 0.268 | 0.670 | 0.448 | 0.201 | +0.56 / +0.71 / +0.53 / +0.28 / +0.82 / +0.47 / +0.57 / +0.18 / +0.27 / +0.27 / +0.60 / +0.49 / +0.66 / +0.36 / +0.36 / +0.73 |
| GBT_noship_A_fine | 16 | 0.480 | 0.281 | 0.612 | 0.249 | 0.179 | +0.59 / +0.66 / +0.48 / +0.32 / +0.81 / +0.48 / +0.57 / +0.24 / +0.48 / +0.13 / +0.56 / +0.46 / +0.73 / +0.22 / +0.35 / +0.55 |

GBT_ship median rho = 0.510 [95% CI 0.362, 0.604] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/tile | rho | +0.030 [-0.112, +0.053] | 9 / 7 / 0 of 16 | 0.402 |
| GBT_ship_A_fine − ref_C | A_fine/tile | rho | +0.105 [-0.037, +0.238] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_s_lr | A_fine/tile | rho | +0.045 [-0.068, +0.218] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/tile | rho | +0.085 [-0.026, +0.209] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/tile | rho | +0.162 [-0.048, +0.283] | 7 / 5 / 0 of 12 | 0.387 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/tile | rho | -0.003 [-0.241, +0.058] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/tile | capture20 | +0.058 [-0.059, +0.208] | 9 / 7 / 0 of 16 | 0.402 |
| GBT_ship_A_fine − ref_C | A_fine/tile | capture20 | +0.217 [-0.031, +0.271] | 8 / 8 / 0 of 16 | 0.598 |
| GBT_ship_A_fine − ref_s_lr | A_fine/tile | capture20 | +0.149 [-0.062, +0.289] | 7 / 9 / 0 of 16 | 0.773 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/tile | capture20 | +0.138 [-0.044, +0.201] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/tile | capture20 | +0.184 [-0.016, +0.251] | 8 / 4 / 0 of 12 | 0.194 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/tile | capture20 | +0.095 [-0.072, +0.323] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/tile | r2_loglog | -0.013 [-0.097, +0.091] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_C | A_fine/tile | r2_loglog | +0.074 [-0.042, +0.244] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_s_lr | A_fine/tile | r2_loglog | +0.002 [-0.071, +0.216] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/tile | r2_loglog | +0.007 [-0.041, +0.217] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/tile | r2_loglog | +0.106 [-0.035, +0.269] | 8 / 4 / 0 of 12 | 0.194 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/tile | r2_loglog | -0.001 [-0.210, +0.088] | 3 / 4 / 0 of 7 | 0.773 |
Ineligible pairs (< 10 tiles with finite target and prediction): []; units dropped: []

### T_fine/tile (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.537 | 0.313 | 0.687 | 0.437 | 0.417 | +0.57 / +nan / +0.52 / +0.33 / +nan / +nan / +0.79 / +nan / +0.54 / +nan / +nan / +0.82 / +nan / +0.36 / +nan / +nan |
| ref_C | 16 | 0.410 | 0.242 | 0.514 | 0.527 | 0.298 | +0.53 / +0.32 / +0.37 / +0.31 / +0.28 / +0.58 / +0.62 / +0.25 / +0.47 / +0.03 / +0.43 / +0.71 / +0.87 / +0.25 / +0.39 / +0.49 |
| ref_s_lr | 16 | 0.485 | 0.251 | 0.549 | 0.437 | 0.328 | +0.54 / +0.61 / +0.50 / +0.24 / +0.31 / +0.58 / +0.80 / +0.03 / +0.46 / -0.15 / +0.30 / +0.75 / +0.87 / +0.22 / +0.38 / +0.51 |
| ref_shipstd128 | 16 | 0.461 | 0.273 | 0.582 | 0.400 | 0.310 | +0.54 / +0.58 / +0.56 / +0.28 / +0.35 / +0.55 / +0.73 / +0.12 / +0.38 / -0.11 / +0.32 / +0.70 / +0.83 / +0.22 / +0.39 / +0.53 |
| ref_rsd128 | 12 | 0.338 | 0.213 | 0.460 | 0.204 | 0.175 | +0.64 / +nan / +0.37 / +0.30 / +nan / +0.29 / +0.17 / +nan / +0.19 / +0.31 / +nan / +0.49 / +0.56 / +0.18 / +0.49 / +0.50 |
| GBT_ship_T_fine | 16 | 0.586 | 0.425 | 0.636 | 0.466 | 0.284 | +0.61 / +0.57 / +0.53 / +0.35 / +0.81 / +0.61 / +0.72 / +0.22 / +0.55 / +0.24 / +0.67 / +0.77 / +0.83 / +0.23 / +0.36 / +0.63 |
| GBT_noship_T_fine | 16 | 0.611 | 0.440 | 0.661 | 0.342 | 0.285 | +0.60 / +0.62 / +0.53 / +0.37 / +0.85 / +0.63 / +0.74 / +0.15 / +0.49 / +0.04 / +0.71 / +0.82 / +0.89 / +0.25 / +0.39 / +0.64 |

GBT_ship median rho = 0.586 [95% CI 0.362, 0.672] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/tile | rho | -0.025 [-0.058, +0.044] | 5 / 11 / 0 of 16 | 0.962 |
| GBT_ship_T_fine − ref_C | T_fine/tile | rho | +0.176 [+0.045, +0.278] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_s_lr | T_fine/tile | rho | +0.101 [+0.009, +0.299] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/tile | rho | +0.125 [-0.004, +0.276] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/tile | rho | +0.239 [-0.006, +0.363] | 9 / 3 / 0 of 12 | 0.073 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/tile | rho | +0.012 [-0.071, +0.038] | 4 / 3 / 0 of 7 | 0.500 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/tile | capture20 | -0.025 [-0.130, +0.018] | 7 / 8 / 1 of 16 | 0.696 |
| GBT_ship_T_fine − ref_C | T_fine/tile | capture20 | +0.121 [-0.053, +0.230] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_s_lr | T_fine/tile | capture20 | +0.086 [-0.143, +0.271] | 9 / 6 / 1 of 16 | 0.304 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/tile | capture20 | +0.053 [-0.144, +0.176] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/tile | capture20 | +0.130 [-0.095, +0.298] | 9 / 3 / 0 of 12 | 0.073 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/tile | capture20 | -0.036 [-0.077, +0.025] | 1 / 6 / 0 of 7 | 0.992 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/tile | r2_loglog | -0.015 [-0.047, +0.092] | 8 / 8 / 0 of 16 | 0.598 |
| GBT_ship_T_fine − ref_C | T_fine/tile | r2_loglog | +0.183 [+0.059, +0.361] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_s_lr | T_fine/tile | r2_loglog | +0.174 [+0.060, +0.336] | 14 / 2 / 0 of 16 | 0.002 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/tile | r2_loglog | +0.152 [+0.056, +0.330] | 14 / 2 / 0 of 16 | 0.002 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/tile | r2_loglog | +0.180 [+0.051, +0.344] | 10 / 2 / 0 of 12 | 0.019 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/tile | r2_loglog | +0.079 [-0.062, +0.079] | 3 / 4 / 0 of 7 | 0.773 |
Ineligible pairs (< 10 tiles with finite target and prediction): []; units dropped: []

Training rows per unit after the 200-row cap (tile): lu00 200/2591, lu02 200/300, lu04 184/184, lu05 200/6458, lu06 69/69, lu07 200/351, lu08 200/1252, lu09 200/772, lu10 174/174, lu11 200/206, lu12 200/2497, lu13 200/6413, lu14 52/52, lu15 200/1278, nu00 200/732, vu02 109/109

### A_fine/block (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.467 | 0.209 | 0.670 | 0.283 | 0.218 | +0.47 / +nan / +0.30 / +0.21 / +nan / +nan / +0.57 / +nan / +0.49 / +nan / +nan / +0.49 / +nan / +0.31 / +nan / +nan |
| ref_C | 16 | 0.345 | 0.157 | 0.517 | 0.420 | 0.140 | +0.43 / +0.34 / +0.16 / +0.24 / +0.27 / +0.42 / +0.42 / +0.18 / +0.44 / +0.02 / +0.28 / +0.42 / +0.69 / +0.25 / +0.35 / +0.45 |
| ref_s_lr | 16 | 0.329 | 0.150 | 0.477 | 0.344 | 0.212 | +0.44 / +0.62 / +0.30 / +0.23 / +0.21 / +0.34 / +0.59 / -0.05 / +0.43 / -0.11 / +0.23 / +0.55 / +0.70 / +0.20 / +0.31 / +0.41 |
| ref_shipstd128 | 16 | 0.376 | 0.173 | 0.457 | 0.302 | 0.190 | +0.45 / +0.64 / +0.36 / +0.22 / +0.24 / +0.39 / +0.56 / +0.04 / +0.40 / -0.09 / +0.25 / +0.47 / +0.76 / +0.22 / +0.34 / +0.41 |
| ref_rsd128 | 12 | 0.261 | 0.107 | 0.465 | 0.159 | 0.185 | +0.57 / +nan / +0.24 / +0.28 / +nan / +0.22 / +0.15 / +nan / +0.21 / +0.16 / +nan / +0.65 / +0.37 / +0.17 / +0.41 / +0.37 |
| GBT_ship_A_fine | 16 | 0.362 | 0.183 | 0.622 | 0.352 | 0.151 | +0.47 / +0.34 / +0.34 / +0.25 / +0.41 / +0.31 / +0.51 / +0.10 / +0.47 / +0.32 / +0.32 / +0.47 / +0.60 / +0.17 / +0.38 / +0.43 |
| GBT_noship_A_fine | 16 | 0.379 | 0.180 | 0.573 | 0.253 | 0.150 | +0.48 / +0.48 / +0.32 / +0.24 / +0.40 / +0.36 / +0.53 / +0.05 / +0.45 / +0.00 / +0.32 / +0.46 / +0.69 / +0.12 / +0.36 / +0.44 |

GBT_ship median rho = 0.362 [95% CI 0.315, 0.469] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/block | rho | -0.017 [-0.083, +0.024] | 9 / 7 / 0 of 16 | 0.402 |
| GBT_ship_A_fine − ref_C | A_fine/block | rho | +0.017 [-0.056, +0.107] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_s_lr | A_fine/block | rho | +0.033 [-0.084, +0.130] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/block | rho | -0.014 [-0.067, +0.107] | 9 / 7 / 0 of 16 | 0.402 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/block | rho | +0.147 [+0.015, +0.259] | 7 / 5 / 0 of 12 | 0.387 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/block | rho | +0.002 [-0.057, +0.043] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/block | capture20 | +0.049 [-0.005, +0.082] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_C | A_fine/block | capture20 | +0.104 [-0.035, +0.210] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_s_lr | A_fine/block | capture20 | +0.144 [-0.045, +0.235] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/block | capture20 | +0.164 [-0.049, +0.210] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/block | capture20 | +0.162 [+0.018, +0.247] | 10 / 2 / 0 of 12 | 0.019 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/block | capture20 | -0.002 [-0.024, +0.075] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_A_fine − GBT_noship_A_fine | A_fine/block | r2_loglog | +0.003 [-0.026, +0.055] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_C | A_fine/block | r2_loglog | +0.026 [-0.027, +0.097] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_A_fine − ref_s_lr | A_fine/block | r2_loglog | +0.033 [-0.032, +0.081] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_shipstd128 | A_fine/block | r2_loglog | +0.010 [-0.033, +0.075] | 10 / 6 / 0 of 16 | 0.227 |
| GBT_ship_A_fine − ref_rsd128 | A_fine/block | r2_loglog | +0.107 [-0.014, +0.174] | 8 / 4 / 0 of 12 | 0.194 |
| GBT_ship_A_fine − sigma_m_R02 | A_fine/block | r2_loglog | +0.012 [-0.049, +0.042] | 4 / 3 / 0 of 7 | 0.500 |
Ineligible pairs (< 10 tiles with finite target and prediction): []; units dropped: []

### T_fine/block (16 units)

| predictor | n units | median rho | median r2_loglog | median capture20 | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.468 | 0.211 | 0.691 | 0.366 | 0.324 | +0.47 / +nan / +0.32 / +0.24 / +nan / +nan / +0.70 / +nan / +0.49 / +nan / +nan / +0.80 / +nan / +0.27 / +nan / +nan |
| ref_C | 16 | 0.340 | 0.172 | 0.539 | 0.492 | 0.260 | +0.43 / +0.33 / +0.18 / +0.26 / +0.21 / +0.43 / +0.56 / +0.18 / +0.44 / +0.02 / +0.29 / +0.71 / +0.74 / +0.22 / +0.35 / +0.44 |
| ref_s_lr | 16 | 0.328 | 0.153 | 0.538 | 0.396 | 0.287 | +0.44 / +0.61 / +0.30 / +0.24 / +0.17 / +0.35 / +0.73 / -0.06 / +0.43 / -0.12 / +0.22 / +0.76 / +0.73 / +0.16 / +0.30 / +0.39 |
| ref_shipstd128 | 16 | 0.383 | 0.174 | 0.488 | 0.363 | 0.283 | +0.45 / +0.64 / +0.37 / +0.18 / +0.19 / +0.40 / +0.67 / +0.04 / +0.41 / -0.09 / +0.26 / +0.71 / +0.81 / +0.19 / +0.33 / +0.39 |
| ref_rsd128 | 12 | 0.256 | 0.119 | 0.438 | 0.175 | 0.155 | +0.57 / +nan / +0.24 / +0.27 / +nan / +0.23 / +0.17 / +nan / +0.23 / +0.16 / +nan / +0.50 / +0.45 / +0.16 / +0.40 / +0.37 |
| GBT_ship_T_fine | 16 | 0.359 | 0.191 | 0.640 | 0.445 | 0.285 | +0.48 / +0.36 / +0.34 / +0.27 / +0.36 / +0.40 / +0.66 / +0.11 / +0.45 / +0.26 / +0.34 / +0.75 / +0.77 / +0.23 / +0.34 / +0.48 |
| GBT_noship_T_fine | 16 | 0.349 | 0.205 | 0.596 | 0.342 | 0.271 | +0.49 / +0.31 / +0.34 / +0.27 / +0.34 / +0.40 / +0.63 / +0.09 / +0.47 / +0.00 / +0.32 / +0.75 / +0.81 / +0.23 / +0.36 / +0.42 |

GBT_ship median rho = 0.359 [95% CI 0.336, 0.481] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/block | rho | +0.011 [-0.013, +0.045] | 9 / 7 / 0 of 16 | 0.402 |
| GBT_ship_T_fine − ref_C | T_fine/block | rho | +0.019 [-0.025, +0.118] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_s_lr | T_fine/block | rho | +0.031 [-0.076, +0.130] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/block | rho | -0.024 [-0.069, +0.130] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/block | rho | +0.172 [-0.001, +0.280] | 9 / 3 / 0 of 12 | 0.073 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/block | rho | -0.013 [-0.038, +0.018] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/block | capture20 | +0.044 [-0.090, +0.100] | 9 / 7 / 0 of 16 | 0.402 |
| GBT_ship_T_fine − ref_C | T_fine/block | capture20 | +0.101 [-0.031, +0.196] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_s_lr | T_fine/block | capture20 | +0.103 [-0.055, +0.220] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/block | capture20 | +0.152 [-0.047, +0.250] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/block | capture20 | +0.203 [+0.035, +0.255] | 11 / 1 / 0 of 12 | 0.003 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/block | capture20 | -0.030 [-0.075, +0.066] | 2 / 5 / 0 of 7 | 0.938 |
| GBT_ship_T_fine − GBT_noship_T_fine | T_fine/block | r2_loglog | -0.014 [-0.024, +0.029] | 6 / 10 / 0 of 16 | 0.895 |
| GBT_ship_T_fine − ref_C | T_fine/block | r2_loglog | +0.018 [-0.023, +0.107] | 12 / 4 / 0 of 16 | 0.038 |
| GBT_ship_T_fine − ref_s_lr | T_fine/block | r2_loglog | +0.038 [+0.027, +0.114] | 13 / 3 / 0 of 16 | 0.011 |
| GBT_ship_T_fine − ref_shipstd128 | T_fine/block | r2_loglog | +0.016 [-0.010, +0.085] | 11 / 5 / 0 of 16 | 0.105 |
| GBT_ship_T_fine − ref_rsd128 | T_fine/block | r2_loglog | +0.107 [+0.001, +0.279] | 10 / 2 / 0 of 12 | 0.019 |
| GBT_ship_T_fine − sigma_m_R02 | T_fine/block | r2_loglog | +0.063 [-0.039, +0.063] | 3 / 4 / 0 of 7 | 0.773 |
Ineligible pairs (< 10 tiles with finite target and prediction): []; units dropped: []

### E_1m/block (16 units)

| predictor | n units | median rho | median capture20 | median auprc | pooled rho | pooled within-R² / AUPRC | rho per unit lu00 / lu02 / lu04 / lu05 / lu06 / lu07 / lu08 / lu09 / lu10 / lu11 / lu12 / lu13 / lu14 / lu15 / nu00 / vu02 |
|---|---|---|---|---|---|---|---|
| sigma_m_R02 | 7 | 0.340 | 0.746 | 0.611 | 0.263 | 0.512 | +0.40 / +nan / +0.29 / +0.17 / +nan / +nan / +0.34 / +nan / +0.36 / +nan / +nan / +0.38 / +nan / +0.25 / +nan / +nan |
| ref_C | 11 | 0.313 | 0.504 | 0.727 | 0.377 | 0.528 | +0.37 / +nan / +0.16 / +0.19 / +nan / +0.42 / +0.30 / +0.19 / +0.36 / +nan / +0.38 / +0.31 / +0.70 / +0.14 / +nan / +nan |
| ref_s_lr | 11 | 0.285 | 0.447 | 0.697 | 0.308 | 0.536 | +0.38 / +nan / +0.29 / +0.17 / +nan / +0.24 / +0.35 / -0.04 / +0.29 / +nan / +0.23 / +0.40 / +0.71 / +0.17 / +nan / +nan |
| ref_shipstd128 | 11 | 0.308 | 0.509 | 0.728 | 0.268 | 0.485 | +0.39 / +nan / +0.35 / +0.16 / +nan / +0.31 / +0.34 / +0.05 / +0.30 / +nan / +0.24 / +0.35 / +0.78 / +0.16 / +nan / +nan |
| ref_rsd128 | 9 | 0.232 | 0.446 | 0.779 | 0.165 | 0.480 | +0.48 / +nan / +0.23 / +0.21 / +nan / +0.08 / +0.17 / +nan / +0.25 / +nan / +nan / +0.43 / +0.41 / +0.13 / +nan / +nan |
| P_GBT_ship_E_1m | 11 | 0.339 | 0.485 | 0.752 | 0.275 | 0.526 | +0.43 / +nan / +0.31 / +0.20 / +nan / +0.37 / +0.31 / +0.17 / +0.34 / +nan / +0.40 / +0.35 / +0.56 / +0.17 / +nan / +nan |
| GBT_ship_E_1m | 11 | 0.337 | 0.477 | 0.760 | 0.230 | 0.433 | +0.46 / +nan / +0.30 / +0.20 / +nan / +0.41 / +0.22 / +0.19 / +0.34 / +nan / +0.41 / +0.38 / +0.72 / +0.21 / +nan / +nan |
| P_GBT_noship_E_1m | 11 | 0.326 | 0.488 | 0.711 | 0.042 | 0.369 | +0.44 / +nan / +0.31 / +0.28 / +nan / +0.25 / +0.36 / +0.03 / +0.36 / +nan / +0.33 / +0.39 / +0.64 / +0.17 / +nan / +nan |
| GBT_noship_E_1m | 11 | 0.337 | 0.501 | 0.729 | 0.113 | 0.350 | +0.46 / +nan / +0.32 / +0.28 / +nan / +0.37 / +0.28 / +0.07 / +0.36 / +nan / +0.36 / +0.34 / +0.76 / +0.19 / +nan / +nan |

GBT_ship median rho = 0.337 [95% CI 0.206, 0.412] (bootstrap over units)

| comparison | target/scale | metric | Δ median [95% CI] | units a better / b better / ties of N | sign-test p (one-sided) |
|---|---|---|---|---|---|
| GBT_ship_E_1m − GBT_noship_E_1m | E_1m/block | rho | +0.000 [-0.071, +0.048] | 6 / 5 / 0 of 11 | 0.500 |
| GBT_ship_E_1m − ref_C | E_1m/block | rho | +0.024 [-0.089, +0.110] | 8 / 3 / 0 of 11 | 0.113 |
| GBT_ship_E_1m − ref_s_lr | E_1m/block | rho | +0.052 [-0.074, +0.171] | 9 / 2 / 0 of 11 | 0.033 |
| GBT_ship_E_1m − ref_shipstd128 | E_1m/block | rho | +0.030 [-0.126, +0.139] | 8 / 3 / 0 of 11 | 0.113 |
| GBT_ship_E_1m − ref_rsd128 | E_1m/block | rho | +0.105 [-0.037, +0.238] | 6 / 3 / 0 of 9 | 0.254 |
| GBT_ship_E_1m − sigma_m_R02 | E_1m/block | rho | -0.039 [-0.124, +0.013] | 3 / 4 / 0 of 7 | 0.773 |
| GBT_ship_E_1m − GBT_noship_E_1m | E_1m/block | capture20 | -0.024 [-0.098, +0.128] | 6 / 5 / 0 of 11 | 0.500 |
| GBT_ship_E_1m − ref_C | E_1m/block | capture20 | -0.027 [-0.069, +0.198] | 8 / 3 / 0 of 11 | 0.113 |
| GBT_ship_E_1m − ref_s_lr | E_1m/block | capture20 | +0.030 [-0.114, +0.252] | 8 / 3 / 0 of 11 | 0.113 |
| GBT_ship_E_1m − ref_shipstd128 | E_1m/block | capture20 | -0.032 [-0.132, +0.176] | 8 / 3 / 0 of 11 | 0.113 |
| GBT_ship_E_1m − ref_rsd128 | E_1m/block | capture20 | +0.031 [-0.123, +0.170] | 8 / 1 / 0 of 9 | 0.020 |
| GBT_ship_E_1m − sigma_m_R02 | E_1m/block | capture20 | -0.093 [-0.172, +0.026] | 3 / 4 / 0 of 7 | 0.773 |
| P_GBT_ship_E_1m − P_GBT_noship_E_1m | E_1m/block | auprc | +0.041 [-0.130, +0.054] | 6 / 5 / 0 of 11 | 0.500 |

Pairs excluded from detection metrics (block-scale prevalence of E_1m > 0 outside [5 %, 95 %]): EX1402L2__MGDS_31073 (0.030), FK151121__MGDS_22383 (0.985), KM0923__MGDS_31429 (0.991), NA076__MGDS_32317 (0.003), TN299__MGDS_31253 (0.036), cal_dig_morro_bay__20180426m2_Channel1000 (0.003), cal_dig_morro_bay__20180427m1_Channel700 (0.002), cal_dig_morro_bay__20180427m3_PockmarkNorth (0.002), cal_dig_morro_bay__20180428m1_Cable (0.000), cal_dig_morro_bay__201804_LuciaChica2m (0.007), cal_dig_morro_bay__20190314m4_LuciaChica970m (0.016), cal_dig_morro_bay__20190316m1_BankTop (0.003), cal_dig_morro_bay__20190317m1_1000mGully (0.000), cal_dig_morro_bay__20190317m2_600mGully (0.012), cal_dig_morro_bay__20190318m2_Transect601060m (0.002), cal_dig_morro_bay__20190510m1_BankFlankHoles (0.001), cal_dig_morro_bay__20190510m2_BankFlankIncipCh (0.005), cal_dig_morro_bay__20190511m1_6thHeadlessCany (0.026), cal_dig_morro_bay__20190511m2_BankTopEofCanyon3 (0.002), cal_dig_morro_bay__basin_flank (0.006), cal_dig_morro_bay__pockmark_south (0.001), ccz_so268_1 (0.011), discol_so242_1 (0.009); units with no eligible detection pair: ['lu02', 'lu06', 'lu11', 'nu00', 'vu02']
Ineligible pairs (< 10 tiles with finite target and prediction): []; units dropped: ['lu02', 'lu06', 'lu11', 'nu00', 'vu02']

Training rows per unit after the 200-row cap (block): lu00 200/40352, lu02 200/4693, lu04 200/2732, lu05 200/99693, lu06 200/1045, lu07 200/4708, lu08 200/19603, lu09 200/12076, lu10 200/2636, lu11 200/3166, lu12 200/37156, lu13 200/99786, lu14 200/602, lu15 200/19440, nu00 200/11418, vu02 200/1671

### EW0207 (lu16), tile scale, A_fine — separate one-unit table (2716 rows; models refit on the 16 development units)

| predictor | rho | log-log R² | capture20 |
|---|---|---|---|
| ref_C | 0.478 | 0.245 | 0.409 |
| ref_s_lr | 0.552 | 0.330 | 0.438 |
| ref_shipstd128 | 0.507 | 0.276 | 0.416 |
| ref_rsd128 | nan | nan | nan |
| GBT_ship_A_fine | 0.467 | 0.259 | 0.296 |
| GBT_noship_A_fine | 0.556 | 0.266 | 0.420 |

### EW0207 (lu16), block scale, A_fine — separate one-unit table (41484 rows; models refit on the 16 development units)

| predictor | rho | log-log R² | capture20 |
|---|---|---|---|
| ref_C | 0.406 | 0.173 | 0.399 |
| ref_s_lr | 0.446 | 0.216 | 0.433 |
| ref_shipstd128 | 0.421 | 0.184 | 0.415 |
| ref_rsd128 | nan | nan | nan |
| GBT_ship_A_fine | 0.388 | 0.167 | 0.360 |
| GBT_noship_A_fine | 0.405 | 0.158 | 0.436 |
