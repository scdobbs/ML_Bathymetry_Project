
### §2 T_ship / mosaic (1563 tiles, 7 native units)

| predictor | within-unit R² | ρ median | ρ per unit NA090 / NA080 / TN399 / RR1506 / AT37-13 / FK181031 / TN299 | capture20 median | capture per unit |
|---|---|---|---|---|---|
| log s_lr | 0.521 | 0.714 | 0.63 / 0.82 / 0.64 / 0.76 / 0.50 / 0.82 / 0.71 | 0.724 | 0.74 / 0.72 / 0.66 / 0.75 / 0.49 / 0.87 / 0.50 |
| log sigma_norm | 0.150 | 0.117 | 0.12 / 0.11 / 0.43 / -0.09 / 0.21 / -0.30 / 0.70 | 0.251 | 0.13 / 0.25 / 0.38 / 0.23 / 0.46 / 0.06 / 0.40 |
| log sigma_m (R02) | 0.799 | 0.846 | 0.82 / 0.92 / 0.90 / 0.83 / 0.67 / 0.85 / 0.94 | 0.755 | 0.80 / 0.75 / 0.70 / 0.79 / 0.56 / 0.87 / 0.57 |
| roughness-3 (R02) | 0.749 | 0.863 | 0.66 / 0.89 / 0.86 / 0.86 / 0.47 / 0.85 / 0.90 | 0.743 | 0.77 / 0.74 / 0.67 / 0.78 / 0.51 / 0.88 / 0.55 |
| s_lr + roughness-3 | 0.756 | 0.874 | 0.69 / 0.90 / 0.86 / 0.87 / 0.53 / 0.88 / 0.89 | 0.735 | 0.77 / 0.73 / 0.66 / 0.79 / 0.51 / 0.89 / 0.56 |

Differences (bootstrap over units, 2,000 resamples):

| comparison | Δ within R² [95% CI] | Δ ρ median [CI] | Δ capture median [CI] | units a better (ρ / capture) of N |
|---|---|---|---|---|
| log sigma_m (R02) - log s_lr | +0.278 [+0.132, +0.377] | +0.132 [+0.028, +0.265] | +0.031 [+0.031, +0.070] | 7 / 6 of 7 |
| log sigma_norm - log s_lr | -0.370 [-0.603, -0.111] | -0.596 [-0.856, -0.205] | -0.473 [-0.619, -0.106] | 0 / 0 of 7 |
| s_lr + roughness-3 - roughness-3 (R02) | +0.008 [-0.005, +0.033] | +0.012 [+0.000, +0.027] | -0.008 [-0.008, +0.007] | 6 / 4 of 7 |
| log sigma_m (R02) - s_lr + roughness-3 | +0.042 [-0.030, +0.092] | -0.028 [-0.046, +0.127] | +0.020 [-0.002, +0.036] | 5 / 5 of 7 |

Fraction of σ_m's within-unit R² available from log s_lr alone: **0.652** (0.521 / 0.799). Within-unit partial correlation of log σ_norm with log T_ship given log s_lr: pooled r = **+0.780** (partial R² 0.608), 95% CI over units [+0.595, +0.871]; per unit: NA090 +0.81, NA080 +0.72, TN399 +0.82, RR1506 +0.69, AT37-13 +0.62, FK181031 +0.40, TN299 +0.91.

### §2 T_cnn / mosaic (1563 tiles, 7 native units)

| predictor | within-unit R² | ρ median | ρ per unit NA090 / NA080 / TN399 / RR1506 / AT37-13 / FK181031 / TN299 | capture20 median | capture per unit |
|---|---|---|---|---|---|
| log s_lr | 0.484 | 0.667 | 0.56 / 0.84 / 0.59 / 0.73 / 0.47 / 0.83 / 0.67 | 0.726 | 0.73 / 0.73 / 0.67 / 0.75 / 0.49 / 0.86 / 0.49 |
| log sigma_norm | 0.185 | 0.187 | 0.19 / 0.09 / 0.51 / -0.06 / 0.23 / -0.31 / 0.74 | 0.240 | 0.14 / 0.24 / 0.47 / 0.24 / 0.48 / 0.06 / 0.43 |
| log sigma_m (R02) | 0.792 | 0.857 | 0.75 / 0.94 / 0.90 / 0.80 / 0.64 / 0.86 / 0.93 | 0.758 | 0.80 / 0.76 / 0.75 / 0.78 / 0.59 / 0.86 / 0.58 |
| roughness-3 (R02) | 0.736 | 0.850 | 0.64 / 0.90 / 0.85 / 0.86 / 0.48 / 0.85 / 0.88 | 0.738 | 0.77 / 0.74 / 0.71 / 0.78 / 0.53 / 0.87 / 0.57 |
| s_lr + roughness-3 | 0.737 | 0.864 | 0.66 / 0.91 / 0.85 / 0.86 / 0.51 / 0.87 / 0.88 | 0.741 | 0.76 / 0.74 / 0.71 / 0.79 / 0.54 / 0.87 / 0.57 |

Differences (bootstrap over units, 2,000 resamples):

| comparison | Δ within R² [95% CI] | Δ ρ median [CI] | Δ capture median [CI] | units a better (ρ / capture) of N |
|---|---|---|---|---|
| log sigma_m (R02) - log s_lr | +0.308 [+0.139, +0.415] | +0.190 [+0.027, +0.315] | +0.032 [+0.032, +0.094] | 7 / 6 of 7 |
| log sigma_norm - log s_lr | -0.299 [-0.600, -0.008] | -0.480 [-0.797, -0.084] | -0.486 [-0.609, -0.026] | 1 / 0 of 7 |
| s_lr + roughness-3 - roughness-3 (R02) | +0.000 [-0.009, +0.019] | +0.014 [-0.005, +0.018] | +0.003 [-0.001, +0.003] | 5 / 5 of 7 |
| log sigma_m (R02) - s_lr + roughness-3 | +0.056 [-0.024, +0.104] | -0.007 [-0.063, +0.098] | +0.017 [-0.003, +0.039] | 5 / 5 of 7 |

Fraction of σ_m's within-unit R² available from log s_lr alone: **0.611** (0.484 / 0.792). Within-unit partial correlation of log σ_norm with log T_cnn given log s_lr: pooled r = **+0.806** (partial R² 0.649), 95% CI over units [+0.602, +0.889]; per unit: NA090 +0.79, NA080 +0.74, TN399 +0.87, RR1506 +0.68, AT37-13 +0.62, FK181031 +0.41, TN299 +0.92.

### §2 T_ship / tile (1563 tiles, 7 native units)

| predictor | within-unit R² | ρ median | ρ per unit NA090 / NA080 / TN399 / RR1506 / AT37-13 / FK181031 / TN299 | capture20 median | capture per unit |
|---|---|---|---|---|---|
| log s_lr | 0.521 | 0.714 | 0.63 / 0.82 / 0.64 / 0.76 / 0.50 / 0.82 / 0.71 | 0.724 | 0.74 / 0.72 / 0.66 / 0.75 / 0.49 / 0.87 / 0.50 |
| log sigma_norm | 0.240 | 0.284 | 0.24 / 0.28 / 0.42 / -0.07 / 0.42 / -0.03 / 0.74 | 0.369 | 0.29 / 0.37 / 0.42 / 0.24 / 0.51 / 0.33 / 0.41 |
| log sigma_m (R02) | 0.758 | 0.817 | 0.79 / 0.91 / 0.87 / 0.81 / 0.67 / 0.82 / 0.93 | 0.743 | 0.80 / 0.74 / 0.71 / 0.78 / 0.49 / 0.86 / 0.54 |
| roughness-3 (R02) | 0.749 | 0.863 | 0.66 / 0.89 / 0.86 / 0.86 / 0.47 / 0.85 / 0.90 | 0.743 | 0.77 / 0.74 / 0.67 / 0.78 / 0.51 / 0.88 / 0.55 |
| s_lr + roughness-3 | 0.756 | 0.874 | 0.69 / 0.90 / 0.86 / 0.87 / 0.53 / 0.88 / 0.89 | 0.735 | 0.77 / 0.73 / 0.66 / 0.79 / 0.51 / 0.89 / 0.56 |

Differences (bootstrap over units, 2,000 resamples):

| comparison | Δ within R² [95% CI] | Δ ρ median [CI] | Δ capture median [CI] | units a better (ρ / capture) of N |
|---|---|---|---|---|
| log sigma_m (R02) - log s_lr | +0.237 [+0.093, +0.335] | +0.103 [-0.001, +0.236] | +0.019 [+0.019, +0.061] | 6 / 6 of 7 |
| log sigma_norm - log s_lr | -0.280 [-0.579, -0.021] | -0.429 [-0.830, -0.218] | -0.355 [-0.459, -0.084] | 1 / 1 of 7 |
| s_lr + roughness-3 - roughness-3 (R02) | +0.008 [-0.005, +0.033] | +0.012 [+0.000, +0.027] | -0.008 [-0.008, +0.007] | 6 / 4 of 7 |
| log sigma_m (R02) - s_lr + roughness-3 | +0.002 [-0.071, +0.055] | -0.058 [-0.064, +0.101] | +0.008 [-0.019, +0.051] | 5 / 3 of 7 |

Fraction of σ_m's within-unit R² available from log s_lr alone: **0.687** (0.521 / 0.758). Within-unit partial correlation of log σ_norm with log T_ship given log s_lr: pooled r = **+0.752** (partial R² 0.566), 95% CI over units [+0.538, +0.852]; per unit: NA090 +0.77, NA080 +0.73, TN399 +0.78, RR1506 +0.69, AT37-13 +0.61, FK181031 +0.14, TN299 +0.90.

### §2 T_cnn / tile (1563 tiles, 7 native units)

| predictor | within-unit R² | ρ median | ρ per unit NA090 / NA080 / TN399 / RR1506 / AT37-13 / FK181031 / TN299 | capture20 median | capture per unit |
|---|---|---|---|---|---|
| log s_lr | 0.498 | 0.672 | 0.58 / 0.85 / 0.64 / 0.72 / 0.46 / 0.83 / 0.67 | 0.712 | 0.75 / 0.71 / 0.68 / 0.75 / 0.47 / 0.85 / 0.55 |
| log sigma_norm | 0.278 | 0.286 | 0.29 / 0.22 / 0.46 / -0.02 / 0.47 / -0.02 / 0.78 | 0.329 | 0.27 / 0.33 / 0.51 / 0.25 / 0.55 / 0.28 / 0.45 |
| log sigma_m (R02) | 0.759 | 0.831 | 0.74 / 0.92 / 0.90 / 0.77 / 0.64 / 0.83 / 0.92 | 0.765 | 0.83 / 0.73 / 0.77 / 0.77 / 0.50 / 0.85 / 0.59 |
| roughness-3 (R02) | 0.718 | 0.839 | 0.66 / 0.87 / 0.84 / 0.85 / 0.51 / 0.79 / 0.89 | 0.718 | 0.81 / 0.72 / 0.69 / 0.78 / 0.54 / 0.81 / 0.63 |
| s_lr + roughness-3 | 0.720 | 0.839 | 0.68 / 0.88 / 0.84 / 0.85 / 0.56 / 0.82 / 0.88 | 0.711 | 0.81 / 0.71 / 0.69 / 0.79 / 0.55 / 0.83 / 0.62 |

Differences (bootstrap over units, 2,000 resamples):

| comparison | Δ within R² [95% CI] | Δ ρ median [CI] | Δ capture median [CI] | units a better (ρ / capture) of N |
|---|---|---|---|---|
| log sigma_m (R02) - log s_lr | +0.261 [+0.093, +0.369] | +0.159 [+0.003, +0.260] | +0.053 [+0.020, +0.081] | 7 / 6 of 7 |
| log sigma_norm - log s_lr | -0.220 [-0.593, +0.070] | -0.387 [-0.744, -0.110] | -0.383 [-0.484, -0.039] | 2 / 1 of 7 |
| s_lr + roughness-3 - roughness-3 (R02) | +0.002 [-0.013, +0.033] | +0.000 [+0.000, +0.028] | -0.007 [-0.007, +0.007] | 6 / 3 of 7 |
| log sigma_m (R02) - s_lr + roughness-3 | +0.039 [-0.024, +0.077] | -0.008 [-0.084, +0.061] | +0.054 [-0.028, +0.078] | 6 / 4 of 7 |

Fraction of σ_m's within-unit R² available from log s_lr alone: **0.656** (0.498 / 0.759). Within-unit partial correlation of log σ_norm with log T_cnn given log s_lr: pooled r = **+0.786** (partial R² 0.618), 95% CI over units [+0.556, +0.881]; per unit: NA090 +0.75, NA080 +0.68, TN399 +0.84, RR1506 +0.69, AT37-13 +0.64, FK181031 +0.14, TN299 +0.92.