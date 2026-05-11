# Stochastic Bathymetric Super-Resolution: Full Method Description

**Project:** Conditional diffusion model for 16× seafloor bathymetry super-resolution  
**Goal:** Upsample 32 m MBES/CRM grids to 2 m resolution; generate stochastic realizations to quantify spatial uncertainty  
**Study area:** Morro Bay, California continental shelf and slope (UTM Zone 10N)

---

## 1. Scientific Context

Multibeam echosounders (MBES) provide broad-area seafloor coverage at 32 m resolution. Autonomous underwater vehicles (AUVs) provide 1–2 m resolution surveys, but only over small targeted areas. The goal is to train a diffusion model on paired patches extracted from the overlapping regions of these two data types, then use it to super-resolve the entire 32 m grid at 2 m resolution. Running N independent samples through the reverse diffusion process yields N stochastic realizations; the pixel-wise standard deviation of those realizations is the uncertainty map.

---

## 2. Data

### Input files (all GeoTIFF, UTM Zone 10N, Float32, elevation in metres; negative = below sea level)

| File | Resolution | Role |
|---|---|---|
| `MorroBay_all_2020_interp_32m_converted.tif` | 32 m | Low-resolution (LR) conditioning grid |
| `*_Topo1m_*_utm.tif` / `*_converted.tif` (×19 surveys) | 1 m originally, resampled to 2 m | High-resolution (HR) ground truth patches |

HR surveys resampled from 1 m to 2 m during patch extraction (bilinear resampling via rasterio). The SR ratio is therefore **16×** (32 m → 2 m).

### Train / val / test split (by survey, not by patch)

The split is **geographic** — all patches from a given survey belong to exactly one split. This prevents data leakage from spatially autocorrelated terrain.

**Test (held out, never seen during training):**
- `LuciaChica_2007_Topo1m_UTM_utm.tif` — open shelf
- `HeadlessCanyon_Topo1m_converted.tif` — canyon head
- `BankTopEastofCanyon3_Topo1m_converted.tif` — bank top

**Validation (one per geomorphic category):**
- `600mGully_Topo1m_UTMinterp_utm.tif` — gully/channel
- `BankFlankHoles_MAUV_Topo1m_UTM_utm.tif` — bank flank
- `PockmarkNorth_MAUV_Topo1m_UTM_utm.tif` — pockmark field
- `Transit_frmPocN_chan_cab_UTM_utm.tif` — multi-environment transect
- `LuciaChica2019_repeat_970m_Topo1m_UTM_utm.tif` — repeat survey

**Train:** All remaining surveys (~14 surveys across the Morro Bay shelf/slope).

Total dataset: ~1.04 M training patches, ~77 K validation patches.

---

## 3. Preprocessing Pipeline

**Script:** `preprocessing/extract_paired_patches.py`  
**Utilities:** `preprocessing/data_utils.py`

### 3.1 LR–HR overlap detection

For each HR survey file:
1. Open the HR raster and get its bounding box in its native CRS.
2. Compare against all available LR grids. Reproject LR bounds to HR CRS if CRS differ (`rasterio.warp.transform_bounds`).
3. Select the LR grid with the greatest spatial overlap (intersection area). If no LR grid overlaps and `--synthetic_lr_fallback` is set, generate a synthetic LR by Gaussian-blurring the HR with `sigma = scale/2` then downsampling by the scale factor (16×).

### 3.2 Patch extraction

For each (HR survey, LR grid) pair within each split:

1. **Fast pre-screen:** Read the HR overlap region at 1/8 resolution to find which row/col bands have valid data. Constrains the sliding window to avoid iterating over all-nodata tiles in large transect surveys.

2. **Sliding window:** Step a `patch_size × patch_size` (default 64×64) window over the valid HR overlap region with stride `stride` (default 32 pixels = 64 m). For each position:

   a. **Read HR patch** (64×64 pixels at 2 m = 128×128 m footprint) using `rasterio.windows.Window`.
   
   b. **Read matching LR patch:** Convert HR pixel offsets to geographic bounds (in HR CRS), reproject to LR CRS if needed, then read the corresponding LR window into a `4×4` array (`lr_patch_size = patch_size // 16 = 4`) using bilinear resampling (`rasterio.enums.Resampling.bilinear`).

   c. **Quality checks:**
      - Reject if >10% of HR or LR pixels are nodata.
      - Reject if HR elevation range < 0.05 m (flat, featureless ocean floor).

   d. **Fill residual NaNs** with the patch's nanmean value.

   e. **Normalize using HR patch statistics (per-patch z-score):**
      ```
      mean = nanmean(hr_patch)
      std  = nanstd(hr_patch)          # if std < 1e-6, set to 1.0
      hr_norm = (hr_patch - mean) / std
      ```

   f. **Bicubic upsample** the 4×4 LR patch to 64×64 using `scipy.ndimage.zoom(order=3)`, then apply the same normalization:
      ```
      lr_up   = bicubic_upsample(lr_patch_raw, 64)   # 4×4 → 64×64 via cubic spline
      lr_norm = (lr_up - mean) / std                  # same mean/std as HR
      ```

   g. **Save as `.npz`:**
      ```
      {
        'lr':    float32 (1, 64, 64)  — bicubic-upsampled, normalized LR
        'hr':    float32 (1, 64, 64)  — normalized HR
        'stats': float32 (2,)         — [mean, std] for denormalization
      }
      ```

**Critical normalization note:** Both LR and HR are normalized using the *HR patch's* mean and std. The LR patch covers the same geographic footprint but represents a smoother (regional slope) signal, so its true mean differs from the HR mean. When normalized by the HR stats, the LR input carries an inherent DC offset relative to the HR target. This is addressed in training via residual mode (Section 5.1).

### 3.3 Data augmentation (training only)

At dataset load time (`BathymetryPatchDataset.__getitem__`), each (HR, LR) pair receives identical random spatial augmentations:
- Horizontal flip with probability 0.5
- Vertical flip with probability 0.5
- 90° rotation (k = 0, 1, 2, 3) uniformly at random

Both arrays receive the same transformation to preserve alignment.

---

## 4. Model Architecture

### 4.1 BathyUNet — Conditional U-Net

**File:** `models/unet.py`  
**Class:** `BathyUNet` (explicit 4-level U-Net; preferred over the generic `UNet` class in the same file)  
**Parameters:** ~22.6 M (with `base_ch=64`)

The U-Net takes as input a 2-channel tensor `[noisy_HR | LR_bicubic]` at every diffusion timestep and predicts the noise `ε` that was added to the clean signal.

#### Channel widths at each spatial level (for `base_ch=64`)

| Level | Spatial size (64×64 input) | Channels | Attention |
|---|---|---|---|
| 0 (encoder) | 64×64 | 64 | No |
| 1 (encoder) | 32×32 | 128 | No |
| 2 (encoder) | 16×16 | 256 | Yes |
| Bottleneck | 8×8 | 512 | Yes |
| 2 (decoder) | 16×16 | 256 | Yes |
| 1 (decoder) | 32×32 | 128 | No |
| 0 (decoder) | 64×64 | 64 | No |

#### Key building blocks

**SinusoidalTimeEmbedding:** Encodes the integer diffusion timestep `t` as a sinusoidal positional embedding of dimension `base_ch=64`:
```
freqs = exp(-log(10000) * arange(32) / 31)
emb   = cat([sin(t * freqs), cos(t * freqs)])   # dim=64
```
Projected to `time_dim=256` via a two-layer MLP with SiLU activations.

**ResBlock:** The core building block. Each block applies:
1. `GroupNorm(min(32, in_ch), in_ch)` → `SiLU` → `Conv2d(in_ch, out_ch, 3, padding=1)`
2. Adds time embedding: `h += Linear(time_dim, out_ch)(SiLU(t_emb))[:, :, None, None]`
3. `GroupNorm` → `Dropout(0.1)` → `SiLU` → `Conv2d(out_ch, out_ch, 3, padding=1)`
4. Residual: `h += Conv2d(in_ch, out_ch, 1)(x)` if channels change, else identity skip

**SelfAttention:** Multi-head self-attention (4 heads) applied at 16×16 and 8×8 levels:
```
x_norm = GroupNorm(x)
h = reshape to (B, HW, C) → MultiheadAttention → reshape to (B, C, H, W)
output = x + h
```

**Downsample:** `Conv2d(ch, ch, 3, stride=2, padding=1)` (strided convolution, not max-pool)

**Upsample:** `F.interpolate(scale_factor=2, mode='nearest')` → `Conv2d(ch, ch, 3, padding=1)`

#### Forward pass (explicit structure)

```
Input: x (B, 2, 64, 64), t (B,)
  → in_conv (Conv2d 2→64)
  → enc0a, enc0b (ResBlock 64→64, 64→64), save skip s0   # 64×64
  → down0 (stride-2 conv)
  → enc1a, enc1b (ResBlock 64→128, 128→128), save skip s1 # 32×32
  → down1
  → enc2a, enc2b (ResBlock 128→256, 256→256), attn2, save skip s2  # 16×16
  → down2
  → bot1 (ResBlock 256→512), bot_attn (SelfAttention), bot2 (ResBlock 512→512)  # 8×8
  → up2 (Upsample 512→512)
  → cat([h, s2]) → dec2a (ResBlock 768→256), dec2b (ResBlock 256→256), dattn2  # 16×16
  → up1
  → cat([h, s1]) → dec1a (ResBlock 384→128), dec1b (ResBlock 128→128)  # 32×32
  → up0
  → cat([h, s0]) → dec0a (ResBlock 192→64), dec0b (ResBlock 64→64)    # 64×64
  → GroupNorm → SiLU → Conv2d(64→1, kernel=1)
Output: ε_pred (B, 1, 64, 64)
```

### 4.2 GaussianDiffusion — DDPM Forward/Reverse Process

**File:** `models/diffusion.py`  
**Class:** `GaussianDiffusion`

Wraps `BathyUNet` with the DDPM diffusion process. All schedule tensors are registered as `nn.Module` buffers (move to GPU automatically).

#### Noise schedule

Linear schedule: `β_t` linearly from `β_start=1e-4` to `β_end=0.02` over `T=1000` steps.

Derived quantities:
```
αt    = 1 - βt
ᾱt   = cumprod(α1 ... αt)          # sqrt_alphas_cumprod
√ᾱt  = sqrt(ᾱt)                    # used in q_sample
√(1-ᾱt) = sqrt(1-ᾱt)              # used in q_sample
posterior_variance = βt * (1-ᾱ_{t-1}) / (1-ᾱt)   # for DDPM reverse step
```

#### Forward process — `q_sample(x0, t, noise)`

Samples a noisy version of the clean patch at timestep t:
```
x_t = √ᾱt * x0 + √(1-ᾱt) * ε,   ε ~ N(0, I)
```

#### Training step — `p_losses(x0, lr_up, t)`

1. Sample noise `ε ~ N(0, I)` same shape as `x0`.
2. Compute `x_t` via `q_sample`.
3. Concatenate `[x_t, lr_up]` → `(B, 2, H, W)`.
4. Pass through UNet: `ε_pred = model([x_t, lr_up], t)`.
5. Return `(ε_pred, ε)` — loss computed externally.

#### Reverse process — DDIM sampling `sample_ddim(lr_up, n_realizations, n_steps=50, eta=0.0, add_lr=False)`

Uses DDIM (Song et al. 2020) with 50 evenly-spaced timesteps from T-1 to 0:

```
for t_cur, t_prev in reversed timestep pairs:
    ε = model([x, lr_rep], t_cur)
    x0_pred = (x - √(1-ᾱ_cur) * ε) / √ᾱ_cur    # clamp to [-4, 4]
    σ = eta * sqrt((1-ᾱ_prev)/(1-ᾱ_cur)) * sqrt(1 - ᾱ_cur/ᾱ_prev)
    dir_xt = sqrt(1 - ᾱ_prev - σ²) * ε
    x = √ᾱ_prev * x0_pred + dir_xt + σ * N(0,I)

if add_lr:
    x = x + lr_rep    # residual mode: reconstruct full HR
```

With `eta=0` (default), sampling is deterministic given initial noise. Multiple realizations are generated by starting from different random noise tensors `x ~ N(0,I)`.

---

## 5. Training Procedure

**Script:** `training/train.py`  
**Config:** `training/config.py`

### 5.1 Residual mode (key design decision)

**Problem:** Without residual mode, the model must predict the full normalized HR patch. Because the LR patch covers a larger regional footprint, its mean (after normalization by HR stats) carries a large DC offset from zero. The model tends to spend most of its capacity learning this offset correction, leaving little capacity for high-frequency texture.

**Solution:** In residual mode (`residual_mode=True`, default), the target is `x0 = HR - LR` (computed in normalized space). The model learns to predict the fine-scale texture residual that must be added to the bicubic LR to recover the HR. At inference, the LR is added back: `SR = model_output + LR`.

### 5.2 Loss function — CombinedDiffusionLoss

**File:** `models/losses.py`

Four terms, all computed per batch:

**1. Noise prediction loss (primary):** L1 between predicted and actual noise:
```
L_noise = mean(|ε_pred - ε_true|)
```
L1 chosen over L2 for robustness to elevation outliers common in sonar data.

**2. Spectral loss (weight 1.0):** L1 on the log-power spectrum of the predicted x0:
```
x0_pred = (x_t - √(1-ᾱt) * ε_pred) / √ᾱt    # clamped to [-4, 4]
L_spectral = mean(|log(|FFT(x0_pred)|² + 1e-8) - log(|FFT(x0_true)|² + 1e-8)|)
```
Forces correct spatial frequency content — prevents the model from producing spectrally flat (blurry) outputs.

**3. Gradient loss (weight 0.5):** Sobel edge loss on predicted x0:
```
L_gradient = mean(|Kx*x0_pred - Kx*x0_true|) + mean(|Ky*x0_pred - Ky*x0_true|)
```
Where Kx, Ky are fixed 3×3 Sobel kernels. Directly penalises missing fine-scale edges (canyon walls, bedform crests).

**4. x0 reconstruction loss (weight 0.5):** Direct L1 on the reconstructed clean image:
```
L_x0 = mean(|x0_pred - x0_true|)
```

**Total:** `L = L_noise + L_spectral + L_gradient + L_x0`

### 5.3 Optimizer and scheduler

- **Optimizer:** AdamW, `lr=1e-4`, `betas=(0.9, 0.999)`, `weight_decay=1e-4`
- **Scheduler:** Cosine annealing over all training steps, `eta_min=1e-6`
- **Gradient clipping:** Max norm 1.0

### 5.4 EMA (Exponential Moving Average)

A shadow copy of the `GaussianDiffusion` object (including all buffers) is maintained with `decay=0.9999`. The EMA weights are updated after every optimizer step:
```
shadow_param = 0.9999 * shadow_param + 0.0001 * model_param
```
All inference uses the EMA model. EMA is critical for diffusion model quality — raw weights produce noisier outputs.

### 5.5 Checkpoint format

Saved as a dict with keys: `model` (raw UNet state_dict), `ema` (EMA GaussianDiffusion state_dict — includes all schedule buffers), `optimizer`, `scheduler`, `epoch`, `global_step`.

Checkpoints are saved every 5000 steps and at epoch end. Training auto-resumes from the latest step checkpoint if one exists in `out_dir/checkpoints/`.

### 5.6 Hyperparameters (v4 training run)

| Parameter | Value |
|---|---|
| `base_ch` | 64 |
| `time_dim` | 256 |
| `n_timesteps` | 1000 |
| `beta_start` | 1e-4 |
| `beta_end` | 0.02 |
| `batch_size` | 64 |
| `n_epochs` | 300 |
| `learning_rate` | 1e-4 |
| `ema_decay` | 0.9999 |
| `spectral_weight` | 1.0 |
| `gradient_weight` | 0.5 |
| `x0_weight` | 0.5 |
| `grad_clip` | 1.0 |
| `dropout` | 0.1 |
| `patch_size` | 64 (HR pixels = 128 m footprint) |
| `residual_mode` | True |
| Training patches | ~1.04 M |
| Validation patches | ~77 K |

Training ran for ~96 hours on a single NVIDIA GPU (Stanford Sherlock HPC, SLURM partition `serc`).

---

## 6. Inference Procedure

**Script:** `inference/generate_realizations.py`

### 6.1 Tiled inference over a region

For a given UTM bounding box:

1. Convert bbox to LR pixel coordinates.
2. Extract the LR sub-array for the region.
3. Slide a `4×4 LR tile` (= 64×64 HR patch footprint) over the region with stride 2 LR pixels (= 64 m stride, heavy overlap):
   - Skip tiles with >50% nodata.
   - Fill residual NaNs with local nanmean.
   - Normalize LR tile using HR-style z-score: `normalize_patch(tile)` → `(mean, std)`.
   - Bicubic upsample to 64×64: `bicubic_upsample(tile_norm, 64)`.
   - Run `diffusion.sample_ddim(lr_tensor, n_realizations=N, n_steps=50, add_lr=True)` → `(N, 1, 64, 64)` normalized residuals, with LR added back.
   - Denormalize: `samples * std + mean` → physical elevation in metres.
4. Accumulate samples into `(N, hr_h, hr_w)` arrays using **Gaussian-weighted blending** to smooth tile boundaries:
   ```
   weight(x,y) = exp(-(x² + y²) / (2*(patch_size/4)²))
   accum[n, row:row+64, col:col+64] += samples[n] * gauss_w
   weight_sum[row:row+64, col:col+64] += gauss_w
   final = accum / weight_sum
   ```
5. Save each of the N realizations as a separate GeoTIFF with correct georeference.

### 6.2 Uncertainty maps

Pixel-wise statistics across N realizations:
- `SR_mean = mean(realizations, axis=0)` — best estimate of HR elevation
- `SR_std  = std(realizations, axis=0)` — uncertainty (higher = more variable across samples)
- `SR_CoV  = SR_std / |SR_mean|` — coefficient of variation (relative uncertainty)

---

## 7. Evaluation

**Scripts:** `evaluation/metrics.py`, `evaluation/evaluate.py`  
**Diagnostic scripts:** `sherlock/patch_diagnostic_v4.sbatch`, `sherlock/region_diagnostic_v4.sbatch`

### 7.1 Patch-level metrics (64×64 normalized patches)

- **RMSE** (σ units): `sqrt(mean((SR - HR)²))` per patch, compared against LR RMSE. Note: LR RMSE is inflated because the LR represents regional slope (large dynamic range) while HR patches capture local relief. This makes the improvement look larger than it is in physical metres.
- **PSD comparison:** Radially-averaged 2D power spectral density of LR, SR, and HR patches. A good SR model should shift the SR curve from the LR toward the HR at mid-to-high frequencies.
- **RMSE scatter:** Plot of `RMSE_SR` vs `RMSE_LR` per patch. A flat relationship (no correlation) indicates the model outputs a near-constant prediction regardless of input (failure mode). A collinear relationship below the 1:1 line indicates genuine input-dependent improvement.

### 7.2 Region-level evaluation (physical metres)

For each val/test survey:
1. Run tiled inference to produce SR mean and std.
2. Resample the HR survey from 1 m to 2 m (`rasterio.warp.reproject`, bilinear).
3. Mask nodata using raster nodata metadata + sentinel value detection (-9999, -9998, -32767) + plausibility clamp (-2000 to +50 m).
4. Extract profiles through the highest-variance row and column.
5. Compare LR, SR mean ± 1σ, and HR along each profile in physical metres.

### 7.3 Uncertainty calibration

Bin SR std dev into 20 percentile bins. For each bin, compute mean absolute error `|SR - HR|`. Plot uncertainty (x-axis) vs actual error (y-axis). A well-calibrated model follows the diagonal. Points clustered in the top-left (error >> uncertainty) indicate the model is overconfident — it produces nearly identical realizations (low std) but is systematically wrong.

---

## 8. Known Limitations and Failure Modes

1. **16× SR ratio is very aggressive.** The 4×4 LR patch (covering 128×128 m) does not contain information about the locations of individual seafloor features at 2 m scale. Features smaller than the LR pixel (32 m) — pockmarks (50–200 m diameter), small bedforms — cannot be deterministically recovered from the LR data.

2. **Near-deterministic outputs.** Despite being a diffusion model, the inter-realization std is very small (~0.1–0.7 m) while actual errors are ~3–4 m. The model has collapsed toward a near-point-estimate rather than sampling from the true posterior P(HR|LR). This makes the uncertainty maps unreliable for survey planning.

3. **Overconfident uncertainty.** Calibration shows uncertainty does rank-order actual error correctly (Pearson r ≈ 0.9) but the absolute scale is wrong by ~5–10×. Uncertainty maps show geomorphic pattern (higher uncertainty in rougher terrain) but cannot be used quantitatively.

4. **Bicubic LR as model input.** The model receives the bicubic-upsampled 4×4 LR (already smooth 64×64), not the raw 4×4. This means the model learns to add texture on top of an already-committed smooth baseline, rather than jointly learning the upsampling and texture generation.

5. **Patch normalization artifact.** Per-patch z-score normalization uses the HR patch mean and std. Since the LR and HR patches cover the same area but the LR captures regional slope, the LR's true mean differs from the HR's. The residual mode partially addresses this by making the model predict HR − LR directly.

---

## 9. Project File Structure

```
GAN_Bathymetry_Project/
├── data_for_model/                      # Raw GeoTIFF inputs
│   ├── MorroBay_all_2020_interp_32m_converted.tif   # LR grid
│   └── *_Topo1m_*.tif (×19)                         # HR AUV surveys
├── preprocessing/
│   ├── extract_paired_patches.py        # Patch extraction pipeline
│   └── data_utils.py                    # normalize_patch, bicubic_upsample, Dataset
├── models/
│   ├── unet.py                          # BathyUNet (explicit 4-level) + generic UNet
│   ├── diffusion.py                     # GaussianDiffusion (DDPM + DDIM)
│   └── losses.py                        # CombinedDiffusionLoss
├── training/
│   ├── config.py                        # All hyperparameters (dataclass)
│   └── train.py                         # Training loop, EMA, checkpointing
├── inference/
│   └── generate_realizations.py         # Tiled inference + Gaussian blending
├── sherlock/
│   ├── environment.yml                  # Conda env (PyTorch 2.x + CUDA + rasterio)
│   ├── train_v4.sbatch                  # GPU training job (v4, residual mode)
│   ├── patch_diagnostic_v4.sbatch       # PSD + RMSE patch diagnostics
│   ├── region_diagnostic_v4.sbatch      # Full-region SR + profile + calibration
│   └── profile_diagnostic_v4.sbatch     # Topographic transect figure
└── results/figures_v4/                  # Output figures
```

---

## 10. Key References

- **DDPM:** Ho et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
- **DDIM:** Song et al. (2020). Denoising Diffusion Implicit Models. ICLR 2021.
- **SR3:** Saharia et al. (2021). Image Super-Resolution via Iterative Refinement. IEEE TPAMI.
- **Spectral loss:** Isola et al. (2017); Durall et al. (2020) on spectral artifacts in generative models.
