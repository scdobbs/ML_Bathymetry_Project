# Method — Stochastic Super-Resolution of Seafloor Bathymetry via Rectified Flow

This document describes the model, training procedure, sampler, and
evaluation protocol used in this project. It is written to match the
code as it actually exists (post-rewrite); any discrepancy is a bug —
please flag it.

## 1. Problem setup

Let *x*₀ ∈ ℝ^(H×W) denote a high-resolution (HR) bathymetric patch at
2 m pixel size and *y* ∈ ℝ^(h×w) the co-located low-resolution (LR)
patch at 32 m pixel size, with H = h · 16 and W = w · 16. We model the
posterior *p*(x₀ | y) — not a deterministic regression — so that
repeated sampling produces an ensemble of plausible HR maps whose
pixel-wise variance directly measures the uncertainty in the
super-resolution problem.

Patches are extracted from overlapping survey footprints
(`preprocessing/extract_paired_patches.py`), per-patch z-score
normalized using HR statistics, and saved with both the raw 32 × 32 LR
and a bicubic-upsampled 512 × 512 LR (the latter retained for ablation
only — the model consumes raw LR).

Default patch geometry:
- HR patch: 512 × 512 pixels (1024 m × 1024 m at 2 m resolution)
- LR patch: 32 × 32 pixels (1024 m × 1024 m at 32 m resolution)
- SR ratio: 16× (configurable as `cfg.sr_ratio`)
- Stride: 256 HR pixels (50% overlap)

Splits are by-survey, not random patch — training, validation, and test
draw from disjoint surveys so spatial autocorrelation never leaks
between splits.

## 2. Generative model — rectified flow

We adopt the **linear interpolant** form of flow matching. Given a
clean signal *x*₀ ~ p(x₀ | y) and Gaussian noise *z* ~ 𝒩(0, I), define

$$x_t = (1 - t)\, x_0 + t\, z, \qquad t \in [0, 1].$$

The marginal velocity along this path is the constant

$$v(x_t, t) = \frac{dx_t}{dt} = z - x_0.$$

We train a neural network *v*<sub>θ</sub>(x_t, t, y) to predict this
velocity, conditioned on *y*. The training loss is

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{\,x_0,\,y,\,t,\,z}\bigl[\;
\|\, v_\theta(x_t, t, y) - (z - x_0)\,\|^2 \;\bigr].$$

Time *t* is sampled from a logit-normal distribution
*t* = σ(*u*), *u* ~ 𝒩(0, 1), which concentrates supervision in the
middle of the interval (default `cfg.logit_normal_mean = 0.0`,
`cfg.logit_normal_std = 1.0`). Uniform sampling is available via
`cfg.time_sampling = "uniform"` for ablation.

This formulation has three properties we want:
1. The target velocity is analytic and noise-free; no schedule design is
   required (no β, no α̅).
2. Sampling reduces to integrating an ODE / SDE from *t* = 1 (pure
   noise) to *t* = 0 (clean), which permits flexible 2nd-order
   schemes and much shorter chains than DDPM.
3. The model architecture and conditioning pathway are identical to
   what one would build for a denoising diffusion model — only the
   training target and sampler differ.

### Auxiliary spectral loss (off by default)

Optionally we add a radially-binned log-power-spectrum loss on the
reconstructed clean signal *x̂*₀ = x_t − t · v_θ:

$$\mathcal{L}_{\text{spec}} = \mathbb{E}\Bigl[\;
\bigl\|\, \log \mathrm{P}_\text{radial}(\hat x_0) - \log \mathrm{P}_\text{radial}(x_0) \,\bigr\|_1
\Bigr],$$

where P_radial is the angularly-averaged 2D FFT power. When SNR
weighting is on (`cfg.spectral_loss_snr_weighting = True`), each
sample's contribution is scaled by (1 − t)² so the spectral term
dominates where *x̂*₀ is reliable. The combined loss is

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda \, \mathcal{L}_{\text{spec}},$$

with `cfg.spectral_loss_weight = λ` (default **0**, i.e., the spectral
term is off unless explicitly enabled for an ablation).

The "2d" mode (per-bin L1 over the un-binned 2D log power) is retained
for ablation against the radial form.

## 3. Architecture — hybrid CNN/transformer U-Net

`models/unet.py::BathyUNet`. Six levels at base_ch=64 with multipliers
`(1, 2, 4, 6, 8, 12)`:

| Level | Spatial (at patch_size=512) | Channels | Block type |
|---|---|---|---|
| 0 | 512 × 512 | 64 | 2 × ResBlock (AdaLN-Zero) |
| 1 | 256 × 256 | 128 | 2 × ResBlock |
| 2 | 128 × 128 | 256 | 2 × ResBlock |
| 3 | 64 × 64 | 384 | 2 × ResBlock |
| 4 | 32 × 32 | 512 | 2 × DiT block (self-attn + **cross-attn(LR)** + MLP) |
| 5 (bottleneck) | 16 × 16 | 768 | 2 × DiT block |

Down/Upsample modules change spatial resolution **and** channel count
(stride-2 3 × 3 conv for down, nearest-interpolate + 3 × 3 conv for up),
so block channels stay constant within each level — necessary for the
transformer levels and applied uniformly for symmetry.

### LR conditioning encoder

A small CNN lifts the raw LR (32 × 32) into two K/V feature scales:
- f_L4 ∈ ℝ^(1024 × 512) — flattened (32 × 32) at the level-4 channel
  count, used by all level-4 transformer blocks (encoder and decoder).
- f_L5 ∈ ℝ^(256 × 768) — flattened (16 × 16), used by the level-5
  bottleneck blocks.

The transformer blocks attend to these features via standard
multi-head cross-attention with 8 heads, using
`F.scaled_dot_product_attention` (the Flash backend dispatches
automatically on H100).

### Time / noise-level conditioning — AdaLN-Zero

Continuous *t* is encoded with a sinusoidal embedding (scaled by 1000 to
spread the half-dim frequency range across [0, 1]) and projected through
a 2-layer MLP to a 256-dim representation *c*. Each sub-block (every
ResBlock; each of self-attn, cross-attn, MLP in transformer blocks) has
its own `AdaLN-Zero(c) → (shift, scale, gate)` linear projection,
initialized to zero so that at initialization every block contributes
zero and the residual connection makes it identity. The second conv
inside each ResBlock and the final output Conv2d are also
zero-initialized.

Total: **85,647,000 trainable parameters**.

### Classifier-free guidance

A `null_lr` buffer of zeros at the LR shape lives on the `RectifiedFlow`
wrapper. During training, each sample's LR is replaced with `null_lr`
with probability `cfg.cfg_dropout_prob = 0.1`. At inference, the
sampler can request a guidance scale *s* ≠ 1, in which case each Heun
step runs a batched-x2 forward (cond + uncond) and the guided velocity
is

$$v_{\text{guided}} = v_{\text{uncond}} + s \, (v_{\text{cond}} - v_{\text{uncond}}).$$

Default at inference is `cfg.cfg_scale = 1.5`.

## 4. Training procedure

`training/train.py`. One epoch processes every patch in the training
split, in batches. Per micro-batch:

1. Sample *t* (logit-normal), *z* ~ 𝒩(0, I).
2. Form x_t = (1 − t) x_0 + t z and v_target = z − x_0.
3. Replace LR with null_lr per-sample with probability 0.1 (CFG dropout).
4. Forward through the U-Net (wrapped in `torch.autocast(bf16)` on CUDA).
5. Compute L = L_FM + λ L_spec.
6. Divide by `grad_accum_steps`, backward, accumulate.
7. On the last micro-batch of the cycle: clip gradients (global norm 1.0),
   step AdamW, step cosine LR scheduler, zero grads, update EMA shadow
   (after `ema_warmup_steps = 1000`).

DDP wraps `flow.model` when `WORLD_SIZE > 1`. The same script runs at
WORLD_SIZE=1 (no DDP init) with no code changes. DDP `no_sync()` skips
all-reduce on intermediate micro-batches inside a grad-accum cycle.

EMA uses `torch._foreach_mul_` / `torch._foreach_add_` for fused
multi-tensor updates; the shadow is a deep-copy taken before
DDP/compile wrapping so checkpoint keys are clean.

`torch.compile(flow.model, mode='default')` is applied on CUDA after
DDP wrapping (matches PyTorch's documented order).

### Default hyperparameters (`training/config.py`)

| Knob | Default | Notes |
|---|---|---|
| optimizer | AdamW | β = (0.9, 0.999), weight_decay = 1e-4 |
| learning_rate | 2e-4 | cosine to 1e-6 over `n_epochs` |
| batch_size (per GPU) | 16 | |
| n_epochs | 500 | |
| grad_clip | 1.0 | global L2 norm |
| ema_decay | 0.9999 | |
| ema_warmup_steps | 1000 | |
| precision | "bf16" | autocast; no GradScaler needed |
| compile_model | True | `torch.compile` on the U-Net |
| grad_accum_steps | 1 | |
| spectral_loss_weight | 0.0 | off by default; ablation knob |
| cfg_dropout_prob | 0.1 | |
| time_sampling | "logit_normal" | mean 0, std 1 |
| num_workers | 16 | persistent + prefetch_factor=4 |

## 5. Sampling — Heun integration

`models/diffusion.py::RectifiedFlow.sample`. Initial state
x₁ ~ 𝒩(0, I) at *t* = 1. A linear time schedule from 1 → 0 in
`n_sampling_steps = 50` increments. At step *i* with (t_cur, t_next):

1. Predict v_cur = v_θ(x, t_cur, y), with CFG if `cfg_scale ≠ 1`.
2. Euler half-step: x_E = x + Δt · v_cur, where Δt = t_next − t_cur < 0.
3. If t_next > 0 and sampler == "heun": predict v_next at (x_E, t_next),
   take x_next = x + Δt · ½(v_cur + v_next).
4. Optional Langevin churn: if `churn > 0` and t_next > 0,
   x_next ← x_next + √(churn · |Δt|) · 𝒩(0, I).
5. x ← x_next.

Default `churn = 0` makes the sampler fully deterministic given a seed.
Setting `churn > 0` injects fresh noise mid-trajectory — useful when
the deterministic Heun trajectory under-samples the posterior.

Stochasticity across realizations comes from (i) the initial noise x₁,
and (ii) churn noise if enabled. The model itself is deterministic.

## 6. Inference over the full grid

`inference/generate_realizations.py`. The LR raster is tiled into
overlapping LR patches (default stride = 256 HR pixels = 16 LR pixels);
each tile produces N realizations. We:

1. Pre-screen each tile for nodata, normalize per-tile.
2. Process `tiles_per_batch` tiles per forward, with each tile
   contributing `n_realizations_per_forward` realizations (the LR tensor
   is repeated R times along the batch axis); loop until
   `n_realizations` are produced for that tile.
3. Wrap the forward in `torch.autocast(bf16)` on CUDA.
4. Denormalize and stitch into a Gaussian-weighted accumulation buffer
   (σ = patch_size / 4); divide by accumulated weights at the end.
5. Set uncovered pixels to NaN.
6. Write each realization as a single-band GeoTIFF in the LR raster's CRS.

The SLURM inference job can optionally be converted to an `--array=0-N`
job (see `sherlock/inference.sbatch`): each array task processes a
bounding-box slice of the LR grid, parallelizing inference across many
H100s.

## 7. Uncertainty maps

`inference/uncertainty_maps.py`. From the N realization GeoTIFFs we
compute, per pixel:
- Mean (best estimate).
- Standard deviation (uncertainty in meters).
- Coefficient of variation σ / |μ|.
- 10th / 90th percentile, interquartile range.
- Survey-priority mask: top `priority_fraction` (default 10%) of pixels
  by standard deviation, as a binary GeoTIFF.

All outputs are saved as single-band GeoTIFFs in the HR CRS.

## 8. Evaluation

`evaluation/evaluate.py` runs on the held-out test patches:

| Class | Metric | Notes |
|---|---|---|
| Deterministic | PSNR, SSIM, MAE, RMSE | mean realization vs. ground truth |
| Probabilistic | CRPS | continuous ranked probability score over N realizations |
| Probabilistic | coverage(α) | fraction of HR pixels in the predicted α-CI |
| Probabilistic | sharpness | mean width of the 80% prediction interval |
| Probabilistic | calibration | reliability diagram, nominal vs empirical coverage |
| Geomorphological | slope KS | Kolmogorov-Smirnov statistic between predicted and target slope distributions |

The CRPS estimator is the standard pairwise form
$\mathrm{CRPS} = \mathbb{E}|X - y| - \tfrac{1}{2}\mathbb{E}|X - X'|$.
A vectorized estimator is a future-work item.

## 9. What is *not* in this method

The following appeared in earlier iterations or planning documents but
have no role in the current implementation:

- **DDPM/DDIM** noise schedules (β_t, α̅_t, ε-prediction). Replaced by
  rectified flow.
- **Bicubic-upsample-then-concat** LR conditioning. Replaced by raw-LR
  cross-attention at two scales.
- **L_gradient**, **L_x0** auxiliary losses, **residual_mode**. Never
  implemented in the current branch; only `L_FM` and the optional
  `L_spec` exist.
- **`vanilla nn.MultiheadAttention`** with manual softmax. Replaced by
  `F.scaled_dot_product_attention`.
