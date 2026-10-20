> **Superseded 2026-10; see README.md.** This file is the agent context for the original project: a rectified-flow
> super-resolution model with a heteroscedastic head (`het_production_v1`) and its sigma product, trained on a synthetic-LR
> regime. The GRL revision of 2026GL125574 does not rest on any of it; the revised analysis lives under
> `grl_reviews-and-prompt-reports/directives_and_reports/` (R01-R19, phase4/, phase5/). Kept for the record (directive R20,
> 2026-10-20).

# GAN Bathymetry Project — Claude Context

## Status

Six-stage architectural rewrite (DDPM → rectified-flow) is complete on the
`rewrite/flow-matching-2026` branch (the active development line).
Active work is **empirical iteration on training recipe and loss design**.
The model architecture has stabilized; the open problem is producing
realizations that have *structured* high-frequency content (not just the
right HF magnitude).

The project name "GAN Bathymetry" is historical — the current model is a
conditional rectified-flow model, not a GAN.

## Project Goal

Proof-of-concept paper demonstrating **stochastic super-resolution of
seafloor bathymetry**:
- Upsample a 32 m resolution MBES grid to 2 m resolution (16× SR).
- Generate multiple stochastic realizations to quantify spatial uncertainty.
- Identify high-uncertainty zones to guide targeted future surveys.

## Data

Raw GeoTIFFs in `data_for_model/`. Three LR coverage areas, ~24 HR surveys total.

| Region | LR file | HR surveys |
|---|---|---|
| Morro Bay, CA shelf/slope/canyon | `MorroBay_all_2020_interp_32m_converted.tif` | ~18 |
| NorCal (Drakes Bay, Bodega, etc.) | a single LR raster covering the region | ~4 |
| Santa Barbara channel | regional LR raster | ~2 |

- CRS: UTM. Format: float32 GeoTIFF, elevation in meters (negative = depth).
- SR ratio: **16×** (32 m → 2 m), configurable via `cfg.sr_ratio`.
- Per-patch z-score normalization using HR statistics; both HR and LR are
  normalized by the HR patch's mean/std so they share a common scale.
- A `data_for_model/`-equivalent for the worldwide MBES dataset just needs to
  provide co-registered HR/LR rasters in a similar layout; the preprocessor
  finds spatial overlaps automatically.

### Multi-LR handling

`extract_paired_patches.py` accepts multiple LR rasters and picks the first
one that spatially overlaps each HR survey (first-run-wins). Add new regions
by extending the LR raster list in the preprocessing CLI — no schema change.

## Architecture

**Conditional rectified flow** on a hybrid CNN/transformer U-Net.

- **Training objective**: rectified flow. Linear interpolant
  `x_t = (1 − t) x_0 + t z`, target velocity `v = z − x_0`. Primary loss
  is configurable: `fm_loss_type ∈ {"mse", "huber"}` (see Loss Design).
  Logit-normal `t` sampling concentrates supervision in the middle of the
  time interval (SDXL/SD3 recipe).
- **Model**: 6-level U-Net, ~24 M params at `base_ch=32` (or 85 M at 64).
  Levels 0–3 are convolutional ResBlocks with AdaLN-Zero time modulation.
  Levels 4–5 are DiT-style transformer blocks with self-attention,
  **cross-attention onto raw LR features**, and an MLP — all gated by
  AdaLN-Zero. Attention uses `F.scaled_dot_product_attention` (Flash on H100).
- **Conditioning**: a small CNN lifts raw LR (32 × 32) into K/V feature
  maps at two scales (32, 16) matching the transformer levels.
- **Optional residual prediction mode** (`cfg.residual_mode=True`,
  recommended): the model takes `bicubic(LR)` as a 2nd input channel and
  predicts the **HR residual** on top of it. At sampling the bicubic is
  added back: `final_HR = integrated_residual + bicubic(LR)`. This is the
  best fix for the "model ignores LR" problem — see Empirical Findings.
- **CFG**: training-time conditioning dropout (prob 0.1) replaces the
  cross-attention LR with a learned-zero null token. In residual mode the
  bicubic input is **always present** (it's architectural input, not
  conditioning). Inference-time guidance: `v = v_uncond + s · (v_cond − v_uncond)`.
- **Sampling**: 2nd-order Heun integration from `t=1` to `t=0` with
  optional Langevin churn. Default 50 steps. Deterministic at `churn=0`.

Initialization is DiT-style identity: AdaLN-Zero output gates and the
final output Conv2d are zero-initialized; the second conv inside each
ResBlock is zero-initialized. The model returns zero velocity at init.

## Loss Design

`models/losses.py:FlowMatchingLoss` composes up to three terms:

```
total = fm_loss(v_pred, v_target)                                  # always
      + spectral_loss_weight  * spectral_loss(x0_pred, x0_true)    # optional
      + consistency_loss_weight * consistency_loss(x0_pred, lr_raw) # optional
```

**fm_loss** — primary flow-matching MSE *or* Huber (smooth_l1, β=1).
Use Huber when a small tail of outlier patches (extreme `HR − bicubic`
residuals) destabilizes training. See "Training stability" below.

**spectral_loss** — radial-binned 2D FFT log-power loss on the predicted
clean signal `x0_pred`. SNR-weighted by `(1-t)²` so it only matters where
`x0_pred` is reliable. Off by default; **on for runs that need to force
structured HF content**. Currently untested in this regime.

**consistency_loss** — `Huber(avg_pool(x0_pred, sr_ratio), lr_raw)`:
forces the predicted HR, coarsened back to LR, to match the LR
conditioning. Uses smooth-L1 (not MSE) to bound outlier-sample influence,
gated to `t < consistency_t_max` (default 0.5) where `x0_pred` is a
reliable HR reconstruction, and masked to skip samples whose `lr_raw` was
replaced by the null token during CFG dropout.

## Training stability (gotchas learned the hard way)

1. **Outlier patches exist in real bathymetry data.** ~4% of patches have
   `lr_raw |max| > 5` (in normalized units) — these are patches where the
   HR sits on a feature with smaller variance than the surrounding 1km of
   LR (e.g., a flat shelf inside a region with canyon edges). These cause
   pathologically large `HR − bicubic` residuals.

2. **Pure MSE velocity loss diverges on these.** Even with
   `grad_clip=1.0`, residual-mode training with `fm_loss_type=mse`
   collapsed catastrophically around epoch 19–20: val_loss exploded from
   ~0.07 to ~120 and never recovered. Switching to `fm_loss_type=huber`
   (smooth_l1, β=1) extended stable training to epoch ~94 before
   eventually still hitting NaN. **Always use Huber in residual mode**.

3. **Per-sample MSE on small grids is outlier-sensitive.** The consistency
   loss averages over only `sr_ratio² = 256` LR cells per sample, so one
   bad sample dominates a batch under pure MSE. Always use Huber for the
   consistency term too. `fm` averages over ~4 M pixels per batch and
   hides these per-sample failures — train fm can look healthy while one
   patch silently corrupts the optimizer state. **Watch val_loss, not
   train fm**, as the diagnostic for catastrophic failures.

4. **`normalize_patch` clamps `std=1.0` when raw HR is flat.** For such
   patches the saved `stats[1]` is 1.0 (not the original ~0), and the LR
   ends up in unnormalized meters. A degenerate-patch filter using
   `stats[1] < threshold` will never fire. Filter via `hr.std()` per
   sample at the loss level, or check `lr_raw.abs().max() > threshold`.

## Current best configuration

Best results to date (still imperfect — see Empirical Findings):

```bash
python -m training.train \
    --patches_dir /scratch/.../patches_512_s128 \
    --out_dir     /scratch/.../training_run \
    --batch_size 16 --n_epochs 125 \
    --base_ch 32 \
    --precision bf16 --compile \
    --residual_mode \
    --fm_loss_type huber \
    --consistency_loss_weight 1.0
```

Knobs that depart from `Config()` defaults:
- `base_ch=32` (24 M params) instead of 64 (85 M). Better capacity-vs-data
  match for ~40 K independent patches.
- `residual_mode=True`. The model predicts HR residual on top of bicubic LR.
- `fm_loss_type="huber"`. Required for stability in residual mode.
- `consistency_loss_weight=1.0`. Reinforces the LR-respect property.
- `stride=128` (not 256) at preprocessing for ~4× denser patches.

## Empirical findings (what's been tried)

| Run | base_ch | mode | fm loss | cons | val_loss best | Realization quality |
|---|---|---|---|---|---|---|
| baseline (`training_h100`) | 64 | direct HR | mse | 0 | ~0.09 | Jigsaw artifacts |
| `consistency_s128_v3` | 32 | direct HR | mse | 1.0 (Huber) | ~0.09 | Jigsaw; ignores LR mean |
| `residual_s128` | 32 | residual | mse | 0 | **0.04** | LR-respecting, structureless HF; **diverged epoch 20** |
| `residual_huber_s128` | 32 | residual | huber | 1.0 (Huber) | ~0.078 | (TBD: realizations pending); diverged epoch 95 |

Key qualitative observations:
- **Direct-HR runs ignore the LR baseline.** Realizations put high
  topography in low-elevation areas. Cross-attention alone is too indirect
  to enforce the LR constraint on absolute elevation.
- **Residual mode fixes LR fidelity completely.** The bicubic baseline is
  added back at sampling time, so every realization respects the LR.
- **Residual mode produces structureless HF.** Realizations have the right
  amplitude of high-frequency content but blobby/patchy character — the
  model learned the *spectrum* of the HF residual but not the *spatial
  structure*. Real HR has linear canyon tributaries, smooth gradients,
  organized features; realizations replace these with generic random texture.
- **Inter-realization variance is low in residual mode.** The model
  partially "gives up" — outputting roughly the same HF noise pattern
  regardless of the noise seed. This is the well-known failure mode of
  flow-matching models when the conditioning over-constrains.

## Open problem & next experiments

Producing realizations with **structured** HF content (linear features,
organized texture) instead of generic noise. Candidate fixes in order:

1. **Turn on the spectral loss** (`spectral_loss_weight > 0`). The radial
   power-spectrum loss is the term designed to force the HF to have the
   right shape, not just the right amplitude. Untested in residual mode.
2. **Filter outlier patches** at the dataset level to fully eliminate the
   NaN-divergence risk, then train past 100 epochs.
3. **Reduce CFG dropout or increase CFG scale at inference** to push the
   model toward higher-variance realizations.

## Project Structure

```
GAN_Bathymetry_Project/
├── CLAUDE.md                            # This file
├── METHOD.md                            # Method document for the paper
├── pyproject.toml                       # Editable install for python -m ...
├── data_for_model/                      # Raw GeoTIFF inputs (do not modify)
├── preprocessing/
│   ├── extract_paired_patches.py        # Multi-LR support; first-overlap-wins
│   ├── data_utils.py                    # Returns {lr_raw, lr_up, hr, stats}
│   └── validate_dataset.py              # 3-tensor visualizer
├── models/
│   ├── unet.py                          # BathyUNet: 6-level hybrid CNN + transformer
│   ├── diffusion.py                     # RectifiedFlow (+ residual-mode logic)
│   └── losses.py                        # FlowMatchingLoss + RadialSpectralLoss
├── training/
│   ├── config.py                        # All hyperparameters as a dataclass
│   └── train.py                         # DDP / bf16 / compile / foreach-EMA / grad-accum
├── inference/
│   ├── generate_realizations.py         # Tile + chunk + autocast + CFG/churn
│   └── uncertainty_maps.py              # Mean/std/CoV/IQR/priority masks
├── evaluation/
│   ├── metrics.py                       # PSNR, SSIM, CRPS, coverage, slope KS
│   └── evaluate.py                      # Run metrics on held-out test patches
├── utils/
│   ├── device.py                        # get_device() (cuda → mps → cpu)
│   └── distributed.py                   # setup_distributed, all_reduce_mean, ...
└── sherlock/
    ├── environment.yml                  # python=3.12, pytorch=2.4.1, cuda=12.1
    ├── preprocess.sbatch                # CPU job, single-region
    ├── preprocess_s128.sbatch           # CPU job, multi-region, stride=128
    ├── train.sbatch                     # baseline (base_ch=64, MSE, no residual)
    ├── train_small.sbatch               # base_ch=32, MSE, no residual
    ├── train_consistency.sbatch         # base_ch=32, +consistency_loss=1.0
    ├── train_residual.sbatch            # base_ch=32, +residual_mode, MSE fm
    ├── train_residual_huber.sbatch      # CURRENT BEST: residual + Huber + cons=1
    ├── realization_comparison.sbatch    # 8 realizations / survey, std + deviations
    └── inference.sbatch                 # full-region tiled inference
```

Entry points run as `python -m pkg.mod ...` from the project root. The
sbatch files do this through `python3 -m torch.distributed.run --module ...`.

## Workflow

### 1 — Preprocessing
```bash
# Single-region (legacy)
python -m preprocessing.extract_paired_patches \
    --lr_path data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
    --hr_dir  data_for_model/ \
    --out_dir /scratch/.../bathy_patches/ \
    --patch_size 512 --stride 128 --sr_ratio 16

# Multi-region: pass multiple --lr_path flags (first-overlap-wins)
sbatch sherlock/preprocess_s128.sbatch
```

### 2 — Training
```bash
sbatch sherlock/train_residual_huber.sbatch   # current best recipe
```

### 3 — Inference + uncertainty
```bash
sbatch sherlock/inference.sbatch              # → realizations/
python -m inference.uncertainty_maps \
    --realizations_dir /scratch/.../realizations \
    --out_dir          results/uncertainty/
```

### 4 — Realization comparison (qualitative diagnostic)
```bash
sbatch --export=ALL,\
CKPT=/scratch/.../ckpt_step00250000.pt,\
BASE_CH=32,\
RESIDUAL=1,\
PATCHES_DIR=/scratch/.../patches_512_s128/test \
    sherlock/realization_comparison.sbatch
```
Generates 8 stochastic realizations of one median-complexity patch per
held-out survey, plus per-realization-vs-mean deviation maps.

### 5 — Evaluation
```bash
python -m evaluation.evaluate \
    --checkpoint   /scratch/.../ckpt_step00250000.pt \
    --test_patches /scratch/.../patches_512_s128/test/
```

## Key configurable knobs (training/config.py)

Data: `patch_size=512`, `sr_ratio=16`, `lr_patch_size=32`, `stride=128`
(at preprocessing time).

Model: `base_ch=32` (or 64), `ch_mult=(1,2,4,6,8,12)`, `n_heads=8`,
`transformer_levels=(4,5)`, `transformer_depth=2`, `time_dim=256`.

Flow: `time_sampling="logit_normal"`, `n_sampling_steps=50`,
`sampler="heun"`, `churn=0.0`.

CFG: `cfg_dropout_prob=0.1`, `cfg_scale=1.5`.

Loss:
- `fm_loss_type="mse"` (use `"huber"` with `residual_mode=True`)
- `fm_loss_huber_beta=1.0`
- `spectral_loss_weight=0.0` (turn on to force HF structure)
- `spectral_loss_mode="radial"`, `spectral_loss_snr_weighting=True`
- `consistency_loss_weight=0.0` (1.0 recommended)
- `consistency_loss_t_max=0.5`, `consistency_loss_snr_weighting=True`

Residual mode: `residual_mode=False` (set `True` for the best recipe).

Training: `batch_size=16`, `learning_rate=2e-4`, `n_epochs=125`,
`grad_clip=1.0`, `ema_decay=0.9999`, `ema_warmup_steps=1000`,
`precision="bf16"`, `compile_model=True`, `grad_accum_steps=1`.

DataLoader: `num_workers=16`, `pin_memory=True`,
`persistent_workers=True`, `prefetch_factor=4`.

Distributed: `distributed=True`, `ddp_find_unused_params=False`.

## Held-Out Test Surveys

Never included in train/val splits — used only for qualitative realization
comparison and quantitative evaluation:
- `LuciaChica_2007_Topo1m_UTM_utm.tif` — Morro Bay slope
- `HeadlessCanyon_Topo1m_converted.tif` — Morro Bay canyon
- `BankTopEastofCanyon3_Topo1m_converted.tif` — Morro Bay bank
- `Bathymetry_DrakesBay.tif` — NorCal shelf
- `Bathymetry_OffshoreRefugioBeach.tif` — Santa Barbara channel

## Sherlock Notes

- Partition: `serc`. Module: `py-pytorch/2.4.1_py312`.
- Scratch: `/scratch/users/$USER/` for patches, checkpoints, realizations.
- Group scratch: `/scratch/groups/hilley/bathy_diffusion_2026/` for shared patches.
- Home: `/home/users/$USER/` for code.
- H100 GPU: `--gres=gpu:1 --constraint=GPU_SKU:H100_SXM5`. Training takes
  ~13 h per 125 epochs at base_ch=32.

## Python Dependencies

Core: `pytorch=2.4.1`, `pytorch-cuda=12.1`, `rasterio>=1.3`, `numpy`,
`scipy`, `matplotlib`, `tqdm`, `tensorboard`, `einops`. Pip-only:
`flash-attn==2.6.3` (CUDA 12.x), plus the package itself via
`pip install -e .`. Triton ships with the PyTorch wheel.

Install locally:
```bash
conda env create -f sherlock/environment.yml
conda activate bathy_diffusion
```

## Notes for adapting to a new dataset (e.g. worldwide MBES)

If you're starting a new repo with the same architecture but different
underlying data (e.g. global MBES coverage instead of regional AUV):

1. **Sanity-check the residual distribution first.** Run a quick scan over
   1000 random patches and report `(hr - bicubic(lr_raw)).abs().max()`
   percentiles. If p99 is > 5 in normalized units, you have outlier
   patches — start with `fm_loss_type=huber` from the first run.
2. **Per-patch z-score using HR stats is the assumed normalization scheme.**
   Both `lr_raw` and `hr` are saved already-normalized in the .npz patches.
   If your dataset has very different elevation ranges per region, this is
   the right choice (each patch self-normalizes), but be aware of the
   flat-patch `std=1.0` clamp gotcha.
3. **The 6-level U-Net + `(1,2,4,6,8,12)` mult assumes patch_size=512 and
   sr_ratio=16.** Change `lr_patch_size` if `sr_ratio` changes;
   `transformer_levels=(4,5)` is hardwired in the LR encoder
   (`LRConditioningEncoder` produces K/V at exactly 2 scales).
4. **Start with the current best config** (residual mode + Huber + cons=1.0)
   unless you have a reason not to. Diverging from this just to "try MSE"
   will cost you a 13-h training run.
5. **Watch val_loss, not train fm**, as the divergence canary. Train fm
   can look reasonable while one patch silently destroys the optimizer.
