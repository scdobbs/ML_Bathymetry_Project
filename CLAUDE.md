# GAN Bathymetry Project — Claude Context

## Status

Six-stage rewrite from a 2022-era DDPM/DDIM design to a 2026-era flow-
matching design is **complete** as of Stage 6. The `main` branch retains
the original DDPM baseline for historical comparison; the
`rewrite/flow-matching-2026` branch hosts the new code and is the active
development line. End-to-end smoke runs on Sherlock are pending.

The project name "GAN Bathymetry" is historical — the current model is a
conditional rectified-flow model, not a GAN.

## Project Goal

Proof-of-concept paper demonstrating **stochastic super-resolution of
seafloor bathymetry**:
- Upsample a 32 m resolution MBES grid to 2 m resolution.
- Generate multiple stochastic realizations to quantify spatial uncertainty.
- Identify high-uncertainty zones to guide targeted future surveys.

## Data

All data is in `data_for_model/`.

| File | Resolution | Description |
|---|---|---|
| `MorroBay_all_2020_interp_32m_converted.tif` | 32 m | Full-area LR grid |
| `*_Topo1m_*_utm.tif` / `*_converted.tif` (×18) | 2 m (resampled from 1 m) | HR AUV survey patches |

- CRS: UTM. Format: float32 GeoTIFF, elevation in meters (negative = below sea level).
- SR ratio: **16×** (configurable; not hardcoded).
- Area: Morro Bay, CA continental shelf / slope.

## Architecture (post-rewrite)

**Conditional rectified flow** on a hybrid CNN/transformer U-Net.

- **Training objective**: rectified flow. Linear interpolant
  x_t = (1 − t) x_0 + t z, target velocity v = z − x_0, primary loss is
  MSE between predicted and target velocity. Logit-normal t sampling
  concentrates supervision in the middle of the time interval.
- **Model**: 6-level U-Net, 85 M parameters. Levels 0–3 are convolutional
  ResBlocks with AdaLN-Zero time modulation. Levels 4–5 are DiT-style
  transformer blocks with self-attention, **cross-attention onto raw LR
  features**, and an MLP — all gated by AdaLN-Zero. Attention uses
  `F.scaled_dot_product_attention` (Flash backend on H100).
- **Conditioning**: a small CNN lifts raw LR (32 × 32) into K/V feature
  maps at two scales (32 × 32, 16 × 16) matching the transformer levels.
  No bicubic-upsample-then-concat path remains.
- **CFG**: training-time conditioning dropout (prob 0.1) with a learned-
  zero null token; inference-time guidance via the standard
  v = v_uncond + s · (v_cond − v_uncond).
- **Sampling**: 2nd-order Heun integration from t = 1 to t = 0 with
  optional Langevin churn. Default 50 steps. Deterministic at churn=0.

Initialization is DiT-style identity: AdaLN-Zero output gates and the
final output Conv2d are zero-initialized; the second conv inside each
ResBlock is zero-initialized. The model returns zero velocity at init
and training starts from a stable identity-residual configuration.

## Project Structure

```
GAN_Bathymetry_Project/
├── CLAUDE.md                            # This file
├── METHOD.md                            # Method document for the paper
├── pyproject.toml                       # Editable install for python -m ...
├── data_for_model/                      # Raw GeoTIFF inputs (do not modify)
├── preprocessing/
│   ├── __init__.py
│   ├── extract_paired_patches.py        # 512×512 paired patches at SR=16×
│   ├── data_utils.py                    # Returns {lr_raw, lr_up, hr, stats}
│   └── validate_dataset.py              # 3-tensor visualizer
├── models/
│   ├── __init__.py
│   ├── unet.py                          # BathyUNet: 6-level hybrid CNN + transformer
│   ├── diffusion.py                     # RectifiedFlow + build_*_from_checkpoint
│   └── losses.py                        # FlowMatchingLoss + RadialSpectralLoss
├── training/
│   ├── __init__.py
│   ├── config.py                        # All hyperparameters as a dataclass
│   └── train.py                         # DDP / bf16 / compile / foreach-EMA / grad-accum
├── inference/
│   ├── __init__.py
│   ├── generate_realizations.py         # Tile + chunk + autocast + CFG/churn
│   └── uncertainty_maps.py              # Mean/std/CoV/IQR/priority masks
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                       # PSNR, SSIM, CRPS, coverage, slope KS
│   └── evaluate.py                      # Run metrics on held-out test patches
├── utils/
│   ├── __init__.py
│   ├── device.py                        # get_device() helper (cuda → mps → cpu)
│   └── distributed.py                   # setup_distributed, all_reduce_mean, ...
└── sherlock/
    ├── environment.yml                  # python=3.12, pytorch=2.4.1, cuda=12.1
    ├── preprocess.sbatch                # CPU job: extract patches
    ├── train.sbatch                     # torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE
    └── inference.sbatch                 # Optional --array=0-N region parallelism
```

Entry points run as `python -m pkg.mod ...` from the project root, e.g.
`python -m training.train ...`. The sbatch files do this through
`torchrun --module training.train`.

## Workflow

### 1 — Preprocessing
```bash
sbatch sherlock/preprocess.sbatch
# or locally:
python -m preprocessing.extract_paired_patches \
    --lr_path data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
    --hr_dir  data_for_model/ \
    --out_dir /scratch/.../bathy_patches/ \
    --patch_size 512 --stride 256 --sr_ratio 16
python -m preprocessing.validate_dataset --patches_dir /scratch/.../bathy_patches/
```

### 2 — Training
```bash
sbatch sherlock/train.sbatch
# Same script runs on 1 GPU (WORLD_SIZE=1) or N GPUs (--gres=gpu:h100:N).
```

### 3 — Inference
```bash
sbatch sherlock/inference.sbatch
# Or use --array=0-N to parallelize across sub-regions.
```

### 4 — Uncertainty maps (already chained from inference.sbatch)
```bash
python -m inference.uncertainty_maps \
    --realizations_dir /scratch/.../realizations \
    --out_dir          results/uncertainty/
```

### 5 — Evaluation
```bash
python -m evaluation.evaluate \
    --checkpoint   /scratch/.../ckpt_final.pt \
    --test_patches /scratch/.../bathy_patches/test/
```

## Key configurable knobs (training/config.py)

- `patch_size = 512`, `sr_ratio = 16`, `lr_patch_size = 32` (= patch / sr_ratio)
- Model: `base_ch=64`, `ch_mult=(1,2,4,6,8,12)`, `n_heads=8`,
  `transformer_levels=(4,5)`, `transformer_depth=2`, `time_dim=256`
- Flow: `time_sampling="logit_normal"`, `n_sampling_steps=50`,
  `sampler="heun"`, `churn=0.0`
- CFG: `cfg_dropout_prob=0.1`, `cfg_scale=1.5`
- Loss: `spectral_loss_weight=0.0` (off by default; on for ablations),
  `spectral_loss_mode="radial"`, `spectral_loss_snr_weighting=True`
- Training: `batch_size=16`, `learning_rate=2e-4`, `n_epochs=500`,
  `grad_clip=1.0`, `ema_decay=0.9999`, `ema_warmup_steps=1000`,
  `precision="bf16"`, `compile_model=True`, `grad_accum_steps=1`
- Inference: `n_realizations_per_forward=10`, `tiles_per_batch=1`
- DataLoader: `num_workers=16`, `pin_memory=True`,
  `persistent_workers=True`, `prefetch_factor=4`
- Distributed: `distributed=True`, `ddp_find_unused_params=False`

## Held-Out Test Surveys

Never included in training:
- `LuciaChica_2007_Topo1m_UTM_utm.tif`
- `HeadlessCanyon_Topo1m_converted.tif`
- `BankTopEastofCanyon3_Topo1m_converted.tif`

## Sherlock Notes

- Partition: `serc`. Modules: `py-pytorch/2.4.1_py312`.
- Scratch: `/scratch/users/$USER/` for patches, checkpoints, realizations.
- Home: `/home/users/$USER/` for code.
- H100 GPU requested via `--gres=gpu:h100:1`. Fall back to `--gres=gpu:1`
  if the h100 selector isn't enabled on serc (documented in the sbatch
  comments).

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
