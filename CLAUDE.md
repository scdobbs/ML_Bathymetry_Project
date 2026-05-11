# GAN Bathymetry Project — Claude Context

## Status

**Work in progress** — undergoing a staged rewrite from the original
2022-era DDPM/DDIM design to a 2026-era flow-matching design with H100
optimization. The current `main` branch is the DDPM baseline; the
`rewrite/flow-matching-2026` branch hosts the migration. Stage 1
(housekeeping) is in this branch; later stages will replace the
diffusion process, U-Net attention, conditioning pathway, training
loop, and inference sampler.

The project name "GAN Bathymetry" is historical — the current model is
a conditional diffusion model, not a GAN.

## Project Goal

Proof-of-concept paper demonstrating **stochastic super-resolution of seafloor bathymetry**:
- Upsample a 32m resolution MBES grid to 2m resolution
- Generate multiple stochastic realizations to quantify spatial uncertainty
- Identify high-uncertainty zones to guide targeted future surveys

## Data

All data is in `data_for_model/`.

| File | Resolution | Description |
|---|---|---|
| `MorroBay_all_2020_interp_32m_converted.tif` | 32m | Full-area low-resolution grid (LR) |
| `*_Topo1m_*_utm.tif` / `*_converted.tif` (×18) | 2m (resampled from 1m) | High-resolution AUV survey patches (HR) |

- **CRS**: All files in UTM projection
- **Format**: Float32 GeoTIFF, elevation in meters (negative = below sea level)
- **SR ratio**: 16× (32m → 2m)
- **Area**: Morro Bay, CA continental shelf/slope

## Architecture (current baseline)

Conditional DDPM (SR3-style):
- Models the full posterior `P(HR | LR)` — principled stochasticity
- No mode collapse (unlike GANs)
- Run N=50–100 samples → pixel-wise std dev = uncertainty map
- U-Net backbone conditioned on bicubic-upsampled LR concatenated to the
  noisy HR at every diffusion step
- Linear β schedule, ε-prediction, DDIM sampler with eta=0 (deterministic)

This baseline will be replaced in later rewrite stages by rectified-flow /
flow-matching training with EDM2 preconditioning, hybrid U-Net with
transformer middle blocks, multi-scale cross-attention onto raw LR,
512×512 tiles, and a stochastic sampler.

## Project Structure

```
GAN_Bathymetry_Project/
├── CLAUDE.md                            # This file
├── data_for_model/                      # Raw GeoTIFF inputs (do not modify)
├── preprocessing/
│   ├── __init__.py
│   ├── extract_paired_patches.py        # Extract paired LR/HR patches from overlapping areas
│   ├── data_utils.py                    # Normalization, augmentation, dataset class
│   └── validate_dataset.py              # Visualize and QC the extracted patches
├── models/
│   ├── __init__.py
│   ├── unet.py                          # BathyUNet (4-level, attention at 16×16 + 8×8)
│   ├── diffusion.py                     # GaussianDiffusion + build_diffusion_from_checkpoint
│   └── losses.py                        # L1 noise loss + 2D log-power spectral loss
├── training/
│   ├── __init__.py
│   ├── config.py                        # All hyperparameters as a dataclass
│   └── train.py                         # Main training loop (EMA, TensorBoard, checkpoints)
├── inference/
│   ├── __init__.py
│   ├── generate_realizations.py         # Tile LR grid, generate N realizations, stitch
│   └── uncertainty_maps.py              # Compute mean/std/CoV/IQR, export GeoTIFF
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                       # PSNR, SSIM, CRPS, calibration
│   └── evaluate.py                      # Run evaluation on held-out HR surveys
├── utils/
│   ├── __init__.py
│   └── device.py                        # get_device() helper (cuda → mps → cpu)
└── sherlock/
    ├── environment.yml                  # Conda environment (PyTorch 2.4.1 + CUDA 12.1)
    ├── preprocess.sbatch                # CPU job: extract patches
    ├── train.sbatch                     # GPU job: train diffusion model
    └── inference.sbatch                 # GPU job: generate realizations + uncertainty maps
```

All entry points are run as Python modules from the project root, e.g.
`python -m training.train ...`. The sbatch files already do this.

## Workflow

### Step 1 — Preprocessing (local or Sherlock CPU)
```bash
# Extract paired patches from all overlap regions
python -m preprocessing.extract_paired_patches \
    --lr_path data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
    --hr_dir data_for_model/ \
    --out_dir /path/to/patches/ \
    --patch_size 64 \
    --stride 32

# Validate output
python -m preprocessing.validate_dataset --patches_dir /path/to/patches/
```

### Step 2 — Training (Sherlock GPU)
```bash
sbatch sherlock/train.sbatch
# Monitor via: tensorboard --logdir $OUT_DIR/logs/
```

### Step 3 — Inference (Sherlock GPU)
```bash
sbatch sherlock/inference.sbatch
# Generates N=50 realizations over the full 32m grid + uncertainty maps
```

### Step 4 — Uncertainty Maps (local, optional standalone)
```bash
python -m inference.uncertainty_maps \
    --realizations_dir /path/to/realizations/ \
    --out_dir results/
```

### Step 5 — Evaluation (uses held-out HR surveys)
```bash
python -m evaluation.evaluate \
    --checkpoint /path/to/checkpoint.pt \
    --test_patches /path/to/patches/test/
```

## Key Design Choices (current baseline)

- **Patch size**: 64×64 HR pixels (128m × 128m footprint at 2m resolution)
- **LR condition**: 4×4 pixel window from 32m grid, bicubic-upsampled to 64×64
- **Normalization**: Per-patch (subtract mean, divide by std dev) — preserves relative topography
- **Train/val/test split**: By survey area, NOT random patches — prevents data leakage
- **DDIM sampling**: 50 steps with eta=0 (deterministic)
- **EMA**: Exponential moving average of model weights (critical for diffusion quality)
- **Spectral loss**: 2D log-power FFT L1 loss to enforce correct texture wavelengths

## Held-Out Test Surveys

The following surveys are held out for final evaluation (never included in training):
- `LuciaChica_2007_Topo1m_UTM_utm.tif`
- `HeadlessCanyon_Topo1m_converted.tif`
- `BankTopEastofCanyon3_Topo1m_converted.tif`

(Selected to cover diverse geomorphological settings: open shelf, canyon head, bank top.)

## Sherlock Notes

- Cluster: Stanford Sherlock, SLURM scheduler
- Partition: `serc`
- Module: `module load py-pytorch/2.4.1_py312` (used by all three sbatch files).
  Rewrite target is PyTorch 2.5+; upgrade as soon as a 2.5+ build is published.
- Scratch storage: `/scratch/users/$USER/` for patches, checkpoints, realizations
- Home storage: `/home/users/$USER/` for code and configs

## Python Dependencies

Core: `pytorch=2.4.1`, `pytorch-cuda=12.1`, `rasterio`, `numpy`, `scipy`,
`matplotlib`, `tqdm`, `tensorboard`, `einops`. Pip: `flash-attn==2.6.3`
(CUDA 12.x only).

Install locally for development:
```bash
conda env create -f sherlock/environment.yml
conda activate bathy_diffusion
```
