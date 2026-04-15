# GAN Bathymetry Project — Claude Context

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

## Architecture Decision: Conditional Diffusion Model (SR3-style)

The previous MUNIT GAN approach was replaced for the following reasons:
- MUNIT is designed for **unpaired style transfer**, not super-resolution
- Used unaligned (shuffled) training — no geographic correspondence enforced
- Stored data as 8-bit PNG, losing elevation precision
- Failed with blurry outputs, wrong geomorphology, and training instability

The new approach uses a **conditional DDPM** (Denoising Diffusion Probabilistic Model):
- Models the full posterior `P(HR | LR)` — principled stochasticity
- No mode collapse (unlike GANs)
- Run N=50–100 samples → pixel-wise std dev = uncertainty map
- U-Net backbone conditioned on bicubic-upsampled LR at every diffusion step

## Project Structure

```
GAN_Bathymetry_Project/
├── CLAUDE.md                            # This file
├── data_for_model/                      # Raw GeoTIFF inputs (do not modify)
├── DEM_Sampler_multiple_grids.ipynb     # Reference notebook (kept for context)
├── StochasticMSDRP/                     # Old MUNIT GAN code (kept for reference)
├── preprocessing/
│   ├── extract_paired_patches.py        # Extract paired LR/HR patches from overlapping areas
│   ├── data_utils.py                    # Normalization, augmentation, dataset class
│   └── validate_dataset.py             # Visualize and QC the extracted patches
├── models/
│   ├── unet.py                          # Conditional U-Net with time embedding
│   ├── diffusion.py                     # DDPM forward/reverse process + DDIM sampler
│   └── losses.py                        # L1 noise loss + spectral loss
├── training/
│   ├── config.py                        # All hyperparameters as a dataclass
│   └── train.py                         # Main training loop (EMA, TensorBoard, checkpoints)
├── inference/
│   ├── generate_realizations.py         # Tile LR grid, generate N realizations, stitch
│   └── uncertainty_maps.py             # Compute mean/std/CoV/IQR, export GeoTIFF
├── evaluation/
│   ├── metrics.py                       # PSNR, SSIM, CRPS, calibration
│   └── evaluate.py                      # Run evaluation on held-out HR surveys
├── sherlock/
│   ├── environment.yml                  # Conda environment (PyTorch 2.x + CUDA + rasterio)
│   ├── preprocess.sbatch               # CPU job: extract patches
│   ├── train.sbatch                     # GPU job: train diffusion model
│   └── inference.sbatch                # GPU job array: generate realizations
└── notebooks/
    ├── 01_data_exploration.ipynb        # EDA: footprints, elevation distributions
    └── 02_results_analysis.ipynb       # Visualize SR outputs + uncertainty maps
```

## Workflow

### Step 1 — Preprocessing (local or Sherlock CPU)
```bash
# Extract paired patches from all overlap regions
python preprocessing/extract_paired_patches.py \
    --lr_path data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
    --hr_dir data_for_model/ \
    --out_dir /path/to/patches/ \
    --patch_size 64 \
    --stride 32

# Validate output
python preprocessing/validate_dataset.py --patches_dir /path/to/patches/
```

### Step 2 — Training (Sherlock GPU)
```bash
sbatch sherlock/train.sbatch
# Monitors via: tensorboard --logdir $OUT_DIR/logs/
```

### Step 3 — Inference (Sherlock GPU)
```bash
sbatch sherlock/inference.sbatch
# Generates N=50 realizations over the full 32m grid
```

### Step 4 — Uncertainty Maps (local)
```bash
python inference/uncertainty_maps.py \
    --realizations_dir /path/to/realizations/ \
    --lr_path data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
    --out_dir results/
```

### Step 5 — Evaluation (uses held-out HR surveys)
```bash
python evaluation/evaluate.py \
    --model_checkpoint /path/to/checkpoint.pt \
    --test_patches /path/to/patches/test/
```

## Key Design Choices

- **Patch size**: 64×64 HR pixels (128m × 128m footprint at 2m resolution)
- **LR condition**: 4×4 pixel window from 32m grid, bicubic-upsampled to 64×64
- **Normalization**: Per-patch (subtract mean, divide by std dev) — preserves relative topography
- **Train/val/test split**: By survey area, NOT random patches — prevents data leakage
- **DDIM sampling**: 50 steps (vs 1000 for DDPM) for fast inference
- **EMA**: Exponential moving average of model weights (critical for diffusion quality)
- **Spectral loss**: Auxiliary 2D power spectrum loss to enforce correct texture wavelengths

## Held-Out Test Surveys

The following surveys should be held out for final evaluation (do not include in training):
- `LuciaChica_2007_Topo1m_UTM_utm.tif`
- `HeadlessCanyon_Topo1m_converted.tif`
- `BankTopEastofCanyon3_Topo1m_converted.tif`

(Selected to cover diverse geomorphological settings: open shelf, canyon head, bank top)

## Sherlock Notes

- Cluster: Stanford Sherlock, SLURM scheduler
- Partition: `serc` (as used in original `StochasticMSDRP/jobs/`)
- Modules: load via `module load` or use conda environment in `sherlock/environment.yml`
- Scratch storage: `/scratch/users/$USER/` for large outputs
- Home storage: `/home/users/$USER/` for code and configs

## Python Dependencies

Core: `torch>=2.0`, `rasterio`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `tensorboard`

Install locally for development:
```bash
conda env create -f sherlock/environment.yml
conda activate bathy_diffusion
```
