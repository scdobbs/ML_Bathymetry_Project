"""
Generate N stochastic HR realizations from the trained diffusion model
over the full 32m LR grid (or any subregion of it).

Strategy:
    - Tile the LR grid into overlapping patches
    - For each tile: generate N realizations using DDIM sampling
    - Stitch tiles back together with Gaussian-weighted blending
    - Denormalize back to elevation (meters)

Usage:
    python inference/generate_realizations.py \
        --checkpoint /scratch/users/scdobbs/bathy_diffusion/checkpoints/ckpt_final.pt \
        --lr_tif data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --out_dir /scratch/users/scdobbs/bathy_realizations/ \
        --n_realizations 50 \
        --ddim_steps 50 \
        --patch_size 64 \
        --stride 32

On Sherlock: use sherlock/inference.sbatch which runs this script.
"""

import os
import sys
import argparse
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from preprocessing.data_utils import normalize_patch, denormalize_patch, bicubic_upsample
from training.config import Config


def load_model(checkpoint_path, cfg, device):
    """Load the EMA model from a checkpoint."""
    unet = BathyUNet(
        in_ch=2, out_ch=1,
        base_ch=cfg.base_ch, time_dim=cfg.time_dim, dropout=0.0  # no dropout at inference
    ).to(device)
    diffusion = GaussianDiffusion(unet, n_timesteps=cfg.n_timesteps,
                                  beta_start=cfg.beta_start,
                                  beta_end=cfg.beta_end).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    # Load EMA weights (higher quality than raw model weights)
    ema_state = state.get('ema', state.get('model'))
    diffusion.load_state_dict(ema_state)
    diffusion.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return diffusion


def make_gaussian_weight(patch_size):
    """
    Create a 2D Gaussian weight mask for patch blending.
    Center has weight ~1, edges taper to near 0.
    """
    sigma = patch_size / 4.0
    coords = np.linspace(-(patch_size // 2), patch_size // 2, patch_size)
    x, y = np.meshgrid(coords, coords)
    w = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return w.astype(np.float32)


def read_lr_tif(lr_tif_path):
    """Read the full LR GeoTIFF and return (array, profile)."""
    with rasterio.open(lr_tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def run_inference(diffusion, lr_arr, lr_profile, patch_size, stride,
                  n_realizations, ddim_steps, device, out_dir):
    """
    Tile the LR grid, generate N realizations per tile, and stitch.

    Returns:
        realizations: (n_realizations, H_hr, W_hr) float32 array of HR elevations
        lr_profile:   updated rasterio profile for the HR output
    """
    lr_h, lr_w = lr_arr.shape
    lr_res = lr_profile['transform'].a      # pixel size in map units (meters)
    hr_res = lr_res / 16.0                   # target resolution (2m)
    scale  = 16

    hr_h = lr_h * scale
    hr_w = lr_w * scale
    hr_patch = patch_size                    # HR patch size
    lr_tile  = patch_size // scale           # LR tile size (4×4)

    # Accumulation buffers: (n_realizations, hr_h, hr_w)
    accum   = np.zeros((n_realizations, hr_h, hr_w), dtype=np.float32)
    weights = np.zeros((hr_h, hr_w), dtype=np.float32)
    gauss_w = make_gaussian_weight(hr_patch)

    # LR stride (in LR pixels) corresponding to HR stride
    lr_stride = stride // scale
    if lr_stride < 1:
        lr_stride = 1

    # Generate tile positions (in LR pixel space)
    row_positions = list(range(0, lr_h - lr_tile + 1, lr_stride))
    col_positions = list(range(0, lr_w - lr_tile + 1, lr_stride))
    # Ensure we cover the last tile
    if row_positions[-1] + lr_tile < lr_h:
        row_positions.append(lr_h - lr_tile)
    if col_positions[-1] + lr_tile < lr_w:
        col_positions.append(lr_w - lr_tile)

    total_tiles = len(row_positions) * len(col_positions)
    print(f"Generating {n_realizations} realizations for {total_tiles} tiles...")
    print(f"Output HR grid size: {hr_h} × {hr_w} pixels at {hr_res:.1f}m resolution")

    with torch.no_grad():
        for lr_row in tqdm(row_positions, desc="Rows"):
            for lr_col in col_positions:
                # Extract LR tile
                lr_tile_arr = lr_arr[lr_row : lr_row + lr_tile,
                                     lr_col : lr_col + lr_tile]

                # Handle nodata
                nodata_frac = (~np.isfinite(lr_tile_arr)).mean()
                if nodata_frac > 0.5:
                    continue

                lr_tile_arr = np.where(np.isfinite(lr_tile_arr),
                                       lr_tile_arr, np.nanmean(lr_tile_arr))

                # Normalize using LR tile statistics
                lr_norm, tile_mean, tile_std = normalize_patch(lr_tile_arr)

                # Bicubic upsample to HR patch size
                lr_up = bicubic_upsample(lr_norm, hr_patch)  # (hr_patch, hr_patch)

                lr_tensor = torch.from_numpy(lr_up[np.newaxis, np.newaxis]).float().to(device)
                # (1, 1, hr_patch, hr_patch)

                # Generate N realizations
                samples = diffusion.sample_ddim(lr_tensor, n_realizations=n_realizations,
                                                n_steps=ddim_steps)
                # samples: (n_realizations, 1, hr_patch, hr_patch)
                samples_np = samples.squeeze(1).cpu().numpy()  # (N, hr_patch, hr_patch)

                # Denormalize to elevation
                samples_np = denormalize_patch(samples_np, tile_mean, tile_std)

                # Stitch into accumulation buffer
                hr_row = lr_row * scale
                hr_col = lr_col * scale

                accum[:, hr_row:hr_row+hr_patch, hr_col:hr_col+hr_patch] += \
                    samples_np * gauss_w[np.newaxis]
                weights[hr_row:hr_row+hr_patch, hr_col:hr_col+hr_patch] += gauss_w

    # Normalize by accumulated weights
    w_safe = np.where(weights > 0, weights, 1.0)
    realizations = accum / w_safe[np.newaxis]

    # Set uncovered pixels to NaN
    mask = weights == 0
    realizations[:, mask] = np.nan

    # Build HR rasterio profile
    lr_transform = lr_profile['transform']
    hr_transform = from_origin(
        lr_transform.c, lr_transform.f,   # top-left corner
        hr_res, hr_res                      # pixel size
    )
    hr_profile = lr_profile.copy()
    hr_profile.update({
        'width':  hr_w,
        'height': hr_h,
        'transform': hr_transform,
        'dtype': 'float32',
        'count': 1,
        'compress': 'deflate',
        'nodata': np.nan,
    })

    return realizations, hr_profile


def save_realizations(realizations, hr_profile, out_dir):
    """Save each realization as a separate GeoTIFF."""
    os.makedirs(out_dir, exist_ok=True)
    n = realizations.shape[0]
    for i in tqdm(range(n), desc="Saving GeoTIFFs"):
        out_path = os.path.join(out_dir, f'realization_{i:04d}.tif')
        with rasterio.open(out_path, 'w', **hr_profile) as dst:
            dst.write(realizations[i].astype(np.float32), 1)
    print(f"Saved {n} realizations to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate stochastic SR realizations from trained diffusion model')
    parser.add_argument('--checkpoint',     required=True)
    parser.add_argument('--lr_tif',         required=True)
    parser.add_argument('--out_dir',        required=True)
    parser.add_argument('--n_realizations', type=int, default=50)
    parser.add_argument('--ddim_steps',     type=int, default=50)
    parser.add_argument('--patch_size',     type=int, default=64)
    parser.add_argument('--stride',         type=int, default=32)
    parser.add_argument('--base_ch',        type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    cfg = Config()
    cfg.base_ch = args.base_ch

    diffusion = load_model(args.checkpoint, cfg, device)

    print(f"Reading LR grid: {args.lr_tif}")
    lr_arr, lr_profile = read_lr_tif(args.lr_tif)
    print(f"LR grid size: {lr_arr.shape[0]} × {lr_arr.shape[1]}")

    realizations, hr_profile = run_inference(
        diffusion, lr_arr, lr_profile,
        patch_size=args.patch_size,
        stride=args.stride,
        n_realizations=args.n_realizations,
        ddim_steps=args.ddim_steps,
        device=device,
        out_dir=args.out_dir,
    )

    save_realizations(realizations, hr_profile, args.out_dir)


if __name__ == '__main__':
    main()
