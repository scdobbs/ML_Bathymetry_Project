"""
Generate N stochastic HR realizations from the trained rectified-flow
model over the full LR grid (or any subregion of it).

Strategy:
    - Tile the LR grid into overlapping patches
    - For each tile: generate N realizations by Heun-integrating the flow
    - Stitch tiles back together with Gaussian-weighted blending
    - Denormalize back to elevation (meters)

The model now consumes raw LR tiles directly via cross-attention; the
bicubic upsample step from the DDPM era is gone.

Usage:
    python -m inference.generate_realizations \
        --checkpoint /path/to/ckpt_final.pt \
        --lr_tif     data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --out_dir    /scratch/.../bathy_realizations/ \
        --n_realizations 50 \
        --n_steps 50

On Sherlock: use sherlock/inference.sbatch which runs this script.
"""

import os
import argparse
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

from models.diffusion import build_rectified_flow_from_checkpoint
from preprocessing.data_utils import normalize_patch, denormalize_patch
from training.config import Config
from utils.device import get_device


def load_model(checkpoint_path, cfg, device):
    """Load the EMA model from a checkpoint."""
    flow = build_rectified_flow_from_checkpoint(checkpoint_path, cfg, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return flow


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


def run_inference(flow, lr_arr, lr_profile, patch_size, stride, sr_ratio,
                  n_realizations, n_steps, cfg_scale, device, out_dir):
    """
    Tile the LR grid, generate N realizations per tile, and stitch.

    Returns:
        realizations: (n_realizations, H_hr, W_hr) float32 array of HR elevations
        hr_profile:   rasterio profile for the HR output
    """
    lr_h, lr_w = lr_arr.shape
    lr_res = lr_profile['transform'].a       # pixel size in map units (meters)
    hr_res = lr_res / sr_ratio               # target resolution
    scale  = sr_ratio

    hr_h = lr_h * scale
    hr_w = lr_w * scale
    hr_patch = patch_size                    # HR patch size
    lr_tile  = patch_size // scale           # LR tile size

    accum   = np.zeros((n_realizations, hr_h, hr_w), dtype=np.float32)
    weights = np.zeros((hr_h, hr_w), dtype=np.float32)
    gauss_w = make_gaussian_weight(hr_patch)

    lr_stride = max(1, stride // scale)

    row_positions = list(range(0, lr_h - lr_tile + 1, lr_stride))
    col_positions = list(range(0, lr_w - lr_tile + 1, lr_stride))
    if row_positions[-1] + lr_tile < lr_h:
        row_positions.append(lr_h - lr_tile)
    if col_positions[-1] + lr_tile < lr_w:
        col_positions.append(lr_w - lr_tile)

    total_tiles = len(row_positions) * len(col_positions)
    print(f"Generating {n_realizations} realizations for {total_tiles} tiles "
          f"(n_steps={n_steps}, cfg_scale={cfg_scale})...")
    print(f"Output HR grid size: {hr_h} × {hr_w} pixels at {hr_res:.1f}m resolution")

    with torch.no_grad():
        for lr_row in tqdm(row_positions, desc="Rows"):
            for lr_col in col_positions:
                lr_tile_arr = lr_arr[lr_row : lr_row + lr_tile,
                                     lr_col : lr_col + lr_tile]

                nodata_frac = (~np.isfinite(lr_tile_arr)).mean()
                if nodata_frac > 0.5:
                    continue

                lr_tile_arr = np.where(np.isfinite(lr_tile_arr),
                                       lr_tile_arr, np.nanmean(lr_tile_arr))

                # Normalize using tile statistics (per-patch normalization)
                lr_norm, tile_mean, tile_std = normalize_patch(lr_tile_arr)

                # Pass raw normalized LR directly — the model handles
                # multi-scale conditioning via cross-attention.
                lr_tensor = torch.from_numpy(
                    lr_norm[np.newaxis, np.newaxis]
                ).float().to(device)  # (1, 1, lr_tile, lr_tile)

                samples = flow.sample(
                    lr_tensor,
                    n_realizations=n_realizations,
                    n_steps=n_steps,
                    cfg_scale=cfg_scale,
                )
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
    parser.add_argument('--n_steps',        type=int, default=None,
                        help='Heun integration steps (defaults to Config.n_sampling_steps)')
    parser.add_argument('--cfg_scale',      type=float, default=None,
                        help='Classifier-free guidance scale (defaults to Config.cfg_scale)')
    parser.add_argument('--patch_size',     type=int, default=None,
                        help='HR patch size (defaults to Config.patch_size)')
    parser.add_argument('--stride',         type=int, default=None,
                        help='Patch stride in HR pixels (defaults to Config.stride)')
    parser.add_argument('--sr_ratio',       type=int, default=None,
                        help='Super-resolution ratio (defaults to Config.sr_ratio)')
    parser.add_argument('--base_ch',        type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    cfg = Config()
    if args.base_ch is not None: cfg.base_ch = args.base_ch
    patch_size = args.patch_size if args.patch_size is not None else cfg.patch_size
    stride     = args.stride     if args.stride     is not None else cfg.stride
    sr_ratio   = args.sr_ratio   if args.sr_ratio   is not None else cfg.sr_ratio
    n_steps    = args.n_steps    if args.n_steps    is not None else cfg.n_sampling_steps
    cfg_scale  = args.cfg_scale  if args.cfg_scale  is not None else cfg.cfg_scale

    flow = load_model(args.checkpoint, cfg, device)

    print(f"Reading LR grid: {args.lr_tif}")
    lr_arr, lr_profile = read_lr_tif(args.lr_tif)
    print(f"LR grid size: {lr_arr.shape[0]} × {lr_arr.shape[1]}")

    realizations, hr_profile = run_inference(
        flow, lr_arr, lr_profile,
        patch_size=patch_size,
        stride=stride,
        sr_ratio=sr_ratio,
        n_realizations=args.n_realizations,
        n_steps=n_steps,
        cfg_scale=cfg_scale,
        device=device,
        out_dir=args.out_dir,
    )

    save_realizations(realizations, hr_profile, args.out_dir)


if __name__ == '__main__':
    main()
