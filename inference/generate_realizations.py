"""
Generate N stochastic HR realizations from the trained diffusion model
over specified regions of the 32m LR grid.

Strategy:
    - Process one region at a time (test surveys, bbox, or auto-tiled full grid)
    - For each region: tile into overlapping patches, generate N realizations
    - Stitch tiles with Gaussian-weighted blending, denormalize to elevation
    - Save each realization as a GeoTIFF

Usage — test survey regions (recommended for evaluation):
    python3 inference/generate_realizations.py \
        --checkpoint /path/to/ckpt_final.pt \
        --lr_tif data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --out_dir /scratch/users/$USER/bathy_diffusion/realizations \
        --test_surveys \
        --n_realizations 50

Usage — custom bounding box (UTM coordinates):
    python3 inference/generate_realizations.py \
        --checkpoint /path/to/ckpt_final.pt \
        --lr_tif data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --out_dir /scratch/users/$USER/bathy_diffusion/realizations \
        --bbox 633000 3873000 641000 3877000 \
        --region_name HeadlessCanyon \
        --n_realizations 50

Usage — full grid (auto-tiled):
    python3 inference/generate_realizations.py \
        --checkpoint /path/to/ckpt_final.pt \
        --lr_tif data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --out_dir /scratch/users/$USER/bathy_diffusion/realizations \
        --full_grid \
        --n_realizations 50
"""

import os
import sys
import argparse
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from preprocessing.data_utils import normalize_patch, denormalize_patch, bicubic_upsample
from training.config import Config


# Test survey bounding boxes (UTM Zone 10N) with ~10 LR pixel padding
TEST_SURVEY_REGIONS = {
    'LuciaChica_2007': (605500, 3953800, 611900, 3959900),
    'HeadlessCanyon':  (632700, 3873100, 640500, 3876500),
    'BankTopEastofCanyon3': (639100, 3873800, 648400, 3877300),
}

VAL_SURVEY_REGIONS = {
    '600mGully':          (631341, 3924798, 643220, 3933526),
    'BankFlankHoles':     (640004, 3904796, 645172, 3908140),
    'PockmarkNorth':      (621676, 3929586, 625875, 3933262),
    'LuciaChica2019_970': (608306, 3953267, 612700, 3957122),
    'Transit':            (624075, 3913994, 650503, 3937197),
}


def load_model(checkpoint_path, cfg, device):
    """Load the EMA model from a checkpoint."""
    unet = BathyUNet(
        in_ch=2, out_ch=1,
        base_ch=cfg.base_ch, time_dim=cfg.time_dim, dropout=0.0
    ).to(device)
    diffusion = GaussianDiffusion(unet, n_timesteps=cfg.n_timesteps,
                                  beta_start=cfg.beta_start,
                                  beta_end=cfg.beta_end).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ema_state = state.get('ema', state.get('model'))
    diffusion.load_state_dict(ema_state)
    diffusion.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return diffusion


def make_gaussian_weight(patch_size):
    """2D Gaussian weight for overlapping patch blending."""
    sigma = patch_size / 4.0
    coords = np.linspace(-(patch_size // 2), patch_size // 2, patch_size)
    x, y = np.meshgrid(coords, coords)
    w = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return w.astype(np.float32)


def read_lr_tif(lr_tif_path):
    """Read the full LR GeoTIFF."""
    with rasterio.open(lr_tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def utm_bbox_to_lr_pixels(bbox, lr_profile):
    """Convert UTM (xmin, ymin, xmax, ymax) to LR pixel (row_start, col_start, row_end, col_end)."""
    xmin, ymin, xmax, ymax = bbox
    t = lr_profile['transform']
    # Transform: col = (x - t.c) / t.a, row = (y - t.f) / t.e
    col_start = max(0, int((xmin - t.c) / t.a))
    col_end   = min(lr_profile['width'],  int(np.ceil((xmax - t.c) / t.a)))
    row_start = max(0, int((ymax - t.f) / t.e))  # t.e is negative
    row_end   = min(lr_profile['height'], int(np.ceil((ymin - t.f) / t.e)))
    return row_start, col_start, row_end, col_end


def process_region(diffusion, lr_arr, lr_profile, region_name, bbox,
                   patch_size, stride, n_realizations, ddim_steps, device, out_dir,
                   residual=False):
    """
    Generate N realizations for a single region defined by a UTM bbox.
    Saves each realization as a separate GeoTIFF in out_dir/region_name/.
    """
    scale = 16
    lr_tile = patch_size // scale

    # Convert bbox to LR pixel coords
    r0, c0, r1, c1 = utm_bbox_to_lr_pixels(bbox, lr_profile)
    lr_h = r1 - r0
    lr_w = c1 - c0

    if lr_h < lr_tile or lr_w < lr_tile:
        print(f"  [skip] Region {region_name} too small: {lr_h}×{lr_w} LR pixels")
        return

    hr_h = lr_h * scale
    hr_w = lr_w * scale

    # Memory estimate
    mem_gb = n_realizations * hr_h * hr_w * 4 / 1e9
    print(f"\nRegion '{region_name}': LR {lr_h}×{lr_w} → HR {hr_h}×{hr_w}")
    print(f"  Memory for {n_realizations} realizations: {mem_gb:.1f} GB")

    if mem_gb > 50:
        print(f"  WARNING: region too large ({mem_gb:.0f} GB). Consider using a smaller bbox.")
        return

    lr_region = lr_arr[r0:r1, c0:c1]

    # Accumulation buffers
    accum   = np.zeros((n_realizations, hr_h, hr_w), dtype=np.float32)
    weights = np.zeros((hr_h, hr_w), dtype=np.float32)
    gauss_w = make_gaussian_weight(patch_size)

    # LR stride
    lr_stride = stride // scale
    if lr_stride < 1:
        lr_stride = 1

    # Tile positions (in local LR pixel space)
    row_positions = list(range(0, lr_h - lr_tile + 1, lr_stride))
    col_positions = list(range(0, lr_w - lr_tile + 1, lr_stride))
    if row_positions[-1] + lr_tile < lr_h:
        row_positions.append(lr_h - lr_tile)
    if col_positions[-1] + lr_tile < lr_w:
        col_positions.append(lr_w - lr_tile)

    total_tiles = len(row_positions) * len(col_positions)
    print(f"  Tiles: {total_tiles} ({len(row_positions)} rows × {len(col_positions)} cols)")

    with torch.no_grad():
        for lr_row in tqdm(row_positions, desc=f"  {region_name}"):
            for lr_col in col_positions:
                lr_tile_arr = lr_region[lr_row:lr_row+lr_tile, lr_col:lr_col+lr_tile]

                if (~np.isfinite(lr_tile_arr)).mean() > 0.5:
                    continue

                lr_tile_arr = np.where(np.isfinite(lr_tile_arr),
                                       lr_tile_arr, np.nanmean(lr_tile_arr))

                lr_norm, tile_mean, tile_std = normalize_patch(lr_tile_arr)
                lr_up = bicubic_upsample(lr_norm, patch_size)
                lr_tensor = torch.from_numpy(lr_up[np.newaxis, np.newaxis]).float().to(device)

                samples = diffusion.sample_ddim(lr_tensor, n_realizations=n_realizations,
                                                n_steps=ddim_steps, add_lr=residual)
                samples_np = samples.squeeze(1).cpu().numpy()
                samples_np = denormalize_patch(samples_np, tile_mean, tile_std)

                hr_row = lr_row * scale
                hr_col = lr_col * scale
                accum[:, hr_row:hr_row+patch_size, hr_col:hr_col+patch_size] += \
                    samples_np * gauss_w[np.newaxis]
                weights[hr_row:hr_row+patch_size, hr_col:hr_col+patch_size] += gauss_w

    # Normalize by weights
    w_safe = np.where(weights > 0, weights, 1.0)
    realizations = accum / w_safe[np.newaxis]
    realizations[:, weights == 0] = np.nan

    # Build HR profile for this region
    lr_t = lr_profile['transform']
    lr_res = lr_t.a
    hr_res = lr_res / scale
    region_left = lr_t.c + c0 * lr_t.a
    region_top  = lr_t.f + r0 * lr_t.e  # t.e is negative

    hr_transform = from_origin(region_left, region_top, hr_res, hr_res)
    hr_profile = lr_profile.copy()
    hr_profile.update({
        'width': hr_w, 'height': hr_h,
        'transform': hr_transform,
        'dtype': 'float32', 'count': 1,
        'compress': 'deflate', 'nodata': np.nan,
    })

    # Save
    region_dir = os.path.join(out_dir, region_name)
    os.makedirs(region_dir, exist_ok=True)
    for i in tqdm(range(n_realizations), desc=f"  Saving {region_name}"):
        out_path = os.path.join(region_dir, f'realization_{i:04d}.tif')
        with rasterio.open(out_path, 'w', **hr_profile) as dst:
            dst.write(realizations[i].astype(np.float32), 1)

    print(f"  Saved {n_realizations} realizations to {region_dir}")

    # Free memory
    del accum, weights, realizations


def auto_tile_full_grid(lr_profile, max_lr_tile=500, overlap_lr=20):
    """Break the full LR grid into overlapping regions."""
    lr_h = lr_profile['height']
    lr_w = lr_profile['width']
    t = lr_profile['transform']
    regions = {}

    step = max_lr_tile - overlap_lr
    row_starts = list(range(0, lr_h, step))
    col_starts = list(range(0, lr_w, step))

    for ri, r0 in enumerate(row_starts):
        for ci, c0 in enumerate(col_starts):
            r1 = min(r0 + max_lr_tile, lr_h)
            c1 = min(c0 + max_lr_tile, lr_w)
            # Convert to UTM bbox
            xmin = t.c + c0 * t.a
            xmax = t.c + c1 * t.a
            ymax = t.f + r0 * t.e
            ymin = t.f + r1 * t.e
            name = f"tile_r{ri:02d}_c{ci:02d}"
            regions[name] = (xmin, ymin, xmax, ymax)

    return regions


def main():
    parser = argparse.ArgumentParser(
        description='Generate stochastic SR realizations from trained diffusion model')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--lr_tif', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_realizations', type=int, default=50)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--residual', action='store_true',
                        help='Model was trained in residual mode; add LR back to sampled output')

    # Region specification (mutually exclusive)
    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument('--test_surveys', action='store_true',
                              help='Process the 3 held-out test survey regions')
    region_group.add_argument('--val_surveys', action='store_true',
                              help='Process the 5 validation survey regions')
    region_group.add_argument('--bbox', type=float, nargs=4,
                              metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                              help='UTM bounding box for a single region')
    region_group.add_argument('--full_grid', action='store_true',
                              help='Process the full LR grid (auto-tiled)')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    cfg = Config()
    cfg.base_ch = args.base_ch

    diffusion = load_model(args.checkpoint, cfg, device)

    print(f"Reading LR grid: {args.lr_tif}")
    lr_arr, lr_profile = read_lr_tif(args.lr_tif)
    print(f"LR grid size: {lr_arr.shape[0]} × {lr_arr.shape[1]}")

    # Determine regions
    if args.test_surveys:
        regions = TEST_SURVEY_REGIONS
        print(f"\nProcessing {len(regions)} test survey regions")
    elif args.val_surveys:
        regions = VAL_SURVEY_REGIONS
        print(f"\nProcessing {len(regions)} validation survey regions")
    elif args.bbox:
        name = args.region_name if hasattr(args, 'region_name') and args.region_name else 'custom_region'
        regions = {name: tuple(args.bbox)}
    elif args.full_grid:
        regions = auto_tile_full_grid(lr_profile)
        print(f"\nAuto-tiled full grid into {len(regions)} regions")

    for region_name, bbox in regions.items():
        process_region(
            diffusion, lr_arr, lr_profile, region_name, bbox,
            patch_size=args.patch_size, stride=args.stride,
            n_realizations=args.n_realizations, ddim_steps=args.ddim_steps,
            device=device, out_dir=args.out_dir, residual=args.residual,
        )

    print("\nAll regions complete.")


if __name__ == '__main__':
    main()
