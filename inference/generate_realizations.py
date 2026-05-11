"""
Generate N stochastic HR realizations from the trained rectified-flow
model over the full LR grid (or any subregion of it).

Pipeline:
    1. Tile the LR grid into overlapping LR patches.
    2. For each (tile-batch × realization-chunk) — Heun-integrate the
       flow on CUDA with bf16 autocast.
    3. Stitch tiles back together with Gaussian-weighted blending.
    4. Denormalize back to elevation (meters).

Memory shape:
    - cfg.n_realizations_per_forward bounds how many realizations share a
      single forward pass for one tile.
    - cfg.tiles_per_batch lets multiple tiles share one forward (each tile
      contributes its own LR conditioning). At patch_size=512 this is
      usually 1; bump it on high-memory GPUs.

The model consumes raw LR tiles directly via cross-attention — there is
no bicubic-upsample step in the inference path.

Usage:
    python -m inference.generate_realizations \
        --checkpoint /path/to/ckpt_final.pt \
        --lr_tif     data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --out_dir    /scratch/.../bathy_realizations/ \
        --n_realizations 50

On Sherlock: use sherlock/inference.sbatch.
"""

import contextlib
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, cfg, device):
    flow = build_rectified_flow_from_checkpoint(checkpoint_path, cfg, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return flow


def make_gaussian_weight(patch_size):
    """2D Gaussian weight mask for patch blending (σ = patch_size/4)."""
    sigma = patch_size / 4.0
    coords = np.linspace(-(patch_size // 2), patch_size // 2, patch_size)
    x, y = np.meshgrid(coords, coords)
    w = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return w.astype(np.float32)


def read_lr_tif(lr_tif_path):
    with rasterio.open(lr_tif_path) as src:
        return src.read(1).astype(np.float32), src.profile.copy()


def make_autocast(device, precision):
    """bf16 autocast on CUDA; nullcontext otherwise."""
    if device.type == 'cuda' and precision == 'bf16':
        def cm():
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return cm
    return contextlib.nullcontext


# ---------------------------------------------------------------------------
# Chunked sampler — n_realizations split into batches of size
# n_realizations_per_forward, optionally across multiple tiles per batch.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample_chunked(flow, lr_tensors, n_realizations, per_fwd, n_steps,
                    cfg_scale, churn, autocast_cm):
    """Generate n_realizations samples for each of len(lr_tensors) tiles.

    Args:
        lr_tensors: list of (1, 1, lr_h, lr_w) tensors on device — one per tile.

    Returns:
        list of (n_realizations, 1, hr_h, hr_w) tensors on device.
    """
    n_tiles = len(lr_tensors)
    # Build a single stacked LR tensor of shape (n_tiles, 1, h, w)
    lr_stack = torch.cat(lr_tensors, dim=0)  # (n_tiles, 1, h, w)

    # Allocate per-tile sample buffers (we accumulate chunks per tile)
    tile_buffers = [[] for _ in range(n_tiles)]

    remaining = n_realizations
    while remaining > 0:
        r = min(per_fwd, remaining)
        # Repeat each tile r times → (n_tiles * r, 1, h, w)
        lr_chunk = lr_stack.repeat_interleave(r, dim=0)
        with autocast_cm():
            samples = flow.sample(
                lr_chunk,
                n_realizations=n_tiles * r,
                n_steps=n_steps,
                cfg_scale=cfg_scale,
                churn=churn,
            )
        # samples: (n_tiles * r, 1, hr_h, hr_w) → (n_tiles, r, 1, hr_h, hr_w)
        hr_h, hr_w = samples.shape[-2:]
        samples = samples.view(n_tiles, r, 1, hr_h, hr_w)
        for ti in range(n_tiles):
            tile_buffers[ti].append(samples[ti])
        remaining -= r

    # Concatenate chunks per tile → (n_realizations, 1, hr, hr)
    return [torch.cat(buf, dim=0) for buf in tile_buffers]


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(flow, lr_arr, lr_profile, cfg, patch_size, stride, sr_ratio,
                  n_realizations, n_steps, cfg_scale, churn, device):
    lr_h, lr_w = lr_arr.shape
    lr_res = lr_profile['transform'].a
    hr_res = lr_res / sr_ratio
    scale = sr_ratio

    hr_h = lr_h * scale
    hr_w = lr_w * scale
    hr_patch = patch_size
    lr_tile = patch_size // scale

    accum = np.zeros((n_realizations, hr_h, hr_w), dtype=np.float32)
    weights = np.zeros((hr_h, hr_w), dtype=np.float32)
    gauss_w = make_gaussian_weight(hr_patch)

    lr_stride = max(1, stride // scale)
    row_positions = list(range(0, lr_h - lr_tile + 1, lr_stride))
    col_positions = list(range(0, lr_w - lr_tile + 1, lr_stride))
    if row_positions[-1] + lr_tile < lr_h:
        row_positions.append(lr_h - lr_tile)
    if col_positions[-1] + lr_tile < lr_w:
        col_positions.append(lr_w - lr_tile)

    # Flat list of (lr_row, lr_col) positions; we will chunk into tile batches.
    all_positions = [(r, c) for r in row_positions for c in col_positions]
    total_tiles = len(all_positions)
    print(f"Generating {n_realizations} realizations for {total_tiles} tiles "
          f"(n_steps={n_steps}, cfg_scale={cfg_scale}, churn={churn})...")
    print(f"Chunking: tiles_per_batch={cfg.tiles_per_batch}, "
          f"n_realizations_per_forward={cfg.n_realizations_per_forward}")
    print(f"Output HR grid size: {hr_h} × {hr_w} pixels at {hr_res:.1f}m resolution")

    autocast_cm = make_autocast(device, cfg.precision)

    # Pre-screen tiles for nodata; build a list of (pos, lr_tensor, mean, std).
    prepared = []
    for (lr_row, lr_col) in all_positions:
        lr_tile_arr = lr_arr[lr_row:lr_row + lr_tile,
                             lr_col:lr_col + lr_tile]
        nodata_frac = (~np.isfinite(lr_tile_arr)).mean()
        if nodata_frac > 0.5:
            continue
        lr_tile_arr = np.where(np.isfinite(lr_tile_arr),
                               lr_tile_arr, np.nanmean(lr_tile_arr))
        lr_norm, tile_mean, tile_std = normalize_patch(lr_tile_arr)
        lr_tensor = torch.from_numpy(
            lr_norm[np.newaxis, np.newaxis]
        ).float().to(device)
        prepared.append(((lr_row, lr_col), lr_tensor, tile_mean, tile_std))

    print(f"Prepared {len(prepared)} valid tiles "
          f"({total_tiles - len(prepared)} skipped for nodata)")

    # Process tiles in batches of cfg.tiles_per_batch.
    tpb = max(1, cfg.tiles_per_batch)
    n_fwd = cfg.n_realizations_per_forward
    progress = tqdm(total=len(prepared), desc="Tiles")
    for start in range(0, len(prepared), tpb):
        batch = prepared[start:start + tpb]
        lr_tensors = [b[1] for b in batch]
        tile_samples = _sample_chunked(
            flow, lr_tensors,
            n_realizations=n_realizations, per_fwd=n_fwd,
            n_steps=n_steps, cfg_scale=cfg_scale, churn=churn,
            autocast_cm=autocast_cm,
        )
        for ((lr_row, lr_col), _, mean, std), samples_t in zip(batch, tile_samples):
            samples_np = samples_t.squeeze(1).cpu().numpy()
            samples_np = denormalize_patch(samples_np, mean, std)
            hr_row = lr_row * scale
            hr_col = lr_col * scale
            accum[:, hr_row:hr_row + hr_patch, hr_col:hr_col + hr_patch] += \
                samples_np * gauss_w[np.newaxis]
            weights[hr_row:hr_row + hr_patch, hr_col:hr_col + hr_patch] += gauss_w
        progress.update(len(batch))
    progress.close()

    w_safe = np.where(weights > 0, weights, 1.0)
    realizations = accum / w_safe[np.newaxis]
    realizations[:, weights == 0] = np.nan

    lr_transform = lr_profile['transform']
    hr_transform = from_origin(lr_transform.c, lr_transform.f, hr_res, hr_res)
    hr_profile = lr_profile.copy()
    hr_profile.update({
        'width':     hr_w,
        'height':    hr_h,
        'transform': hr_transform,
        'dtype':     'float32',
        'count':     1,
        'compress':  'deflate',
        'nodata':    np.nan,
    })

    return realizations, hr_profile


def save_realizations(realizations, hr_profile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = realizations.shape[0]
    for i in tqdm(range(n), desc="Saving GeoTIFFs"):
        out_path = os.path.join(out_dir, f'realization_{i:04d}.tif')
        with rasterio.open(out_path, 'w', **hr_profile) as dst:
            dst.write(realizations[i].astype(np.float32), 1)
    print(f"Saved {n} realizations to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate stochastic SR realizations from a rectified-flow checkpoint')
    parser.add_argument('--checkpoint',     required=True)
    parser.add_argument('--lr_tif',         required=True)
    parser.add_argument('--out_dir',        required=True)
    parser.add_argument('--n_realizations', type=int, default=50)
    parser.add_argument('--n_steps',        type=int,   default=None)
    parser.add_argument('--cfg_scale',      type=float, default=None,
                        help='CFG scale (default Config.cfg_scale = 1.5)')
    parser.add_argument('--churn',          type=float, default=None,
                        help='Langevin churn at inference (default Config.churn = 0.0)')
    parser.add_argument('--patch_size',     type=int,   default=None)
    parser.add_argument('--stride',         type=int,   default=None)
    parser.add_argument('--sr_ratio',       type=int,   default=None)
    parser.add_argument('--base_ch',        type=int,   default=None)
    parser.add_argument('--tiles_per_batch',                type=int, default=None)
    parser.add_argument('--n_realizations_per_forward',     type=int, default=None)
    args = parser.parse_args()

    device = torch.device(get_device())
    print(f"Device: {device}")

    cfg = Config()
    if args.base_ch is not None: cfg.base_ch = args.base_ch
    if args.tiles_per_batch is not None: cfg.tiles_per_batch = args.tiles_per_batch
    if args.n_realizations_per_forward is not None:
        cfg.n_realizations_per_forward = args.n_realizations_per_forward
    patch_size = args.patch_size if args.patch_size is not None else cfg.patch_size
    stride     = args.stride     if args.stride     is not None else cfg.stride
    sr_ratio   = args.sr_ratio   if args.sr_ratio   is not None else cfg.sr_ratio
    n_steps    = args.n_steps    if args.n_steps    is not None else cfg.n_sampling_steps
    cfg_scale  = args.cfg_scale  if args.cfg_scale  is not None else cfg.cfg_scale
    churn      = args.churn      if args.churn      is not None else cfg.churn

    flow = load_model(args.checkpoint, cfg, device)

    print(f"Reading LR grid: {args.lr_tif}")
    lr_arr, lr_profile = read_lr_tif(args.lr_tif)
    print(f"LR grid size: {lr_arr.shape[0]} × {lr_arr.shape[1]}")

    realizations, hr_profile = run_inference(
        flow, lr_arr, lr_profile, cfg,
        patch_size=patch_size, stride=stride, sr_ratio=sr_ratio,
        n_realizations=args.n_realizations,
        n_steps=n_steps, cfg_scale=cfg_scale, churn=churn,
        device=device,
    )

    save_realizations(realizations, hr_profile, args.out_dir)


if __name__ == '__main__':
    main()
