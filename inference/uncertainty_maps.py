"""
Compute uncertainty statistics from a set of stochastic realizations
and export them as GeoTIFFs.

Outputs (all at 2m resolution, same georeferencing as realizations):
    mean.tif           — mean realization (best estimate)
    std.tif            — pixel-wise standard deviation (uncertainty in meters)
    cov.tif            — coefficient of variation (std / |mean|)
    iqr.tif            — interquartile range (robust spread)
    p10.tif / p90.tif  — 10th and 90th percentile realizations
    survey_priority.tif — binary mask: top X% highest uncertainty pixels

Usage:
    python inference/uncertainty_maps.py \
        --realizations_dir /scratch/users/scdobbs/bathy_realizations/ \
        --out_dir results/uncertainty/ \
        --priority_fraction 0.1
"""

import os
import sys
import argparse
import glob
import numpy as np
import rasterio
from tqdm import tqdm


def load_realizations(realizations_dir):
    """
    Load all realization GeoTIFFs from a directory.
    Returns (N, H, W) float32 array and the rasterio profile of the first file.
    """
    tif_files = sorted(glob.glob(os.path.join(realizations_dir, 'realization_*.tif')))
    if not tif_files:
        raise FileNotFoundError(f"No realization_*.tif files in {realizations_dir}")

    print(f"Loading {len(tif_files)} realizations...")
    arrays = []
    profile = None
    for f in tqdm(tif_files, desc="Reading"):
        with rasterio.open(f) as src:
            if profile is None:
                profile = src.profile.copy()
            arrays.append(src.read(1).astype(np.float32))

    return np.stack(arrays, axis=0), profile  # (N, H, W)


def compute_uncertainty_stats(realizations):
    """
    Compute per-pixel uncertainty statistics from an (N, H, W) array.
    Returns a dict of (H, W) arrays.
    """
    mean = np.nanmean(realizations, axis=0)
    std  = np.nanstd(realizations, axis=0)

    # Coefficient of variation — relative uncertainty
    # Avoid division by very small depths (near zero)
    cov = np.where(np.abs(mean) > 0.1, std / np.abs(mean), np.nan)

    p10 = np.nanpercentile(realizations, 10, axis=0)
    p90 = np.nanpercentile(realizations, 90, axis=0)
    iqr = np.nanpercentile(realizations, 75, axis=0) - \
          np.nanpercentile(realizations, 25, axis=0)

    return {
        'mean': mean,
        'std':  std,
        'cov':  cov,
        'p10':  p10,
        'p90':  p90,
        'iqr':  iqr,
    }


def make_survey_priority_mask(std_map, priority_fraction):
    """
    Return a binary mask: 1 where std is in the top priority_fraction of valid pixels.
    These are the highest-uncertainty areas that should be targeted for future surveys.
    """
    valid = std_map[np.isfinite(std_map)]
    if len(valid) == 0:
        return np.zeros_like(std_map, dtype=np.uint8)
    threshold = np.percentile(valid, 100 * (1 - priority_fraction))
    mask = (std_map >= threshold).astype(np.uint8)
    mask[~np.isfinite(std_map)] = 0
    return mask


def save_tif(array, profile, out_path, nodata=None, dtype='float32'):
    """Save a 2D array as a single-band GeoTIFF."""
    p = profile.copy()
    p.update({'count': 1, 'dtype': dtype, 'compress': 'deflate'})
    if nodata is not None:
        p['nodata'] = nodata
    with rasterio.open(out_path, 'w', **p) as dst:
        dst.write(array.astype(dtype), 1)


def print_summary(stats):
    """Print a brief summary of the uncertainty statistics."""
    std = stats['std']
    valid = std[np.isfinite(std)]
    print("\nUncertainty summary:")
    print(f"  Std dev (meters): min={valid.min():.3f}  median={np.median(valid):.3f}"
          f"  95th={np.percentile(valid, 95):.3f}  max={valid.max():.3f}")
    cov = stats['cov']
    valid_cov = cov[np.isfinite(cov)]
    if len(valid_cov):
        print(f"  CoV:              min={valid_cov.min():.4f}  "
              f"median={np.median(valid_cov):.4f}  95th={np.percentile(valid_cov,95):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute uncertainty maps from SR realizations')
    parser.add_argument('--realizations_dir', required=True,
                        help='Directory containing realization_*.tif files')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for uncertainty GeoTIFFs')
    parser.add_argument('--priority_fraction', type=float, default=0.1,
                        help='Top fraction of high-uncertainty pixels to flag (default: 0.1)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    realizations, profile = load_realizations(args.realizations_dir)
    print(f"Loaded {realizations.shape[0]} realizations, "
          f"grid {realizations.shape[1]}×{realizations.shape[2]}")

    print("Computing statistics...")
    stats = compute_uncertainty_stats(realizations)
    print_summary(stats)

    # Save uncertainty layers
    for name, arr in stats.items():
        out_path = os.path.join(args.out_dir, f'{name}.tif')
        save_tif(arr, profile, out_path, nodata=np.nan)
        print(f"  Saved {out_path}")

    # Survey priority mask
    priority = make_survey_priority_mask(stats['std'], args.priority_fraction)
    priority_path = os.path.join(args.out_dir, 'survey_priority.tif')
    save_tif(priority, profile, priority_path, nodata=255, dtype='uint8')
    n_priority = int(priority.sum())
    total_valid = int(np.isfinite(stats['std']).sum())
    area_km2 = n_priority * (2.0 ** 2) / 1e6  # pixels × (2m)² → km²
    print(f"  Survey priority: {n_priority:,} pixels flagged ({n_priority/total_valid*100:.1f}%) "
          f"= {area_km2:.2f} km²")
    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
