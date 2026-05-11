"""
Extract paired (LR, HR) patches from all areas where high-resolution
AUV surveys overlap with the 32m MBES grid.

Outputs one .npz file per patch containing:
    'lr':    float32 (1, patch_size, patch_size) — LR upsampled & normalized
    'hr':    float32 (1, patch_size, patch_size) — HR normalized
    'stats': float32 (2,) — [mean, std] of the HR patch (for denormalization)

Usage:
    python preprocessing/extract_paired_patches.py \
        --lr_path data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --hr_dir  data_for_model/ \
        --out_dir /scratch/users/$USER/bathy_patches/ \
        --patch_size 64 \
        --stride 32 \
        --nodata_threshold 0.1 \
        --flat_threshold 0.05
"""

import os
import argparse
import glob
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from tqdm import tqdm

from preprocessing.data_utils import normalize_patch, bicubic_upsample


# Surveys held out for testing — never included in train or val
HELD_OUT_SURVEYS = {
    'LuciaChica_2007_Topo1m_UTM_utm.tif',
    'HeadlessCanyon_Topo1m_converted.tif',
    'BankTopEastofCanyon3_Topo1m_converted.tif',
}


def get_overlap_window(hr_src, lr_src):
    """
    Compute the rasterio Window in the HR raster that overlaps with the LR raster.
    Returns None if there is no overlap.
    """
    hr_bounds = hr_src.bounds
    lr_bounds = lr_src.bounds

    # Intersection in geographic space
    left   = max(hr_bounds.left,   lr_bounds.left)
    bottom = max(hr_bounds.bottom, lr_bounds.bottom)
    right  = min(hr_bounds.right,  lr_bounds.right)
    top    = min(hr_bounds.top,    lr_bounds.top)

    if left >= right or bottom >= top:
        return None  # no overlap

    # Convert geographic bounds to pixel window in HR raster
    window = from_bounds(left, bottom, right, top, hr_src.transform)
    return window, (left, bottom, right, top)


def _mask_nodata(arr, nodata_val):
    """Replace the raster's nodata sentinel value with NaN."""
    if nodata_val is not None and np.isfinite(nodata_val):
        arr = np.where(arr == nodata_val, np.nan, arr)
    return arr


def read_hr_patch(hr_src, row_off, col_off, patch_size):
    """Read a patch_size×patch_size window from the HR raster."""
    window = rasterio.windows.Window(col_off, row_off, patch_size, patch_size)
    arr = hr_src.read(1, window=window, boundless=True,
                      fill_value=np.nan).astype(np.float32)
    return _mask_nodata(arr, hr_src.nodata)


def read_lr_patch_for_hr_window(lr_src, hr_src, row_off, col_off, patch_size, lr_patch_size):
    """
    For a given HR window (row_off, col_off, patch_size), find the
    corresponding geographic extent and read the matching LR pixels.
    Returns a (lr_patch_size, lr_patch_size) float32 array.
    """
    # Convert HR pixel offsets to geographic bounds
    hr_transform = hr_src.transform
    left  = hr_transform.c + col_off * hr_transform.a
    top   = hr_transform.f + row_off * hr_transform.e
    right = left  + patch_size * hr_transform.a
    bottom = top  + patch_size * hr_transform.e  # e is negative

    # Read the corresponding LR window
    lr_window = from_bounds(left, bottom, right, top, lr_src.transform)
    arr = lr_src.read(1, window=lr_window, out_shape=(lr_patch_size, lr_patch_size),
                      resampling=Resampling.bilinear, boundless=True,
                      fill_value=np.nan).astype(np.float32)
    return _mask_nodata(arr, lr_src.nodata)


def patch_is_valid(hr_patch, lr_patch, nodata_threshold, flat_threshold):
    """
    Return True if the patch passes quality checks.
    Rejects patches with too many nodata pixels or too-flat terrain.
    """
    hr_valid = np.isfinite(hr_patch)
    if hr_valid.mean() < (1.0 - nodata_threshold):
        return False
    lr_valid = np.isfinite(lr_patch)
    if lr_valid.mean() < (1.0 - nodata_threshold):
        return False
    # Reject flat patches (ocean floor with no relief)
    hr_range = np.nanmax(hr_patch) - np.nanmin(hr_patch)
    if hr_range < flat_threshold:
        return False
    return True


def extract_patches_from_survey(hr_path, lr_src, out_dir, patch_size, stride,
                                 nodata_threshold, flat_threshold, split_label):
    """
    Extract all valid patches from one HR survey file.
    Saves .npz files to out_dir/split_label/.
    Returns count of patches saved.
    """
    os.makedirs(os.path.join(out_dir, split_label), exist_ok=True)
    survey_name = os.path.splitext(os.path.basename(hr_path))[0]
    lr_patch_size = patch_size // 16  # 32m / 2m = 16× ratio; 4×4 LR pixels for 64×64 HR

    count = 0
    with rasterio.open(hr_path) as hr_src:
        result = get_overlap_window(hr_src, lr_src)
        if result is None:
            print(f"  [skip] No overlap: {survey_name}")
            return 0

        overlap_window, overlap_bounds = result

        # Get pixel offsets of the overlap region in the HR raster
        col_start = max(0, int(overlap_window.col_off))
        row_start = max(0, int(overlap_window.row_off))
        col_end   = min(hr_src.width,  col_start + int(overlap_window.width))
        row_end   = min(hr_src.height, row_start + int(overlap_window.height))

        # For transect-style surveys (thin data footprints with large bounding boxes),
        # reading the whole overlap area at coarse resolution to find valid row/col bands
        # avoids iterating over thousands of all-nodata patch positions.
        coarse_scale = max(1, patch_size // 8)   # read at 1/8 resolution for speed
        overview_h = max(1, (row_end - row_start) // coarse_scale)
        overview_w = max(1, (col_end - col_start) // coarse_scale)
        overview_win = rasterio.windows.Window(col_start, row_start,
                                               col_end - col_start, row_end - row_start)
        overview = hr_src.read(1, window=overview_win,
                               out_shape=(overview_h, overview_w),
                               resampling=Resampling.nearest,
                               boundless=True, fill_value=np.nan)
        overview = _mask_nodata(overview.astype(np.float32), hr_src.nodata)
        valid_mask = np.isfinite(overview)
        if valid_mask.mean() < 0.01:
            print(f"  [skip] Nearly all nodata in overlap: {survey_name}")
            return 0

        # Find valid row/col bands in the overview to constrain iteration
        valid_rows_ov = np.where(valid_mask.any(axis=1))[0]
        valid_cols_ov = np.where(valid_mask.any(axis=0))[0]
        row_start_tight = row_start + int(valid_rows_ov[0])  * coarse_scale
        row_end_tight   = row_start + int(valid_rows_ov[-1]) * coarse_scale + coarse_scale
        col_start_tight = col_start + int(valid_cols_ov[0])  * coarse_scale
        col_end_tight   = col_start + int(valid_cols_ov[-1]) * coarse_scale + coarse_scale
        row_end_tight   = min(row_end_tight, row_end)
        col_end_tight   = min(col_end_tight, col_end)

        # Slide a patch_size window over the tightened valid region
        row_positions = range(row_start_tight, row_end_tight - patch_size + 1, stride)
        col_positions = range(col_start_tight, col_end_tight - patch_size + 1, stride)

        for row_off in row_positions:
            for col_off in col_positions:
                # Fast pre-screen using the overview before doing expensive rasterio reads.
                # Map the HR patch center back to overview coordinates.
                ov_row = int((row_off - row_start) / coarse_scale)
                ov_col = int((col_off - col_start) / coarse_scale)
                ov_row = min(ov_row, overview_h - 1)
                ov_col = min(ov_col, overview_w - 1)
                # Check a small neighbourhood in the overview; skip if all nodata
                r0 = max(0, ov_row - 1); r1 = min(overview_h, ov_row + 2)
                c0 = max(0, ov_col - 1); c1 = min(overview_w, ov_col + 2)
                if not valid_mask[r0:r1, c0:c1].any():
                    continue

                hr_patch = read_hr_patch(hr_src, row_off, col_off, patch_size)
                lr_patch_raw = read_lr_patch_for_hr_window(
                    lr_src, hr_src, row_off, col_off, patch_size, lr_patch_size)

                if not patch_is_valid(hr_patch, lr_patch_raw, nodata_threshold, flat_threshold):
                    continue

                # Replace any remaining NaNs with local mean
                hr_patch = np.where(np.isfinite(hr_patch), hr_patch,
                                    np.nanmean(hr_patch))
                lr_patch_raw = np.where(np.isfinite(lr_patch_raw), lr_patch_raw,
                                        np.nanmean(lr_patch_raw))

                # Normalize using HR patch statistics
                hr_norm, mean, std = normalize_patch(hr_patch)

                # Upsample LR to patch_size and apply same normalization
                lr_up = bicubic_upsample(lr_patch_raw, patch_size)
                lr_norm = (lr_up - mean) / std

                fname = f"{survey_name}_r{row_off:05d}_c{col_off:05d}.npz"
                out_path = os.path.join(out_dir, split_label, fname)
                np.savez_compressed(
                    out_path,
                    lr=lr_norm[np.newaxis].astype(np.float32),
                    hr=hr_norm[np.newaxis].astype(np.float32),
                    stats=np.array([mean, std], dtype=np.float32)
                )
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='Extract paired LR/HR bathymetry patches')
    parser.add_argument('--lr_path', required=True,
                        help='Path to 32m LR GeoTIFF')
    parser.add_argument('--hr_dir', required=True,
                        help='Directory containing HR GeoTIFF files')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for .npz patch files')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='HR patch size in pixels (default: 64)')
    parser.add_argument('--stride', type=int, default=32,
                        help='Stride between patches in HR pixels (default: 32)')
    parser.add_argument('--nodata_threshold', type=float, default=0.1,
                        help='Max fraction of nodata pixels allowed (default: 0.1)')
    parser.add_argument('--flat_threshold', type=float, default=0.05,
                        help='Min elevation range in meters to accept patch (default: 0.05)')
    args = parser.parse_args()

    # Find all HR survey files
    hr_files = sorted(
        glob.glob(os.path.join(args.hr_dir, '*.tif'))
    )
    # Exclude the LR grid itself
    hr_files = [f for f in hr_files if os.path.basename(f) != os.path.basename(args.lr_path)]

    print(f"Found {len(hr_files)} HR survey files")
    print(f"Patch size: {args.patch_size}×{args.patch_size} HR pixels "
          f"({args.patch_size * 2}m × {args.patch_size * 2}m)")
    print(f"LR patch: {args.patch_size // 16}×{args.patch_size // 16} pixels "
          f"before upsampling")

    total_train = total_val = total_test = 0

    # Val surveys: every 5th non-test survey
    train_files = [f for f in hr_files if os.path.basename(f) not in HELD_OUT_SURVEYS]
    test_files  = [f for f in hr_files if os.path.basename(f) in HELD_OUT_SURVEYS]
    val_files   = train_files[::5]           # every 5th
    train_files = [f for f in train_files if f not in val_files]

    print(f"\nSplit: {len(train_files)} train | {len(val_files)} val | {len(test_files)} test surveys")

    with rasterio.open(args.lr_path) as lr_src:
        for split_label, file_list in [('train', train_files),
                                        ('val',   val_files),
                                        ('test',  test_files)]:
            print(f"\n--- {split_label.upper()} ---")
            split_count = 0
            for hr_path in tqdm(file_list, desc=split_label):
                n = extract_patches_from_survey(
                    hr_path, lr_src, args.out_dir, args.patch_size,
                    args.stride, args.nodata_threshold, args.flat_threshold,
                    split_label)
                print(f"  {os.path.basename(hr_path)}: {n} patches")
                split_count += n

            print(f"  Total {split_label}: {split_count} patches")
            if split_label == 'train':
                total_train = split_count
            elif split_label == 'val':
                total_val = split_count
            else:
                total_test = split_count

    print(f"\nDone. Total: {total_train} train / {total_val} val / {total_test} test patches")
    print(f"Saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
