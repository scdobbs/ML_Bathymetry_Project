"""
Transect profile plot: LR bicubic vs SR mean (±spread) vs HR ground truth.

For each test region, automatically selects the most informative transect
(the row with maximum relief variance), then plots elevation profiles for
all three data sources along that line. This directly answers whether the
model recovers sub-32m structure not present in the LR input.

Usage:
    python3 notebooks/transect_plot.py \
        --realizations_dir /scratch/users/scdobbs/bathy_diffusion/realizations \
        --uncertainty_dir  /scratch/users/scdobbs/bathy_diffusion/uncertainty_maps \
        --lr_tif           data_for_model/MorroBay_all_2020_interp_32m_converted.tif \
        --hr_dir           data_for_model/ \
        --out_dir          results/figures/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.ndimage import zoom
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIONS = {
    'HeadlessCanyon': {
        'hr_file': 'HeadlessCanyon_Topo1m_converted.tif',
        'title':   'Headless Canyon',
    },
    'LuciaChica_2007': {
        'hr_file': 'LuciaChica_2007_Topo1m_UTM_utm.tif',
        'title':   'Lucia Chica (2007)',
    },
    'BankTopEastofCanyon3': {
        'hr_file': 'BankTopEastofCanyon3_Topo1m_converted.tif',
        'title':   'Bank Top East of Canyon',
    },
}


def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, profile


def read_lr_upsampled(lr_tif, sr_profile):
    sr_bounds = rasterio.transform.array_bounds(
        sr_profile['height'], sr_profile['width'], sr_profile['transform'])
    with rasterio.open(lr_tif) as src:
        window = from_bounds(*sr_bounds, src.transform)
        lr_small = src.read(1, window=window,
                            resampling=Resampling.bilinear,
                            boundless=True,
                            fill_value=np.nan).astype(np.float32)
        if src.nodata is not None:
            lr_small[lr_small == src.nodata] = np.nan
    target_h = sr_profile['height']
    target_w = sr_profile['width']
    if lr_small.shape[0] == 0 or lr_small.shape[1] == 0:
        return np.full((target_h, target_w), np.nan, dtype=np.float32)
    zoom_r = target_h / lr_small.shape[0]
    zoom_c = target_w / lr_small.shape[1]
    return zoom(lr_small, (zoom_r, zoom_c), order=3).astype(np.float32)


def read_hr_resampled(hr_path, sr_profile):
    sr_bounds = rasterio.transform.array_bounds(
        sr_profile['height'], sr_profile['width'], sr_profile['transform'])
    with rasterio.open(hr_path) as src:
        window = from_bounds(*sr_bounds, src.transform)
        hr = src.read(1, window=window,
                      out_shape=(sr_profile['height'], sr_profile['width']),
                      resampling=Resampling.bilinear,
                      boundless=True,
                      fill_value=np.nan).astype(np.float32)
        if src.nodata is not None:
            hr[hr == src.nodata] = np.nan
    return hr


def load_realization_stack(region_dir, n=50):
    """Load all realizations into a (N, H, W) array."""
    files = sorted(glob.glob(os.path.join(region_dir, 'realization_*.tif')))[:n]
    if not files:
        return None
    arrs = []
    for f in files:
        arr, _ = read_tif(f)
        arrs.append(arr)
    return np.stack(arrs, axis=0)


def best_transect(arr, axis='row', n_candidates=10):
    """
    Find the row (or column) with the highest valid-pixel std dev —
    i.e., the most topographically varied transect.
    Returns the index.
    """
    h, w = arr.shape
    if axis == 'row':
        indices = np.linspace(h * 0.1, h * 0.9, n_candidates).astype(int)
        stds = [np.nanstd(arr[i, :]) for i in indices]
    else:
        indices = np.linspace(w * 0.1, w * 0.9, n_candidates).astype(int)
        stds = [np.nanstd(arr[:, i]) for i in indices]
    best = indices[np.argmax(stds)]
    return int(best)


def make_transect_figure(region_name, meta, lr_tif, hr_dir,
                         uncertainty_dir, realizations_dir, out_dir,
                         pixel_size_m=2.0):

    mean_arr, sr_profile = read_tif(os.path.join(uncertainty_dir, region_name, 'mean.tif'))
    p10_arr,  _          = read_tif(os.path.join(uncertainty_dir, region_name, 'p10.tif'))
    p90_arr,  _          = read_tif(os.path.join(uncertainty_dir, region_name, 'p90.tif'))

    lr_up = read_lr_upsampled(lr_tif, sr_profile)
    hr_gt = read_hr_resampled(os.path.join(hr_dir, meta['hr_file']), sr_profile)

    # ---- pick best row and best col transects ----
    row_idx = best_transect(mean_arr, axis='row')
    col_idx = best_transect(mean_arr, axis='col')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{meta['title']} — Elevation Profile Transects\n"
                 f"Comparing LR bicubic input vs SR diffusion output vs HR ground truth",
                 fontsize=12)

    for ax_map, ax_prof, arr_name, idx, axis in [
        (axes[0, 0], axes[0, 1], 'row', row_idx, 'row'),
        (axes[1, 0], axes[1, 1], 'col', col_idx, 'col'),
    ]:
        # Left panel: map with transect line overlaid
        valid = mean_arr[np.isfinite(mean_arr)]
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
        ax_map.imshow(mean_arr, cmap='Blues_r', vmin=vmin, vmax=vmax,
                      interpolation='bilinear')
        h, w = mean_arr.shape
        if axis == 'row':
            ax_map.axhline(idx, color='red', linewidth=1.5, linestyle='--')
            ax_map.set_title(f'SR Mean — horizontal transect (row {idx})', fontsize=9)
            dist = np.arange(w) * pixel_size_m / 1000  # km
            lr_line   = lr_up[idx, :]
            sr_line   = mean_arr[idx, :]
            hr_line   = hr_gt[idx, :]
            p10_line  = p10_arr[idx, :]
            p90_line  = p90_arr[idx, :]
            xlabel = 'Distance W→E (km)'
        else:
            ax_map.axvline(idx, color='red', linewidth=1.5, linestyle='--')
            ax_map.set_title(f'SR Mean — vertical transect (col {idx})', fontsize=9)
            dist = np.arange(h) * pixel_size_m / 1000  # km
            lr_line   = lr_up[:, idx]
            sr_line   = mean_arr[:, idx]
            hr_line   = hr_gt[:, idx]
            p10_line  = p10_arr[:, idx]
            p90_line  = p90_arr[:, idx]
            xlabel = 'Distance N→S (km)'
        ax_map.set_xticks([]); ax_map.set_yticks([])

        # Right panel: profile plot
        ax_prof.fill_between(dist, p10_line, p90_line,
                             alpha=0.25, color='steelblue', label='SR p10–p90')
        ax_prof.plot(dist, lr_line,  color='orange',    linewidth=1.5,
                     label='LR bicubic (32m→2m)', zorder=3)
        ax_prof.plot(dist, sr_line,  color='steelblue', linewidth=1.5,
                     label='SR mean (diffusion)', zorder=4)
        ax_prof.plot(dist, hr_line,  color='black',     linewidth=1.0,
                     alpha=0.7, label='HR ground truth (2m AUV)', zorder=5)

        # Mark NaN gaps in HR as gray spans
        nan_mask = ~np.isfinite(hr_line)
        if nan_mask.any():
            ax_prof.fill_between(dist, ax_prof.get_ylim()[0], ax_prof.get_ylim()[1],
                                 where=nan_mask, color='lightgray', alpha=0.4,
                                 label='No HR data')

        ax_prof.set_xlabel(xlabel, fontsize=9)
        ax_prof.set_ylabel('Elevation (m)', fontsize=9)
        ax_prof.legend(fontsize=7, loc='best')
        ax_prof.tick_params(labelsize=8)
        ax_prof.grid(True, alpha=0.3)

        # Compute simple stats for annotation
        valid_mask = np.isfinite(sr_line) & np.isfinite(hr_line)
        if valid_mask.sum() > 10:
            rmse = np.sqrt(np.nanmean((sr_line[valid_mask] - hr_line[valid_mask])**2))
            lr_rmse = np.sqrt(np.nanmean((lr_line[valid_mask] - hr_line[valid_mask])**2))
            ax_prof.set_title(
                f'Profile: SR RMSE={rmse:.2f}m  |  LR RMSE={lr_rmse:.2f}m',
                fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'transect_{region_name}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Transect profile plots')
    parser.add_argument('--realizations_dir', required=True)
    parser.add_argument('--uncertainty_dir',  required=True)
    parser.add_argument('--lr_tif',           required=True)
    parser.add_argument('--hr_dir',           required=True)
    parser.add_argument('--out_dir',          required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for region_name, meta in REGIONS.items():
        print(f"\n--- {meta['title']} ---")
        hr_path = os.path.join(args.hr_dir, meta['hr_file'])
        if not os.path.exists(hr_path):
            print(f"  [skip] HR file not found: {hr_path}")
            continue
        if not os.path.isdir(os.path.join(args.uncertainty_dir, region_name)):
            print(f"  [skip] Uncertainty maps not found for {region_name}")
            continue

        make_transect_figure(region_name, meta,
                             args.lr_tif, args.hr_dir,
                             args.uncertainty_dir, args.realizations_dir,
                             args.out_dir)

    print(f"\nAll transect figures saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
