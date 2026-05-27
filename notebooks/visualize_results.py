"""
Generate publication-quality figures for the bathymetric SR paper.

Produces for each test region:
  1. comparison_<region>.png  — 4-panel: LR bicubic | SR mean | HR ground truth | SR std dev
  2. realizations_<region>.png — 6 individual stochastic realizations
  3. uncertainty_<region>.png  — std dev map with survey priority overlay

Usage:
    python3 notebooks/visualize_results.py \
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
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Region metadata
# ---------------------------------------------------------------------------

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
    # Validation surveys
    '600mGully': {
        'hr_file': '600mGully_Topo1m_UTMinterp_utm.tif',
        'title':   '600m Gully',
    },
    'BankFlankHoles': {
        'hr_file': 'BankFlankHoles_MAUV_Topo1m_UTM_utm.tif',
        'title':   'Bank Flank Holes',
    },
    'PockmarkNorth': {
        'hr_file': 'PockmarkNorth_MAUV_Topo1m_UTM_utm.tif',
        'title':   'Pockmark North',
    },
    'LuciaChica2019_970': {
        'hr_file': 'LuciaChica2019_repeat_970m_Topo1m_UTM_utm.tif',
        'title':   'Lucia Chica 2019 (970m)',
    },
}

# Colormap for bathymetry (depth)
BATHY_CMAP = 'Blues_r'
STD_CMAP   = 'hot_r'


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    arr[arr == profile.get('nodata', np.nan)] = np.nan
    return arr, profile


def read_lr_for_region(lr_tif, sr_profile):
    """
    Read the LR grid and bicubic-upsample it to match the SR output extent/resolution.
    Returns a 2D array the same shape as the SR output.
    """
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

    # Bicubic upsample to SR shape
    target_h = sr_profile['height']
    target_w = sr_profile['width']
    if lr_small.shape[0] == 0 or lr_small.shape[1] == 0:
        return np.full((target_h, target_w), np.nan, dtype=np.float32)

    zoom_r = target_h / lr_small.shape[0]
    zoom_c = target_w / lr_small.shape[1]
    lr_up = zoom(lr_small, (zoom_r, zoom_c), order=3).astype(np.float32)
    return lr_up


def read_hr_for_region(hr_path, sr_profile):
    """
    Read HR ground truth and resample to the SR output grid.
    Returns a 2D array the same shape as the SR output.
    """
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


def load_n_realizations(region_dir, n=6):
    """Load the first n realization GeoTIFFs from a region directory."""
    import glob
    files = sorted(glob.glob(os.path.join(region_dir, 'realization_*.tif')))[:n]
    arrays = []
    for f in files:
        arr, _ = read_tif(f)
        arrays.append(arr)
    return arrays


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def bathy_imshow(ax, arr, vmin=None, vmax=None, cmap=BATHY_CMAP):
    """Show a bathymetry array; use shared vmin/vmax if provided."""
    valid = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = np.percentile(valid, 2) if len(valid) else 0
    if vmax is None:
        vmax = np.percentile(valid, 98) if len(valid) else 1
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    return im


def add_scalebar(ax, arr_shape, pixel_size_m=2.0, length_m=1000):
    """Add a simple scale bar to the bottom-right of the axes."""
    h, w = arr_shape
    bar_px = length_m / pixel_size_m
    x0 = w * 0.97 - bar_px
    y0 = h * 0.96
    ax.plot([x0, x0 + bar_px], [y0, y0], 'w-', linewidth=2.5, solid_capstyle='butt')
    ax.text(x0 + bar_px / 2, y0 - h * 0.02, f'{length_m}m',
            color='white', ha='center', va='bottom', fontsize=7, fontweight='bold')


def add_colorbar(fig, ax, im, label, orientation='vertical'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


# ---------------------------------------------------------------------------
# Figure 1: Comparison panel
# ---------------------------------------------------------------------------

def make_comparison_figure(region_name, meta, lr_tif, hr_dir,
                            uncertainty_dir, realizations_dir, out_dir):
    """4-panel: LR bicubic | SR mean | HR ground truth | SR std dev"""

    mean_arr, sr_profile = read_tif(
        os.path.join(uncertainty_dir, region_name, 'mean.tif'))
    std_arr, _  = read_tif(os.path.join(uncertainty_dir, region_name, 'std.tif'))
    priority, _ = read_tif(os.path.join(uncertainty_dir, region_name, 'survey_priority.tif'))

    lr_up = read_lr_for_region(lr_tif, sr_profile)
    hr_gt = read_hr_for_region(os.path.join(hr_dir, meta['hr_file']), sr_profile)

    # Shared elevation colormap range from SR mean
    valid = mean_arr[np.isfinite(mean_arr)]
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"{meta['title']} — Stochastic Super-Resolution Results", fontsize=13)

    panels = [
        (lr_up,   'LR Input\n(32m bicubic → 2m)',  BATHY_CMAP, vmin, vmax),
        (mean_arr,'SR Mean\n(diffusion model)',      BATHY_CMAP, vmin, vmax),
        (hr_gt,   'HR Ground Truth\n(AUV survey)',   BATHY_CMAP, vmin, vmax),
        (std_arr, 'SR Uncertainty\n(std dev, m)',    STD_CMAP,   None, None),
    ]

    for ax, (arr, title, cmap, v0, v1) in zip(axes, panels):
        im = bathy_imshow(ax, arr, vmin=v0, vmax=v1, cmap=cmap)
        ax.set_title(title, fontsize=9)
        add_scalebar(ax, arr.shape)
        add_colorbar(fig, ax, im,
                     label='Elevation (m)' if cmap == BATHY_CMAP else 'Std dev (m)')

    # Overlay survey priority on std dev panel
    priority_rgba = np.zeros((*priority.shape, 4), dtype=np.float32)
    mask = priority == 1
    priority_rgba[mask] = [1.0, 0.9, 0.0, 0.5]  # semi-transparent yellow
    axes[3].imshow(priority_rgba, interpolation='none')
    patch = mpatches.Patch(facecolor=[1.0, 0.9, 0.0, 0.7], label='Top 10% uncertainty')
    axes[3].legend(handles=[patch], loc='lower left', fontsize=7, framealpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'comparison_{region_name}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Stochastic realizations spread
# ---------------------------------------------------------------------------

def make_realizations_figure(region_name, meta, realizations_dir, uncertainty_dir, out_dir):
    """Show 6 individual realizations to illustrate stochastic variation."""

    arrays = load_n_realizations(os.path.join(realizations_dir, region_name), n=6)
    if not arrays:
        print(f"  No realizations found for {region_name}")
        return

    mean_arr, _ = read_tif(os.path.join(uncertainty_dir, region_name, 'mean.tif'))
    valid = mean_arr[np.isfinite(mean_arr)]
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{meta['title']} — Individual Stochastic Realizations", fontsize=13)

    for i, (ax, arr) in enumerate(zip(axes.flat, arrays)):
        im = bathy_imshow(ax, arr, vmin=vmin, vmax=vmax)
        ax.set_title(f'Realization {i+1}', fontsize=9)
        add_scalebar(ax, arr.shape)

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.1)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=BATHY_CMAP,
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Elevation (m)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out_path = os.path.join(out_dir, f'realizations_{region_name}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Detailed uncertainty map
# ---------------------------------------------------------------------------

def make_uncertainty_figure(region_name, meta, uncertainty_dir, out_dir):
    """3-panel uncertainty figure: std dev | CoV | IQR with priority overlay."""

    std_arr,  profile = read_tif(os.path.join(uncertainty_dir, region_name, 'std.tif'))
    cov_arr,  _       = read_tif(os.path.join(uncertainty_dir, region_name, 'cov.tif'))
    iqr_arr,  _       = read_tif(os.path.join(uncertainty_dir, region_name, 'iqr.tif'))
    mean_arr, _       = read_tif(os.path.join(uncertainty_dir, region_name, 'mean.tif'))
    priority, _       = read_tif(os.path.join(uncertainty_dir, region_name, 'survey_priority.tif'))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{meta['title']} — Uncertainty Quantification", fontsize=13)

    # Std dev
    valid_std = std_arr[np.isfinite(std_arr)]
    vmax_std = np.percentile(valid_std, 98)
    im0 = bathy_imshow(axes[0], std_arr, vmin=0, vmax=vmax_std, cmap=STD_CMAP)
    axes[0].set_title('Std Dev (m)\ninter-realization spread', fontsize=9)
    add_colorbar(fig, axes[0], im0, 'Std dev (m)')
    add_scalebar(axes[0], std_arr.shape)

    # CoV
    valid_cov = cov_arr[np.isfinite(cov_arr)]
    vmax_cov = np.percentile(valid_cov, 98) if len(valid_cov) else 0.01
    im1 = bathy_imshow(axes[1], cov_arr, vmin=0, vmax=vmax_cov, cmap=STD_CMAP)
    axes[1].set_title('Coeff. of Variation\n(std / |mean|)', fontsize=9)
    add_colorbar(fig, axes[1], im1, 'CoV')
    add_scalebar(axes[1], cov_arr.shape)

    # IQR with priority overlay
    valid_iqr = iqr_arr[np.isfinite(iqr_arr)]
    vmax_iqr = np.percentile(valid_iqr, 98) if len(valid_iqr) else 1.0
    im2 = bathy_imshow(axes[2], iqr_arr, vmin=0, vmax=vmax_iqr, cmap=STD_CMAP)
    axes[2].set_title('IQR (m)\nwith survey priority overlay', fontsize=9)
    add_colorbar(fig, axes[2], im2, 'IQR (m)')
    add_scalebar(axes[2], iqr_arr.shape)

    priority_rgba = np.zeros((*priority.shape, 4), dtype=np.float32)
    priority_rgba[priority == 1] = [1.0, 0.9, 0.0, 0.5]
    axes[2].imshow(priority_rgba, interpolation='none')
    patch = mpatches.Patch(facecolor=[1.0, 0.9, 0.0, 0.7], label='Top 10% uncertainty\n(survey priority)')
    axes[2].legend(handles=[patch], loc='lower left', fontsize=7, framealpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'uncertainty_{region_name}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate SR visualization figures')
    parser.add_argument('--realizations_dir', required=True)
    parser.add_argument('--uncertainty_dir',  required=True)
    parser.add_argument('--lr_tif',           required=True)
    parser.add_argument('--hr_dir',           required=True)
    parser.add_argument('--out_dir',          required=True)
    parser.add_argument('--regions', nargs='+', default=None,
                        help='Subset of region names to process (default: all)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    regions_to_run = {k: v for k, v in REGIONS.items()
                      if args.regions is None or k in args.regions}

    for region_name, meta in regions_to_run.items():
        print(f"\n--- {meta['title']} ---")
        hr_path = os.path.join(args.hr_dir, meta['hr_file'])
        if not os.path.exists(hr_path):
            print(f"  [skip] HR file not found: {hr_path}")
            continue
        if not os.path.isdir(os.path.join(args.uncertainty_dir, region_name)):
            print(f"  [skip] Uncertainty maps not found for {region_name}")
            continue

        make_comparison_figure(region_name, meta,
                               args.lr_tif, args.hr_dir,
                               args.uncertainty_dir, args.realizations_dir,
                               args.out_dir)

        make_realizations_figure(region_name, meta,
                                 args.realizations_dir, args.uncertainty_dir,
                                 args.out_dir)

        make_uncertainty_figure(region_name, meta,
                                args.uncertainty_dir, args.out_dir)

    print(f"\nAll figures saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
