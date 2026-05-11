#!/bin/bash
#SBATCH --job-name=region_v4
#SBATCH --partition=serc
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/region_v4_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/region_v4_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
cd "$PROJ_DIR"
mkdir -p results/figures_v4
echo "Started: $(date)"

python3 - <<'PYEOF'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.path.insert(0, '/home/users/scdobbs/GAN_Bathymetry_Project')

import torch
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from tqdm import tqdm

from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from training.config import Config
from preprocessing.data_utils import normalize_patch, denormalize_patch, bicubic_upsample

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── Load v4 model ─────────────────────────────────────────────────────────
cfg = Config()
cfg.base_ch = 64
state = torch.load('/scratch/users/scdobbs/bathy_diffusion_v4/checkpoints/ckpt_final.pt',
                   map_location=device, weights_only=False)
unet = BathyUNet(base_ch=cfg.base_ch).to(device)
diffusion = GaussianDiffusion(unet, n_timesteps=1000).to(device)
diffusion.load_state_dict(state.get('ema', state.get('model')))
diffusion.eval()
print("Model loaded.")

LR_TIF  = '/home/users/scdobbs/GAN_Bathymetry_Project/data_for_model/MorroBay_all_2020_interp_32m_converted.tif'
DATA_DIR = '/home/users/scdobbs/GAN_Bathymetry_Project/data_for_model'

SURVEYS = {
    'PockmarkNorth': 'PockmarkNorth_MAUV_Topo1m_UTM_utm.tif',
    'LuciaChica2019_970m': 'LuciaChica2019_repeat_970m_Topo1m_UTM_utm.tif',
}

N_REAL    = 10   # realizations per tile (enough for uncertainty map)
PATCH_HR  = 64   # HR patch size
SCALE     = 16
PATCH_LR  = PATCH_HR // SCALE   # 4 LR pixels
STRIDE_LR = 2    # 2 LR pixel stride → 32m stride → heavy overlap

def make_gaussian_weight(size):
    sigma = size / 4.0
    c = np.linspace(-(size//2), size//2, size)
    x, y = np.meshgrid(c, c)
    w = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return w.astype(np.float32)

NODATA_SENTINELS = (-9999, -9998, -32767, -32768)

def mask_nodata(arr, nodata_val=None):
    """Replace nodata sentinels and any implausibly large values with NaN."""
    arr = arr.copy()
    if nodata_val is not None:
        arr[arr == nodata_val] = np.nan
    for s in NODATA_SENTINELS:
        arr[arr == s] = np.nan
    # Bathymetry off Morro Bay: depths should be between +50 and -2000 m
    arr[(arr < -2000) | (arr > 50)] = np.nan
    return arr

def load_lr(lr_tif):
    with rasterio.open(lr_tif) as src:
        arr      = src.read(1).astype(np.float32)
        nodata   = src.nodata
        prof     = src.profile.copy()
    arr = mask_nodata(arr, nodata)
    return arr, prof

def resample_hr_to_2m(hr_path, target_crs, out_bounds, out_res=2.0):
    """Load HR tif (1m), resample to 2m, crop to out_bounds in target_crs."""
    left, bottom, right, top = out_bounds
    w = int(round((right - left)  / out_res))
    h = int(round((top  - bottom) / out_res))
    dst_transform = from_origin(left, top, out_res, out_res)
    dst_arr = np.full((h, w), np.nan, dtype=np.float32)

    with rasterio.open(hr_path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    dst_arr = mask_nodata(dst_arr, src_nodata)
    return dst_arr, dst_transform

def generate_sr(lr_arr, lr_profile, bounds, n_real):
    """Tile the LR region and generate SR realizations; return (mean, std, lr_2m)."""
    t = lr_profile['transform']
    lr_res = t.a   # e.g. 32 m
    xmin, ymin, xmax, ymax = bounds

    # LR pixel coords of the region
    c0 = max(0, int((xmin - t.c) / t.a))
    c1 = min(lr_profile['width'],  int(np.ceil((xmax - t.c) / t.a)))
    r0 = max(0, int((ymax - t.f) / t.e))
    r1 = min(lr_profile['height'], int(np.ceil((ymin - t.f) / t.e)))

    lr_region = lr_arr[r0:r1, c0:c1]
    lr_h, lr_w = lr_region.shape
    hr_h, hr_w = lr_h * SCALE, lr_w * SCALE

    print(f"  LR region: {lr_h}×{lr_w} → HR: {hr_h}×{hr_w}")

    accum   = np.zeros((n_real, hr_h, hr_w), dtype=np.float32)
    weights = np.zeros((hr_h, hr_w), dtype=np.float32)
    gauss_w = make_gaussian_weight(PATCH_HR)

    row_pos = list(range(0, lr_h - PATCH_LR + 1, STRIDE_LR))
    col_pos = list(range(0, lr_w - PATCH_LR + 1, STRIDE_LR))
    if row_pos[-1] + PATCH_LR < lr_h: row_pos.append(lr_h - PATCH_LR)
    if col_pos[-1] + PATCH_LR < lr_w: col_pos.append(lr_w - PATCH_LR)

    with torch.no_grad():
        for lr_row in tqdm(row_pos, desc='  tiling'):
            for lr_col in col_pos:
                tile = lr_region[lr_row:lr_row+PATCH_LR, lr_col:lr_col+PATCH_LR].copy()
                tile = mask_nodata(tile)
                if (~np.isfinite(tile)).mean() > 0.5:
                    continue
                tile = np.where(np.isfinite(tile), tile, np.nanmean(tile))

                lr_norm, tmean, tstd = normalize_patch(tile)
                lr_up   = bicubic_upsample(lr_norm, PATCH_HR)
                lr_t    = torch.from_numpy(lr_up[np.newaxis, np.newaxis]).float().to(device)

                samp = diffusion.sample_ddim(lr_t, n_realizations=n_real,
                                             n_steps=50, add_lr=True)
                samp_np = samp[:, 0].cpu().numpy()
                samp_np = denormalize_patch(samp_np, tmean, tstd)

                hr_r = lr_row * SCALE
                hr_c = lr_col * SCALE
                accum[:, hr_r:hr_r+PATCH_HR, hr_c:hr_c+PATCH_HR] += samp_np * gauss_w
                weights[hr_r:hr_r+PATCH_HR, hr_c:hr_c+PATCH_HR]  += gauss_w

    ws = np.where(weights > 0, weights, 1.0)
    sr_real = accum / ws
    sr_real[:, weights == 0] = np.nan
    sr_mean = np.nanmean(sr_real, axis=0)
    sr_std  = np.nanstd(sr_real,  axis=0)

    # LR bicubic at 2m for comparison (just nearest-neighbour upscale the region)
    lr_2m = np.repeat(np.repeat(lr_region, SCALE, axis=0), SCALE, axis=1).astype(np.float32)

    # HR bounds for this clipped region
    hr_left = t.c + c0 * t.a
    hr_top  = t.f + r0 * t.e
    hr_res  = lr_res / SCALE
    hr_transform = from_origin(hr_left, hr_top, hr_res, hr_res)

    return sr_mean, sr_std, lr_2m, hr_transform, lr_profile['crs']

# ── Load LR grid once ─────────────────────────────────────────────────────
print("Loading LR grid...")
lr_arr, lr_profile = load_lr(LR_TIF)
print(f"LR grid: {lr_arr.shape}")

# ── Process each survey ───────────────────────────────────────────────────
for survey_name, hr_fname in SURVEYS.items():
    print(f"\n=== {survey_name} ===")
    hr_path = os.path.join(DATA_DIR, hr_fname)

    # Get HR bounds
    with rasterio.open(hr_path) as src:
        b = src.bounds
        hr_crs = src.crs

    # Re-project bounds to LR CRS if needed (they should both be UTM)
    bounds = (b.left, b.bottom, b.right, b.top)
    print(f"  HR bounds: {bounds}")

    # Generate SR for this region
    sr_mean, sr_std, lr_2m, hr_transform, crs = generate_sr(
        lr_arr, lr_profile, bounds, N_REAL)

    # Load HR resampled to 2m aligned to same grid
    hr_2m, _ = resample_hr_to_2m(hr_path, crs, bounds, out_res=2.0)

    # Crop to common valid extent (where HR has data)
    valid = np.isfinite(hr_2m)
    rows  = np.where(valid.any(axis=1))[0]
    cols  = np.where(valid.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        print("  No valid HR overlap — skipping")
        continue
    r0v, r1v = rows[0], rows[-1]+1
    c0v, c1v = cols[0], cols[-1]+1

    # Clip all grids to same box (HR coverage)
    # sr_mean/sr_std/lr_2m may be slightly different size — slice to match
    def safe_clip(arr, r0, r1, c0, c1):
        r1 = min(r1, arr.shape[0])
        c1 = min(c1, arr.shape[1])
        return arr[r0:r1, c0:c1]

    hr_c  = safe_clip(hr_2m,   r0v, r1v, c0v, c1v)
    sr_c  = safe_clip(sr_mean, r0v, r1v, c0v, c1v)
    ss_c  = safe_clip(sr_std,  r0v, r1v, c0v, c1v)
    lr_c  = safe_clip(lr_2m,   r0v, r1v, c0v, c1v)

    H, W = hr_c.shape
    print(f"  Cropped valid region: {H}×{W} pixels at 2m")

    # Mask where HR has no data
    mask = ~np.isfinite(hr_c)
    sr_c[mask] = np.nan
    ss_c[mask] = np.nan
    lr_c[mask] = np.nan

    # ── Choose best transect: row with highest HR variance (most features) ──
    hr_var_row = np.nanvar(hr_c, axis=1)
    hr_var_col = np.nanvar(hr_c, axis=0)
    best_row = int(np.nanargmax(hr_var_row))
    best_col = int(np.nanargmax(hr_var_col))
    print(f"  Best transects: row={best_row}, col={best_col}")

    x_m = np.arange(W) * 2.0   # metres
    y_m = np.arange(H) * 2.0

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'v4 residual-mode SR — {survey_name}\n'
                 f'LR input (32m bicubic)  |  SR mean (10 realizations)  |  '
                 f'SR uncertainty (std)  |  HR ground truth (2m)',
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.3,
                           height_ratios=[2.5, 1, 1])

    vp = np.nanpercentile(hr_c, [2, 98])
    sp = np.nanpercentile(ss_c[np.isfinite(ss_c)], [0, 95]) if np.isfinite(ss_c).any() else [0, 1]

    kw = dict(cmap='terrain', vmin=vp[0], vmax=vp[1], interpolation='nearest')

    ax_lr = fig.add_subplot(gs[0, 0]);  im0 = ax_lr.imshow(lr_c, **kw)
    ax_sr = fig.add_subplot(gs[0, 1]);  im1 = ax_sr.imshow(sr_c, **kw)
    ax_un = fig.add_subplot(gs[0, 2]);  im2 = ax_un.imshow(ss_c, cmap='hot_r',
                                                            vmin=sp[0], vmax=sp[1],
                                                            interpolation='nearest')
    ax_hr = fig.add_subplot(gs[0, 3]);  im3 = ax_hr.imshow(hr_c, **kw)

    for ax, title, im in [
        (ax_lr, 'LR bicubic (32m → 2m)', im0),
        (ax_sr, 'SR diffusion mean',      im1),
        (ax_un, 'SR uncertainty (std, m)', im2),
        (ax_hr, 'HR ground truth (2m)',   im3),
    ]:
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02).ax.tick_params(labelsize=6)

    # Mark transect lines on maps
    for ax in [ax_lr, ax_sr, ax_un, ax_hr]:
        ax.axhline(best_row, color='cyan', lw=1.2, ls='--')
        ax.axvline(best_col, color='lime', lw=1.2, ls='--')

    # ── Horizontal profile (row transect) ────────────────────────────────
    ax_hp = fig.add_subplot(gs[1, :])
    p_lr = lr_c[best_row, :]
    p_sr = sr_c[best_row, :]
    p_ss = ss_c[best_row, :]
    p_hr = hr_c[best_row, :]
    valid_prof = np.isfinite(p_hr)
    ax_hp.fill_between(x_m[valid_prof],
                       (p_sr - p_ss)[valid_prof], (p_sr + p_ss)[valid_prof],
                       alpha=0.3, color='steelblue', label='SR ±1σ uncertainty')
    ax_hp.plot(x_m[valid_prof], p_lr[valid_prof], color='orange',    lw=1.8, label='LR bicubic')
    ax_hp.plot(x_m[valid_prof], p_sr[valid_prof], color='steelblue', lw=1.5, label='SR mean')
    ax_hp.plot(x_m[valid_prof], p_hr[valid_prof], color='black',     lw=1.2, ls='--', label='HR truth')
    ax_hp.set_xlabel('W → E distance (m)', fontsize=9)
    ax_hp.set_ylabel('Elevation (m)', fontsize=9)
    rmse_lr_h = np.sqrt(np.nanmean((p_lr - p_hr)**2))
    rmse_sr_h = np.sqrt(np.nanmean((p_sr - p_hr)**2))
    ax_hp.set_title(f'E–W transect (row {best_row}, cyan line) — '
                    f'RMSE: LR={rmse_lr_h:.1f} m   SR={rmse_sr_h:.1f} m', fontsize=9)
    ax_hp.legend(fontsize=8); ax_hp.grid(True, alpha=0.3)
    ax_hp.tick_params(labelsize=8)

    # ── Vertical profile (col transect) ──────────────────────────────────
    ax_vp = fig.add_subplot(gs[2, :])
    q_lr = lr_c[:, best_col]
    q_sr = sr_c[:, best_col]
    q_ss = ss_c[:, best_col]
    q_hr = hr_c[:, best_col]
    valid_profv = np.isfinite(q_hr)
    ax_vp.fill_between(y_m[valid_profv],
                       (q_sr - q_ss)[valid_profv], (q_sr + q_ss)[valid_profv],
                       alpha=0.3, color='steelblue')
    ax_vp.plot(y_m[valid_profv], q_lr[valid_profv], color='orange',    lw=1.8)
    ax_vp.plot(y_m[valid_profv], q_sr[valid_profv], color='steelblue', lw=1.5)
    ax_vp.plot(y_m[valid_profv], q_hr[valid_profv], color='black',     lw=1.2, ls='--')
    ax_vp.set_xlabel('N → S distance (m)', fontsize=9)
    ax_vp.set_ylabel('Elevation (m)', fontsize=9)
    rmse_lr_v = np.sqrt(np.nanmean((q_lr - q_hr)**2))
    rmse_sr_v = np.sqrt(np.nanmean((q_sr - q_hr)**2))
    ax_vp.set_title(f'N–S transect (col {best_col}, lime line) — '
                    f'RMSE: LR={rmse_lr_v:.1f} m   SR={rmse_sr_v:.1f} m', fontsize=9)
    ax_vp.grid(True, alpha=0.3)
    ax_vp.tick_params(labelsize=8)

    out = f'/home/users/scdobbs/GAN_Bathymetry_Project/results/figures_v4/region_{survey_name}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # ── Bonus: calibration plot — does uncertainty predict error? ─────────
    # Bin SR std dev, compute mean |SR - HR| per bin
    err   = np.abs(sr_c - hr_c)
    unc   = ss_c
    ok    = np.isfinite(err) & np.isfinite(unc) & (unc > 0)
    if ok.sum() > 100:
        bins  = np.percentile(unc[ok], np.linspace(0, 100, 21))
        bins  = np.unique(bins)
        idx   = np.digitize(unc[ok], bins) - 1
        idx   = np.clip(idx, 0, len(bins)-2)
        bin_unc = [unc[ok][idx==i].mean() for i in range(len(bins)-1)]
        bin_err = [err[ok][idx==i].mean() for i in range(len(bins)-1)]

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(bin_unc, bin_err, s=60, color='steelblue', edgecolors='navy', zorder=3)
        lim = max(max(bin_unc), max(bin_err)) * 1.1
        ax2.plot([0, lim], [0, lim], 'k--', lw=1.5, label='Perfect calibration')
        ax2.set_xlabel('SR uncertainty (std dev, m)', fontsize=10)
        ax2.set_ylabel('Mean absolute error |SR − HR| (m)', fontsize=10)
        ax2.set_title(f'Uncertainty calibration — {survey_name}\n'
                      'Well-calibrated model: points follow the diagonal', fontsize=10)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        corr = np.corrcoef(bin_unc, bin_err)[0, 1]
        ax2.text(0.05, 0.92, f'Pearson r = {corr:.3f}', transform=ax2.transAxes, fontsize=9)
        out2 = f'/home/users/scdobbs/GAN_Bathymetry_Project/results/figures_v4/calibration_{survey_name}.png'
        plt.savefig(out2, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved calibration: {out2}")

print("\nAll done.")
PYEOF

echo "Finished: $(date)"
