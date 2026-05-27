#!/bin/bash
#SBATCH --job-name=bathy_real_cmp
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/real_cmp_%j.out
#SBATCH --error=logs/real_cmp_%j.err

# ============================================================
# Realization comparison: visualize 8 stochastic SR realizations
# of one representative test patch from each survey.
#
# Diagnostic question — do all realizations show similar
# implausibilities in the same places (architectural / prior bias),
# or do each show different implausibilities (data / coverage issue)?
# ============================================================

mkdir -p logs
type conda >/dev/null 2>&1 && {
    conda deactivate >/dev/null 2>&1 || true
    conda deactivate >/dev/null 2>&1 || true
}
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
CKPT="${CKPT:-/scratch/users/${USER}/bathy_diffusion/training_h100/checkpoints/ckpt_final.pt}"
BASE_CH="${BASE_CH:-64}"             # MUST match the base_ch the checkpoint was trained with
RESIDUAL="${RESIDUAL:-0}"            # 1 if checkpoint was trained in residual_mode
PATCHES_DIR="${PATCHES_DIR:-/scratch/groups/hilley/bathy_diffusion_2026/patches_512/test}"
OUT_DIR="/scratch/users/${USER}/bathy_diffusion/realization_comparison_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"

echo "==============================================="
echo "Realization comparison"
echo "  Checkpoint  : $CKPT"
echo "  base_ch     : $BASE_CH"
echo "  residual    : $RESIDUAL"
echo "  Patches     : $PATCHES_DIR"
echo "  Out dir     : $OUT_DIR"
echo "  Host       : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Started    : $(date)"
echo "==============================================="

python3 - <<PY 2>&1 | tee "${OUT_DIR}/comparison.log"
import os, sys, glob, re
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

torch.manual_seed(0)
np.random.seed(0)

sys.path.insert(0, '${PROJECT_DIR}')
from models.diffusion import build_rectified_flow_from_checkpoint
from preprocessing.data_utils import denormalize_patch
from training.config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = Config()
cfg.base_ch = ${BASE_CH}                # override to match the training base_ch
cfg.residual_mode = bool(${RESIDUAL})   # override to match training-time residual mode
print(f"Device: {device}  base_ch={cfg.base_ch}  residual_mode={cfg.residual_mode}", flush=True)
flow = build_rectified_flow_from_checkpoint('${CKPT}', cfg, device)

# Pick one moderately complex patch from each test survey
all_patches = sorted(glob.glob('${PATCHES_DIR}/*.npz'))
by_survey = defaultdict(list)
for pf in all_patches:
    survey = re.sub(r'_r\d{5}_c\d{5}\.npz', '', os.path.basename(pf))
    by_survey[survey].append(pf)

# For each survey: pick the patch whose HR has the median dynamic range
# (avoids both flat / outlier patches).
selected = []
for survey, plist in by_survey.items():
    ranges = []
    for pf in plist:
        d = np.load(pf)
        hr = d['hr'].squeeze() * float(d['stats'][1]) + float(d['stats'][0])
        ranges.append((float(hr.max() - hr.min()), pf))
    ranges.sort()
    median_idx = len(ranges) // 2
    selected.append(ranges[median_idx][1])

print(f"Selected {len(selected)} patches (one per survey, median dynamic range):", flush=True)
for pf in selected:
    print(f"  {os.path.basename(pf)}", flush=True)

# Generate 8 realizations per patch
N_REAL = 8

for pf in selected:
    d = np.load(pf)
    lr_raw_norm = d['lr_raw'].squeeze()       # (32, 32) normalized
    lr_up_norm  = d['lr_up'].squeeze()        # (512, 512) bicubic-upsampled, normalized
    hr_norm     = d['hr'].squeeze()           # (512, 512) normalized
    stats       = d['stats']
    mean_m, std_m = float(stats[0]), float(stats[1])

    survey = re.sub(r'_r\d{5}_c\d{5}\.npz', '', os.path.basename(pf))
    print(f"\n=== {os.path.basename(pf)} ===", flush=True)
    print(f"  HR dynamic range: {(hr_norm.max()-hr_norm.min())*std_m:.2f} m", flush=True)

    lr_t = torch.from_numpy(lr_raw_norm[np.newaxis, np.newaxis]).float().to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
        samples = flow.sample(lr_t, n_realizations=N_REAL, cfg_scale=cfg.cfg_scale)
    samples_np = samples.squeeze(1).cpu().float().numpy()   # (8, 512, 512) normalized
    samples_m  = denormalize_patch(samples_np, mean_m, std_m)
    hr_m       = denormalize_patch(hr_norm, mean_m, std_m)
    lr_up_m    = denormalize_patch(lr_up_norm, mean_m, std_m)

    real_mean = samples_m.mean(axis=0)
    real_std  = samples_m.std(axis=0)

    # Use HR range for color scale so everything is comparable per-patch
    vmin, vmax = hr_m.min(), hr_m.max()
    std_max = float(real_std.max())

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f'{survey}\n{os.path.basename(pf)}  |  HR range = {vmax-vmin:.1f} m',
        fontsize=10, y=0.995)

    def show(ax, arr, title, **kw):
        im = ax.imshow(arr, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        return im

    # Row 1: HR ground truth, LR bicubic, mean of realizations, std map
    im0 = show(axes[0,0], hr_m,        f'HR ground truth',          cmap='terrain', vmin=vmin, vmax=vmax)
    im1 = show(axes[0,1], lr_up_m,     f'LR bicubic',               cmap='terrain', vmin=vmin, vmax=vmax)
    im2 = show(axes[0,2], real_mean,   f'Realization mean (n={N_REAL})', cmap='terrain', vmin=vmin, vmax=vmax)
    im3 = show(axes[0,3], real_std,    f'Realization std (max={std_max:.2f} m)', cmap='hot_r', vmin=0, vmax=std_max)

    plt.colorbar(im0, ax=axes[0,0], shrink=0.7).ax.tick_params(labelsize=7)
    plt.colorbar(im2, ax=axes[0,2], shrink=0.7).ax.tick_params(labelsize=7)
    plt.colorbar(im3, ax=axes[0,3], shrink=0.7).ax.tick_params(labelsize=7)

    # Rows 2-3: individual realizations 1-8
    for i in range(N_REAL):
        r, c = 1 + i // 4, i % 4
        show(axes[r, c], samples_m[i], f'Realization {i+1}', cmap='terrain', vmin=vmin, vmax=vmax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join('${OUT_DIR}', f'{survey}_realizations.png')
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}", flush=True)

    # Also save a "difference between realizations" plot to highlight
    # where realizations disagree most.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        f'{survey} — Per-realization residual vs realization mean (m)\n'
        f'(highlights where this realization deviates from the consensus)',
        fontsize=10, y=0.99)
    diff_max = float(np.abs(samples_m - real_mean).max())
    for i in range(N_REAL):
        r, c = i // 4, i % 4
        diff = samples_m[i] - real_mean
        im = axes[r, c].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[r, c].set_title(f'Realization {i+1} − mean', fontsize=9)
        axes[r, c].axis('off')
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='deviation (m)')
    out_png2 = os.path.join('${OUT_DIR}', f'{survey}_deviations.png')
    plt.savefig(out_png2, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png2}", flush=True)

print("\nDone.", flush=True)
PY

echo
echo "Done: $(date)"
echo "Outputs at: $OUT_DIR"
