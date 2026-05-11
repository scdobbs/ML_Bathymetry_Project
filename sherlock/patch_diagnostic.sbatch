#!/bin/bash
#SBATCH --job-name=bathy_diag_v3
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/diag_v3_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/diag_v3_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
cd "$PROJ_DIR"
echo "Started: $(date)"

python3 - <<'PYEOF'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, random, os, sys
sys.path.insert(0, '/home/users/scdobbs/GAN_Bathymetry_Project')
sys.path.insert(0, '/home/users/scdobbs/GAN_Bathymetry_Project/preprocessing')

import torch
from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from training.config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── Load model (same as generate_realizations.py) ─────────────────────────
cfg = Config()
cfg.base_ch = 64
ckpt_path = '/scratch/users/scdobbs/bathy_diffusion_v3/checkpoints/ckpt_final.pt'
state = torch.load(ckpt_path, map_location=device, weights_only=False)

unet = BathyUNet(base_ch=cfg.base_ch).to(device)
diffusion = GaussianDiffusion(unet, n_timesteps=1000).to(device)
ema_state = state.get('ema', state.get('model'))
diffusion.load_state_dict(ema_state)
diffusion.eval()
print("Model loaded.")

# ── Load val patches ───────────────────────────────────────────────────────
val_dirs = [
    '/scratch/users/scdobbs/bathy_diffusion/patches/val',
    '/scratch/users/scdobbs/bathy_diffusion/patches_v3/val',
]
all_val = []
for d in val_dirs:
    all_val += glob.glob(os.path.join(d, '*.npz'))
random.seed(42)
selected = random.sample(all_val, 64)
print(f"Running inference on {len(selected)} val patches...")

lr_patches, sr_patches, hr_patches = [], [], []
with torch.no_grad():
    for i, fpath in enumerate(selected):
        d = np.load(fpath)
        lr = torch.from_numpy(d['lr']).float().unsqueeze(0).to(device)
        hr = d['hr'][0]
        sr = diffusion.sample_ddim(lr, n_steps=50)
        sr_np = sr[0, 0].cpu().numpy()
        lr_patches.append(lr[0, 0].cpu().numpy())
        sr_patches.append(sr_np)
        hr_patches.append(hr)
        if (i+1) % 16 == 0:
            print(f"  {i+1}/{len(selected)}")

lr_patches = np.array(lr_patches)
sr_patches = np.array(sr_patches)
hr_patches = np.array(hr_patches)

# ── Radial power spectrum ──────────────────────────────────────────────────
def radial_psd(patches):
    N = patches.shape[-1]
    psds = []
    for p in patches:
        f = np.fft.fft2(p - p.mean())
        psd = np.abs(np.fft.fftshift(f))**2
        psds.append(psd)
    mean_psd = np.mean(psds, axis=0)
    cy, cx = N//2, N//2
    y, x = np.indices(mean_psd.shape)
    r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int)
    radial = np.bincount(r.ravel(), mean_psd.ravel()) / np.bincount(r.ravel())
    freqs = np.arange(len(radial)) / N
    return freqs, radial

freqs, psd_lr = radial_psd(lr_patches)
freqs, psd_sr = radial_psd(sr_patches)
freqs, psd_hr = radial_psd(hr_patches)

rmse_lr = np.sqrt(np.mean((lr_patches - hr_patches)**2, axis=(1,2)))
rmse_sr = np.sqrt(np.mean((sr_patches - hr_patches)**2, axis=(1,2)))
print(f"Mean RMSE — LR: {rmse_lr.mean():.4f}σ  SR: {rmse_sr.mean():.4f}σ")
print(f"SR better than LR: {(rmse_sr < rmse_lr).mean()*100:.1f}% of patches")

# ── Figure 1: Power spectrum + RMSE scatter ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Spatial Frequency Diagnostic — v3 model (64 val patches)', fontsize=12, fontweight='bold')

ax = axes[0]
mask = (freqs > 0) & (freqs < 0.45)
ax.semilogy(freqs[mask], psd_lr[mask], color='orange', lw=2.5, label='LR bicubic (input)')
ax.semilogy(freqs[mask], psd_sr[mask], color='steelblue', lw=2.5, label='SR diffusion (output)')
ax.semilogy(freqs[mask], psd_hr[mask], color='black', lw=2.5, label='HR ground truth')
ax.set_xlabel('Spatial frequency (cycles/pixel)', fontsize=10)
ax.set_ylabel('Power spectral density', fontsize=10)
ax.legend(fontsize=10)
ax.set_title('Radially-averaged 2D power spectrum\nIf SR≈LR → model adds no HF detail', fontsize=10)
ax.grid(True, alpha=0.3)
ax2 = ax.twiny()
tick_wl = [128, 64, 32, 16, 8, 4]
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([2.0/w for w in tick_wl])
ax2.set_xticklabels([f'{w}m' for w in tick_wl], fontsize=8)
ax2.set_xlabel('Wavelength (metres at 2m/px)', fontsize=9)

ax = axes[1]
lim = max(rmse_lr.max(), rmse_sr.max()) * 1.05
ax.scatter(rmse_lr, rmse_sr, alpha=0.6, color='steelblue', s=50, edgecolors='navy', lw=0.5)
ax.plot([0, lim], [0, lim], 'k--', lw=1.5, label='SR = LR (no improvement)')
ax.axhline(rmse_sr.mean(), color='steelblue', lw=1, ls=':', label=f'SR mean={rmse_sr.mean():.3f}σ')
ax.axvline(rmse_lr.mean(), color='orange', lw=1, ls=':', label=f'LR mean={rmse_lr.mean():.3f}σ')
ax.set_xlabel('LR bicubic RMSE vs HR (σ)', fontsize=10)
ax.set_ylabel('SR diffusion RMSE vs HR (σ)', fontsize=10)
ax.set_title(f'Per-patch RMSE — SR better in {(rmse_sr<rmse_lr).mean()*100:.0f}% of patches', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

plt.tight_layout()
out1 = '/home/users/scdobbs/GAN_Bathymetry_Project/results/figures_v3/psd_diagnostic.png'
plt.savefig(out1, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: 8 example patches LR / SR / HR ──────────────────────────────
ex_idx = random.sample(range(len(selected)), 8)
fig, axes = plt.subplots(3, 8, figsize=(20, 8))
fig.suptitle('Individual patch examples — LR input / SR output / HR truth\n(normalized units; same colorscale per column)',
             fontsize=11, fontweight='bold')
row_labels = ['LR bicubic\n(input)', 'SR diffusion\n(output)', 'HR ground truth']
for col, i in enumerate(ex_idx):
    vmin = min(lr_patches[i].min(), hr_patches[i].min())
    vmax = max(lr_patches[i].max(), hr_patches[i].max())
    kw = dict(cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0,col].imshow(lr_patches[i], **kw)
    axes[1,col].imshow(sr_patches[i], **kw)
    im = axes[2,col].imshow(hr_patches[i], **kw)
    axes[0,col].set_title(f'RMSE_LR\n={rmse_lr[i]:.3f}σ', fontsize=7)
    axes[1,col].set_title(f'RMSE_SR\n={rmse_sr[i]:.3f}σ', fontsize=7)
    for r in range(3):
        axes[r,col].axis('off')
    plt.colorbar(im, ax=axes[:,col], shrink=0.5, pad=0.02, fraction=0.05).ax.tick_params(labelsize=5)
for r, lbl in enumerate(row_labels):
    axes[r,0].set_ylabel(lbl, fontsize=8, rotation=90, labelpad=40, va='center')

plt.tight_layout()
out2 = '/home/users/scdobbs/GAN_Bathymetry_Project/results/figures_v3/patch_examples_v3.png'
plt.savefig(out2, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")
PYEOF

echo "Done: $(date)"
