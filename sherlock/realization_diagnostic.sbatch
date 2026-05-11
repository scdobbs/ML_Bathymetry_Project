#!/bin/bash
#SBATCH --job-name=bathy_real_diag
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/real_diag_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/real_diag_%j.err

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

import torch
from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from training.config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load model
cfg = Config()
cfg.base_ch = 64
ckpt_path = '/scratch/users/scdobbs/bathy_diffusion_v3/checkpoints/ckpt_final.pt'
state = torch.load(ckpt_path, map_location=device, weights_only=False)
unet = BathyUNet(base_ch=cfg.base_ch).to(device)
diffusion = GaussianDiffusion(unet, n_timesteps=1000).to(device)
diffusion.load_state_dict(state.get('ema', state.get('model')))
diffusion.eval()
print("Model loaded.")

# Pick 4 val patches with non-trivial HR content
val_files = glob.glob('/scratch/users/scdobbs/bathy_diffusion/patches/val/*.npz')
random.seed(99)
random.shuffle(val_files)

# Select patches with reasonable terrain variation
selected = []
for fpath in val_files:
    d = np.load(fpath)
    hr = d['hr'][0]
    if hr.std() > 0.3:  # skip nearly-flat patches
        selected.append(fpath)
    if len(selected) == 4:
        break

print(f"Selected {len(selected)} patches with terrain variation")
N_REAL = 8  # realizations per patch

# ── Figure 1: individual realizations vs LR / HR ──────────────────────────
# Layout: 4 patches × (LR, real1..8, mean, HR) = 11 columns
ncols = N_REAL + 3  # LR | r1..r8 | mean | HR
nrows = 4

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.5))
fig.suptitle(
    'Individual SR realizations vs LR input / HR ground truth\n'
    'Key question: do realizations show different fine-scale textures?',
    fontsize=11, fontweight='bold'
)

col_labels = ['LR\n(input)'] + [f'Real.\n#{i+1}' for i in range(N_REAL)] + ['SR\nmean', 'HR\n(truth)']
for col, lbl in enumerate(col_labels):
    axes[0, col].set_title(lbl, fontsize=7, pad=3)

for row, fpath in enumerate(selected):
    d = np.load(fpath)
    lr_np = d['lr'][0]
    hr_np = d['hr'][0]
    lr_t = torch.from_numpy(d['lr']).float().unsqueeze(0).to(device)

    with torch.no_grad():
        realizations = []
        for _ in range(N_REAL):
            sr = diffusion.sample_ddim(lr_t, n_steps=50)
            realizations.append(sr[0, 0].cpu().numpy())

    sr_mean = np.mean(realizations, axis=0)
    sr_std  = np.std(realizations, axis=0)

    all_vals = [lr_np] + realizations + [sr_mean, hr_np]
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    kw = dict(cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')

    # LR
    axes[row, 0].imshow(lr_np, **kw)
    axes[row, 0].axis('off')

    # Realizations
    for i, real in enumerate(realizations):
        axes[row, i+1].imshow(real, **kw)
        axes[row, i+1].axis('off')

    # SR mean
    axes[row, N_REAL+1].imshow(sr_mean, **kw)
    axes[row, N_REAL+1].axis('off')

    # HR
    im = axes[row, N_REAL+2].imshow(hr_np, **kw)
    axes[row, N_REAL+2].axis('off')

    # Colorbar on rightmost column
    plt.colorbar(im, ax=axes[row, :], shrink=0.7, pad=0.01, fraction=0.02).ax.tick_params(labelsize=5)

    print(f"  Patch {row+1}: SR std across realizations = {sr_std.mean():.4f}σ  "
          f"(if ~0 → model is deterministic; if >0.1 → genuine stochasticity)")

plt.tight_layout()
out1 = '/home/users/scdobbs/GAN_Bathymetry_Project/results/figures_v3/realizations_diagnostic.png'
plt.savefig(out1, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: zoom into one patch — difference maps ───────────────────────
d = np.load(selected[0])
lr_np = d['lr'][0]
hr_np = d['hr'][0]
lr_t  = torch.from_numpy(d['lr']).float().unsqueeze(0).to(device)

with torch.no_grad():
    reals = [diffusion.sample_ddim(lr_t, n_steps=50)[0,0].cpu().numpy() for _ in range(4)]

sr_mean = np.mean(reals, axis=0)
sr_std  = np.std(reals,  axis=0)

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('Zoom: patch 1 — realizations, differences from LR, and uncertainty\n'
             'Difference = SR_real - LR_bicubic  (non-zero → model adds structure beyond LR)',
             fontsize=10, fontweight='bold')

vmin, vmax = min(lr_np.min(), hr_np.min()), max(lr_np.max(), hr_np.max())
kw = dict(cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')

for i, real in enumerate(reals):
    axes[0, i].imshow(real, **kw)
    axes[0, i].set_title(f'Realization #{i+1}', fontsize=9)
    axes[0, i].axis('off')

    diff = real - lr_np
    dlim = max(abs(diff.min()), abs(diff.max()))
    axes[1, i].imshow(diff, cmap='RdBu_r', vmin=-dlim, vmax=dlim, interpolation='nearest')
    axes[1, i].set_title(f'Diff from LR\n(max={dlim:.3f}σ)', fontsize=9)
    axes[1, i].axis('off')

# Bottom row: LR, SR mean, SR std, HR
for i, (img, title, cmap) in enumerate([
    (lr_np,   'LR bicubic',   'RdBu_r'),
    (sr_mean, 'SR mean',      'RdBu_r'),
    (sr_std,  'SR std\n(uncertainty)', 'hot'),
    (hr_np,   'HR truth',     'RdBu_r'),
]):
    vv = dict(cmap=cmap, vmin=vmin if cmap=='RdBu_r' else 0,
              vmax=vmax if cmap=='RdBu_r' else sr_std.max(), interpolation='nearest')
    im = axes[2, i].imshow(img, **vv)
    axes[2, i].set_title(title, fontsize=9)
    axes[2, i].axis('off')
    plt.colorbar(im, ax=axes[2, i], shrink=0.7, fraction=0.08)

plt.tight_layout()
out2 = '/home/users/scdobbs/GAN_Bathymetry_Project/results/figures_v3/realization_zoom.png'
plt.savefig(out2, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")
PYEOF

echo "Done: $(date)"
