#!/bin/bash
#SBATCH --job-name=profile_v4
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/profile_v4_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/profile_v4_%j.err

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
import glob, random, os, sys
sys.path.insert(0, '/home/users/scdobbs/GAN_Bathymetry_Project')

import torch
from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from training.config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── Load v4 model ─────────────────────────────────────────────────────────
cfg = Config()
cfg.base_ch = 64
ckpt_path = '/scratch/users/scdobbs/bathy_diffusion_v4/checkpoints/ckpt_final.pt'
state = torch.load(ckpt_path, map_location=device, weights_only=False)
unet = BathyUNet(base_ch=cfg.base_ch).to(device)
diffusion = GaussianDiffusion(unet, n_timesteps=1000).to(device)
diffusion.load_state_dict(state.get('ema', state.get('model')))
diffusion.eval()
print("Model loaded.")

# ── Pick 6 diverse val patches ─────────────────────────────────────────────
val_dirs = [
    '/scratch/users/scdobbs/bathy_diffusion/patches/val',
    '/scratch/users/scdobbs/bathy_diffusion/patches_v3/val',
]
all_val = []
for d in val_dirs:
    all_val += glob.glob(os.path.join(d, '*.npz'))

random.seed(99)
selected = random.sample(all_val, 6)
print(f"Selected {len(selected)} patches for profile figure")

# ── Generate 5 SR realizations per patch for uncertainty bands ─────────────
lr_list, sr_mean_list, sr_std_list, hr_list = [], [], [], []
with torch.no_grad():
    for fpath in selected:
        d = np.load(fpath)
        lr = torch.from_numpy(d['lr']).float().unsqueeze(0).to(device)
        hr = d['hr'][0]
        stats = d['stats']  # [mean, std] for denormalization

        # 5 realizations to get spread
        samples = diffusion.sample_ddim(lr, n_realizations=5, n_steps=50, add_lr=True)
        samples_np = samples[:, 0].cpu().numpy()  # (5, H, W)

        # Denormalize to metres
        patch_mean = float(stats[0])
        patch_std  = float(stats[1])
        lr_np   = lr[0, 0].cpu().numpy() * patch_std + patch_mean
        sr_np   = samples_np * patch_std + patch_mean
        hr_np   = hr * patch_std + patch_mean

        lr_list.append(lr_np)
        sr_mean_list.append(sr_np.mean(axis=0))
        sr_std_list.append(sr_np.std(axis=0))
        hr_list.append(hr_np)

# ── Figure: 2 rows × 6 cols — top: 2D map, bottom: horizontal profile ──────
fig = plt.figure(figsize=(22, 10))
fig.suptitle('v4 (residual learning) — Topographic profiles at 2 m resolution\n'
             'Top: 2D maps  |  Bottom: horizontal transect through patch centre',
             fontsize=12, fontweight='bold')

H = lr_list[0].shape[0]
W = lr_list[0].shape[1]
x_hr = np.arange(W) * 2  # metres at 2 m/px
x_lr = np.linspace(0, (W-1)*2, lr_list[0].shape[1])  # LR already upsampled to same size

mid_row = H // 2

for col, (lr, sr_mean, sr_std, hr) in enumerate(
        zip(lr_list, sr_mean_list, sr_std_list, hr_list)):

    # ── 2D map (top row) ──────────────────────────────────────────────────
    ax_map = fig.add_subplot(2, 6, col + 1)
    vmin, vmax = np.percentile(hr, [2, 98])
    # Composite: left third = LR, middle third = SR, right third = HR
    composite = np.concatenate([
        lr[:, :W//3],
        sr_mean[:, W//3:2*W//3],
        hr[:, 2*W//3:]
    ], axis=1)
    im = ax_map.imshow(composite, cmap='terrain', vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')
    ax_map.axhline(mid_row, color='red', lw=1.0, ls='--', alpha=0.7)
    ax_map.axvline(W//3,   color='white', lw=0.8, alpha=0.5)
    ax_map.axvline(2*W//3, color='white', lw=0.8, alpha=0.5)
    if col == 0:
        ax_map.set_title('LR  |  SR  |  HR\n(composite thirds)', fontsize=7)
    else:
        ax_map.set_title('LR | SR | HR', fontsize=7)
    ax_map.axis('off')
    ax_map.text(W//6,      H-4, 'LR', ha='center', fontsize=6, color='white', fontweight='bold')
    ax_map.text(W//2,      H-4, 'SR', ha='center', fontsize=6, color='white', fontweight='bold')
    ax_map.text(5*W//6,    H-4, 'HR', ha='center', fontsize=6, color='white', fontweight='bold')
    plt.colorbar(im, ax=ax_map, shrink=0.6, pad=0.02,
                 label='Elev (m)').ax.tick_params(labelsize=5)

    # ── Transect (bottom row) ──────────────────────────────────────────────
    ax_prof = fig.add_subplot(2, 6, col + 7)
    lr_prof   = lr[mid_row, :]
    sr_prof   = sr_mean[mid_row, :]
    sr_s_prof = sr_std[mid_row, :]
    hr_prof   = hr[mid_row, :]

    ax_prof.fill_between(x_hr, sr_prof - sr_s_prof, sr_prof + sr_s_prof,
                         alpha=0.25, color='steelblue', label='SR ±1σ')
    ax_prof.plot(x_hr, lr_prof,   color='orange',    lw=1.8, label='LR bicubic', zorder=3)
    ax_prof.plot(x_hr, sr_prof,   color='steelblue', lw=1.5, label='SR mean',    zorder=4)
    ax_prof.plot(x_hr, hr_prof,   color='black',     lw=1.2, label='HR truth',   zorder=5,
                 ls='--')

    ax_prof.set_xlabel('Along-transect distance (m)', fontsize=7)
    ax_prof.set_ylabel('Elevation (m)', fontsize=7)
    ax_prof.tick_params(labelsize=6)
    ax_prof.grid(True, alpha=0.3)
    if col == 0:
        ax_prof.legend(fontsize=6, loc='upper right')
    rmse_lr_p = np.sqrt(np.mean((lr_prof - hr_prof)**2))
    rmse_sr_p = np.sqrt(np.mean((sr_prof - hr_prof)**2))
    ax_prof.set_title(f'RMSE  LR={rmse_lr_p:.2f}m  SR={rmse_sr_p:.2f}m', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = '/home/users/scdobbs/GAN_Bathymetry_Project/results/figures_v4/profile_comparison_v4.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print("Done.")
PYEOF

echo "Done: $(date)"
