#!/bin/bash
#SBATCH --job-name=bathy_post_e2e
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:A100_SXM4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/post_e2e_%j.out
#SBATCH --error=logs/post_e2e_%j.err

# Standalone Phase 7 (diversity) + standalone calibration plot from
# an existing e2e checkpoint. The previous 12hr A100 e2e run finished
# Phase 6's eval loop but timed out before _plot_calibration completed
# (suspect: nanpercentile on a 10 × 5.24M-element array).

mkdir -p logs
echo "BEGIN script: $(date)"
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

type conda >/dev/null 2>&1 && { conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; }
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
E2E_DIR="${E2E_DIR:-/scratch/users/scdobbs/bathy_diffusion/e2e_20260512_185142}"
CKPT="${E2E_DIR}/training/checkpoints/ckpt_final.pt"
PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512"
EVAL_DIR="${E2E_DIR}/eval"

cd "$PROJECT_DIR"
echo "E2E_DIR : $E2E_DIR"
echo "CKPT    : $CKPT"
echo

# ============================================================
# Part A — Calibration plot from a small subsample (fast)
# ============================================================
if [ -f "${EVAL_DIR}/calibration.png" ]; then
    echo "### Part A: calibration plot — SKIP (already exists at ${EVAL_DIR}/calibration.png)"
else
echo "### Part A: calibration plot (20 patches, 10 realizations each)"
python3 - <<PY 2>&1 | tee "${E2E_DIR}/post_calibration.log"
import os, sys, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
sys.path.insert(0, '${PROJECT_DIR}')
from models.diffusion import build_rectified_flow_from_checkpoint
from preprocessing.data_utils import denormalize_patch
from training.config import Config
from evaluation.metrics import calibration_data

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
flow = build_rectified_flow_from_checkpoint('${CKPT}', cfg, device)

patch_files = sorted(glob.glob('${PATCHES_DIR}/test/*.npz'))[:20]
print(f"Using {len(patch_files)} test patches × 10 realizations for calibration")

all_samples = []
all_targets = []
for pf in patch_files:
    d = np.load(pf)
    lr = torch.from_numpy(d['lr_raw'][np.newaxis]).float().to(device)
    hr_norm = d['hr'].squeeze()
    stats = d['stats']
    mean_m, std_m = float(stats[0]), float(stats[1])

    with torch.no_grad(), torch.autocast(device_type=device.type,
                                          dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
        samples = flow.sample(lr, n_realizations=10, cfg_scale=cfg.cfg_scale)
    samples_np = samples.squeeze(1).cpu().float().numpy()
    all_samples.append(denormalize_patch(samples_np, mean_m, std_m))
    all_targets.append(denormalize_patch(hr_norm,    mean_m, std_m))
    print(f"  {os.path.basename(pf)}: samples {samples_np.shape}", flush=True)

# Flatten and concatenate — keep realization axis intact
samples_cat = np.concatenate(
    [s.reshape(s.shape[0], -1) for s in all_samples], axis=1)
targets_cat = np.concatenate([t.ravel() for t in all_targets])
print(f"\nshape: samples_cat={samples_cat.shape}  targets_cat={targets_cat.shape}")

# Estimate the cost of the original nanpercentile call
import time
t0 = time.time()
nominal, empirical = calibration_data(samples_cat, targets_cat, n_bins=20)
print(f"calibration_data() took {time.time() - t0:.1f}s")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(nominal, empirical, 'o-', color='steelblue', label='Model')
ax.fill_between(nominal, nominal - 0.05, nominal + 0.05,
                alpha=0.15, color='gray', label='±5% band')
ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Empirical coverage')
ax.set_title('Calibration plot (smoke run — 200 training steps)')
ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); plt.tight_layout()
out = os.path.join('${EVAL_DIR}', 'calibration.png')
plt.savefig(out, dpi=150); plt.close()
print(f"Saved: {out}")
print(f"Nominal vs empirical (first 5): {list(zip(nominal[:5], empirical[:5]))}")
PY
fi
echo

# ============================================================
# Part B — Phase 7 diversity check
# ============================================================
echo "### Part B: sample-diversity check (stratified across test surveys)"
python3 - <<PY 2>&1 | tee "${E2E_DIR}/diversity.log"
import os, sys, glob, json, re, random
from collections import defaultdict
import numpy as np
import torch

torch.manual_seed(0)
random.seed(0)
sys.path.insert(0, '${PROJECT_DIR}')
from models.diffusion import build_rectified_flow_from_checkpoint
from preprocessing.data_utils import denormalize_patch
from training.config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = Config()
flow = build_rectified_flow_from_checkpoint('${CKPT}', cfg, device)

all_patches = sorted(glob.glob('${PATCHES_DIR}/test/*.npz'))
print(f"Test patches: {len(all_patches)}")

by_survey = defaultdict(list)
for pf in all_patches:
    survey = re.sub(r'_r\d{5}_c\d{5}\.npz', '', os.path.basename(pf))
    by_survey[survey].append(pf)

PER_SURVEY = max(1, 20 // max(1, len(by_survey)))
patches = []
for survey, plist in by_survey.items():
    random.shuffle(plist)
    patches.extend(plist[:PER_SURVEY])
print(f"Stratified sample: {len(patches)} patches across {len(by_survey)} surveys "
      f"({PER_SURVEY} per survey)")

per_patch = []
for pf in patches:
    d = np.load(pf)
    lr_raw_norm = d['lr_raw'].squeeze()
    hr_norm     = d['hr'].squeeze()
    mean_elev, std_elev = float(d['stats'][0]), float(d['stats'][1])

    lr_t = torch.from_numpy(lr_raw_norm[np.newaxis, np.newaxis]).float().to(device)
    # Chunk 50 realizations into 5×10 to fit A100 40GB. CFG doubles the
    # effective batch (cond + uncond), so 50 reals × 2 × 512² OOMs the model.
    chunks = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
        for _ in range(5):
            ch = flow.sample(lr_t, n_realizations=10, cfg_scale=cfg.cfg_scale)
            chunks.append(ch)
    samples = torch.cat(chunks, dim=0)
    samples_np = samples.squeeze(1).cpu().float().numpy()
    print(f"  {os.path.basename(pf)}: samples {samples_np.shape}", flush=True)
    samples_m = denormalize_patch(samples_np, mean_elev, std_elev)
    hr_m = denormalize_patch(hr_norm, mean_elev, std_elev)

    per_pixel_std = samples_m.std(axis=0)
    gt_range = float(hr_m.max() - hr_m.min())
    per_patch.append({
        'name': os.path.basename(pf),
        'gt_range_m':            gt_range,
        'std_min_m':             float(np.nanmin(per_pixel_std)),
        'std_median_m':          float(np.nanmedian(per_pixel_std)),
        'std_mean_m':            float(np.nanmean(per_pixel_std)),
        'std_p95_m':             float(np.nanpercentile(per_pixel_std, 95)),
        'std_max_m':             float(np.nanmax(per_pixel_std)),
        'frac_pixels_std_gt_1m': float((per_pixel_std > 1.0).mean()),
    })

print()
print("Per-patch inter-realization std (meters) — sample-diversity check")
print("=" * 100)
print(f"{'patch':<50s} {'gt_range':>9s} {'med_std':>9s} {'mean_std':>9s} {'p95_std':>9s} {'frac_>1m':>9s}")
print("-" * 100)
for p in per_patch:
    print(f"{p['name']:<50s} {p['gt_range_m']:>9.2f} {p['std_median_m']:>9.3f} "
          f"{p['std_mean_m']:>9.3f} {p['std_p95_m']:>9.3f} {p['frac_pixels_std_gt_1m']:>9.3f}")

med_med = float(np.median([p['std_median_m'] for p in per_patch]))
print()
print(f"Median-of-medians inter-realization std: {med_med:.3f} m")
print()
if med_med < 0.1:
    print("WARNING: median std < 0.1 m — possible mode collapse.")
elif med_med < 1.0:
    print("Note: median std < 1 m. Acceptable for low-relief regions.")
else:
    print("OK: meaningful inter-realization spread.")

with open('${EVAL_DIR}/diversity.json', 'w') as f:
    json.dump({'per_patch': per_patch, 'median_of_medians_std_m': med_med}, f, indent=2)
print(f"Saved: ${EVAL_DIR}/diversity.json")
PY

echo
echo "Done: $(date)"
