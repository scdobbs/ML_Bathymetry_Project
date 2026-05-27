#!/bin/bash
#SBATCH --job-name=bathy_eval_only
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_only_%j.out
#SBATCH --error=logs/eval_only_%j.err

# ============================================================
# Phase 6 (eval) + Phase 7 (diversity) only.
#
# Use when a prior infer_eval run completed Phases 4 + 5 but timed out
# during Phase 6.  Re-uses the existing OUT_DIR's eval/ subdir.
#
# Env vars (override before sbatch):
#   CKPT           — checkpoint to evaluate
#   OUT_DIR        — existing infer_eval output dir
#   MAX_PATCHES    — Phase 6 cap (default 100)
#   N_REALIZATIONS — Phase 6 realizations (default 10)
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
OUT_DIR="${OUT_DIR:-/scratch/users/${USER}/bathy_diffusion/infer_eval_h100_20260514_144500}"
PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512"
EVAL_DIR="${OUT_DIR}/eval"
MAX_PATCHES="${MAX_PATCHES:-100}"
N_REALIZATIONS="${N_REALIZATIONS:-10}"

mkdir -p "$EVAL_DIR"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "Eval-only resubmit"
echo "  Host          : $(hostname)"
echo "  Checkpoint    : $CKPT"
echo "  OUT_DIR       : $OUT_DIR"
echo "  EVAL_DIR      : $EVAL_DIR"
echo "  MAX_PATCHES   : $MAX_PATCHES"
echo "  N_REALIZATIONS: $N_REALIZATIONS"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Started       : $(date)"
echo "======================================================================"

[ ! -f "$CKPT" ] && { echo "ERROR: checkpoint not found at $CKPT" >&2; exit 1; }
[ ! -d "${PATCHES_DIR}/test" ] && { echo "ERROR: test patches not found at ${PATCHES_DIR}/test" >&2; exit 1; }

# ============================================================
# Phase 6 — Evaluation on test patches
# ============================================================
echo "### Phase 6: evaluation on test patches (max ${MAX_PATCHES} patches)"
python3 -m evaluation.evaluate \
    --checkpoint     "$CKPT" \
    --test_patches   "${PATCHES_DIR}/test" \
    --out_dir        "$EVAL_DIR" \
    --n_realizations "$N_REALIZATIONS" \
    --max_patches    "$MAX_PATCHES" \
    2>&1 | tee "${OUT_DIR}/eval.log"
echo

# ============================================================
# Phase 7 — Sample-diversity check
# ============================================================
echo "### Phase 7: sample-diversity check"
python3 - <<PY 2>&1 | tee "${OUT_DIR}/diversity.log"
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
print(f"Test patches: {len(all_patches)}", flush=True)

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
      f"({PER_SURVEY} per survey)", flush=True)

per_patch = []
for pf in patches:
    d = np.load(pf)
    lr_raw_norm = d['lr_raw'].squeeze()
    hr_norm     = d['hr'].squeeze()
    mean_elev, std_elev = float(d['stats'][0]), float(d['stats'][1])

    lr_t = torch.from_numpy(lr_raw_norm[np.newaxis, np.newaxis]).float().to(device)
    chunks = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
        for _ in range(5):
            ch = flow.sample(lr_t, n_realizations=10, cfg_scale=cfg.cfg_scale)
            chunks.append(ch)
    samples = torch.cat(chunks, dim=0)
    samples_np = samples.squeeze(1).cpu().float().numpy()
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
    print(f"  {os.path.basename(pf)}: gt_range={gt_range:.2f}m  med_std={per_patch[-1]['std_median_m']:.3f}m", flush=True)

print()
print("Per-patch inter-realization std (meters)")
print("=" * 100)
print(f"{'patch':<50s} {'gt_range':>9s} {'med_std':>9s} {'mean_std':>9s} {'p95_std':>9s} {'frac_>1m':>9s}")
print("-" * 100)
for p in per_patch:
    print(f"{p['name']:<50s} {p['gt_range_m']:>9.2f} {p['std_median_m']:>9.3f} "
          f"{p['std_mean_m']:>9.3f} {p['std_p95_m']:>9.3f} {p['frac_pixels_std_gt_1m']:>9.3f}")

med_med = float(np.median([p['std_median_m'] for p in per_patch]))
print(f"\nMedian-of-medians inter-realization std: {med_med:.3f} m")
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
echo "Outputs at: $EVAL_DIR"
