#!/bin/bash
#SBATCH --job-name=bathy_e2e
#SBATCH -p serc
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/e2e_%j.out
#SBATCH --error=logs/e2e_%j.err

# ============================================================
# End-to-end smoke test for the post-rewrite pipeline.
#
# Phases:
#   1. Preprocess (skipped if patches cache exists).
#   2. Three-way precision benchmark: bf16+compile / bf16-only / fp32.
#      Each runs MAX_BENCH_STEPS optimizer steps; reports steps/sec
#      and peak GPU memory.
#   3. Real training: N_TRAIN_STEPS optimizer steps with bf16+compile.
#   4. Inference on a small bounding box of the LR raster.
#   5. Uncertainty maps from those realizations.
#   6. Evaluation on held-out test patches.
#   7. Sample-diversity check (the metric Steve cares about): print
#      mean / median / quantiles of the inter-realization std on the
#      held-out test patches.
#
# Configurable via environment variables (set before sbatch):
#   N_TRAIN_STEPS   default 5000
#   MAX_BENCH_STEPS default 200
#   INFER_BBOX      "left bottom right top" in the LR raster's CRS units.
#                   If unset, inference runs on the FULL LR raster — only
#                   feasible if it's small.
#
# Multi-GPU smoke: change --gres above to gpu:h100:2 (or more). The
# torchrun line auto-scales via $SLURM_GPUS_ON_NODE.
# ============================================================

set -e
set -o pipefail
mkdir -p logs

module load py-pytorch/2.4.1_py312

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"
E2E_DIR="${SCRATCH_DIR}/e2e_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$E2E_DIR" "${SCRATCH_DIR}/patches"
cd "$PROJECT_DIR"

N_TRAIN_STEPS=${N_TRAIN_STEPS:-5000}
MAX_BENCH_STEPS=${MAX_BENCH_STEPS:-200}
FULL_LR="data_for_model/MorroBay_all_2020_interp_32m_converted.tif"

echo "======================================================================"
echo "E2E smoke test"
echo "  PROJECT_DIR     : $PROJECT_DIR"
echo "  SCRATCH_DIR     : $SCRATCH_DIR"
echo "  E2E_DIR         : $E2E_DIR"
echo "  N_TRAIN_STEPS   : $N_TRAIN_STEPS"
echo "  MAX_BENCH_STEPS : $MAX_BENCH_STEPS"
echo "  GPUs            : ${SLURM_GPUS_ON_NODE:-1}"
echo "  GPU model       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Branch          : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "======================================================================"
echo

# ============================================================
# Phase 1 — Preprocessing
# ============================================================
PATCHES_DIR="${SCRATCH_DIR}/patches"
echo "### Phase 1: preprocessing"
if [ -d "${PATCHES_DIR}/train" ] && [ -n "$(ls -A ${PATCHES_DIR}/train 2>/dev/null)" ]; then
    echo "  (cache hit — using existing ${PATCHES_DIR})"
    echo "  train: $(ls ${PATCHES_DIR}/train/ | wc -l) patches"
    echo "  val  : $(ls ${PATCHES_DIR}/val/ 2>/dev/null | wc -l) patches"
    echo "  test : $(ls ${PATCHES_DIR}/test/ 2>/dev/null | wc -l) patches"
else
    python -m preprocessing.extract_paired_patches \
        --lr_path "$FULL_LR" \
        --hr_dir  "data_for_model/" \
        --out_dir "${PATCHES_DIR}" \
        --patch_size 512 --stride 256 --sr_ratio 16
fi
echo

# ============================================================
# Phase 2 — Three-way precision benchmark
# ============================================================
echo "### Phase 2: three-way precision benchmark (${MAX_BENCH_STEPS} steps each)"
BENCH_SUMMARY="${E2E_DIR}/bench_summary.txt"
: > "$BENCH_SUMMARY"
for CFG in "bf16_compile:bf16:--compile" \
           "bf16_only:bf16:--no-compile" \
           "fp32_only:fp32:--no-compile"; do
    NAME="${CFG%%:*}";  REST="${CFG#*:}"
    PREC="${REST%%:*}"; COMP="${REST#*:}"
    BENCH_OUT="${E2E_DIR}/bench_${NAME}"
    rm -rf "$BENCH_OUT"
    echo "  --- ${NAME} (precision=${PREC}, ${COMP}) ---"

    # Reset peak GPU memory accounting for a clean measurement.
    nvidia-smi --query-gpu=memory.used --format=csv,noheader > /dev/null
    START=$SECONDS

    # Single-GPU benchmark (no torchrun — keeps step accounting simple).
    python -m training.train \
        --patches_dir "$PATCHES_DIR" \
        --out_dir     "$BENCH_OUT" \
        --batch_size  4 \
        --n_epochs    99 \
        --max_steps   "$MAX_BENCH_STEPS" \
        --log_every   20 \
        --precision   "$PREC" \
        $COMP 2>&1 | tee "${E2E_DIR}/bench_${NAME}.log"

    ELAPSED=$((SECONDS - START))
    STEPS_PER_SEC=$(awk "BEGIN { printf \"%.2f\", $MAX_BENCH_STEPS / $ELAPSED }")
    PEAK_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)
    echo "  ${NAME}: ${MAX_BENCH_STEPS} steps in ${ELAPSED}s = ${STEPS_PER_SEC} steps/sec, peak ${PEAK_MEM}" \
        | tee -a "$BENCH_SUMMARY"
done
echo
echo "Benchmark summary:"
cat "$BENCH_SUMMARY"
echo

# ============================================================
# Phase 3 — Real training
# ============================================================
echo "### Phase 3: real training (${N_TRAIN_STEPS} steps, bf16 + compile)"
TRAIN_OUT="${E2E_DIR}/training"
torchrun \
    --standalone \
    --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" \
    --module training.train \
    --patches_dir "$PATCHES_DIR" \
    --out_dir     "$TRAIN_OUT" \
    --batch_size  16 \
    --n_epochs    9999 \
    --max_steps   "$N_TRAIN_STEPS" \
    --log_every   100 \
    --precision   bf16 \
    --compile     2>&1 | tee "${E2E_DIR}/training.log"
echo

CKPT="${TRAIN_OUT}/checkpoints/ckpt_final.pt"
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -t "${TRAIN_OUT}/checkpoints/"ckpt_*.pt 2>/dev/null | head -1)
fi
echo "Using checkpoint: $CKPT"
echo

# ============================================================
# Phase 4 — Inference on a small bbox (or the full grid)
# ============================================================
echo "### Phase 4: inference"
if [ -n "${INFER_BBOX:-}" ]; then
    # Slice out the requested sub-region.
    LR_INPUT="${E2E_DIR}/lr_smoke.tif"
    echo "  Slicing LR to bbox: $INFER_BBOX"
    gdalwarp -te $INFER_BBOX -of GTiff "$FULL_LR" "$LR_INPUT"
else
    LR_INPUT="$FULL_LR"
    echo "  No INFER_BBOX set — running on FULL LR raster (size:"
    gdalinfo "$FULL_LR" | grep -E "Size is|Pixel Size"
    echo "  ); this may take a while."
fi

REAL_DIR="${E2E_DIR}/realizations"
python -m inference.generate_realizations \
    --checkpoint     "$CKPT" \
    --lr_tif         "$LR_INPUT" \
    --out_dir        "$REAL_DIR" \
    --n_realizations 50 \
    2>&1 | tee "${E2E_DIR}/inference.log"
echo

# ============================================================
# Phase 5 — Uncertainty maps
# ============================================================
echo "### Phase 5: uncertainty maps"
UNCERT_DIR="${E2E_DIR}/uncertainty"
python -m inference.uncertainty_maps \
    --realizations_dir "$REAL_DIR" \
    --out_dir          "$UNCERT_DIR" \
    --priority_fraction 0.10 \
    2>&1 | tee "${E2E_DIR}/uncertainty.log"
echo

# ============================================================
# Phase 6 — Evaluation on held-out test patches
# ============================================================
echo "### Phase 6: evaluation on held-out test patches"
EVAL_DIR="${E2E_DIR}/eval"
python -m evaluation.evaluate \
    --checkpoint     "$CKPT" \
    --test_patches   "${PATCHES_DIR}/test" \
    --out_dir        "$EVAL_DIR" \
    --n_realizations 50 \
    2>&1 | tee "${E2E_DIR}/eval.log"
echo

# ============================================================
# Phase 7 — Sample-diversity check (mode-collapse sanity)
# ============================================================
echo "### Phase 7: sample-diversity check on held-out test patches"
python - <<PY 2>&1 | tee "${E2E_DIR}/diversity.log"
"""Compute per-patch inter-realization std on the held-out test set.

The mode-collapse failure mode of the pre-rewrite GAN/DDPM was that
realizations were near-identical. We now expect non-trivial spread —
at minimum, std should be a meaningful fraction of the ground-truth
local elevation range.
"""
import os, glob, json
import numpy as np
import torch

torch.manual_seed(0)
from models.diffusion import build_rectified_flow_from_checkpoint
from preprocessing.data_utils import denormalize_patch
from training.config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = Config()
flow = build_rectified_flow_from_checkpoint('${CKPT}', cfg, device)

patches = sorted(glob.glob('${PATCHES_DIR}/test/*.npz'))
print(f"Test patches: {len(patches)}")

per_patch = []
for pf in patches[:20]:  # cap at 20 for time
    data = np.load(pf)
    lr_raw_norm = data['lr_raw'].squeeze()
    hr_norm     = data['hr'].squeeze()
    mean_elev, std_elev = float(data['stats'][0]), float(data['stats'][1])

    lr_t = torch.from_numpy(lr_raw_norm[np.newaxis, np.newaxis]).float().to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
        samples = flow.sample(lr_t, n_realizations=50, cfg_scale=cfg.cfg_scale)
    samples_np = samples.squeeze(1).cpu().float().numpy()
    samples_m = denormalize_patch(samples_np, mean_elev, std_elev)
    hr_m = denormalize_patch(hr_norm, mean_elev, std_elev)

    # Per-pixel std across realizations
    per_pixel_std = samples_m.std(axis=0)
    # Ground truth local elevation range (a scale for "what's the dynamic range")
    gt_range = float(hr_m.max() - hr_m.min())
    per_patch.append({
        'name': os.path.basename(pf),
        'gt_range_m': gt_range,
        'std_min_m':   float(np.nanmin(per_pixel_std)),
        'std_median_m':float(np.nanmedian(per_pixel_std)),
        'std_mean_m':  float(np.nanmean(per_pixel_std)),
        'std_p95_m':   float(np.nanpercentile(per_pixel_std, 95)),
        'std_max_m':   float(np.nanmax(per_pixel_std)),
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

med_med = np.median([p['std_median_m'] for p in per_patch])
print()
print(f"Median-of-medians inter-realization std: {med_med:.3f} m")
print()
if med_med < 0.1:
    print("WARNING: median std < 0.1 m — possible mode collapse. The whole rewrite")
    print("         exists to fix this; investigate before declaring done.")
elif med_med < 1.0:
    print("Note: median std < 1 m. Acceptable for low-relief regions; verify against")
    print("      gt_range for each patch — std should track terrain complexity.")
else:
    print("OK: meaningful inter-realization spread.")

with open('${EVAL_DIR}/diversity.json', 'w') as f:
    json.dump({'per_patch': per_patch, 'median_of_medians_std_m': med_med}, f, indent=2)
PY
echo

echo "======================================================================"
echo "E2E complete. Outputs:"
echo "  $E2E_DIR"
echo "Key files:"
echo "  $BENCH_SUMMARY"
echo "  ${E2E_DIR}/training.log"
echo "  ${E2E_DIR}/eval/metrics.json"
echo "  ${E2E_DIR}/eval/diversity.json"
echo "  ${UNCERT_DIR}/std.tif  (and friends)"
echo "======================================================================"
