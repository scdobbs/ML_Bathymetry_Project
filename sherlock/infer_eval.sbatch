#!/bin/bash
#SBATCH --job-name=bathy_infer_eval
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/infer_eval_%j.out
#SBATCH --error=logs/infer_eval_%j.err

# ============================================================
# Inference + evaluation against a trained checkpoint.
#
# Defaults to the H100 production run's ckpt_final.pt.  Override:
#   CKPT=/path/to/ckpt.pt sbatch sherlock/infer_eval.sbatch
#
# Runs:
#   Phase 4 — tiled inference over 3 LR datasets (per-LR smoke bboxes)
#   Phase 5 — uncertainty maps per LR
#   Phase 6 — evaluation on MAX_PATCHES test patches → metrics + calibration
#   Phase 7 — stratified sample-diversity check across the 5 test surveys
#
# Each phase writes into a fresh timestamped subdir under SCRATCH_DIR.
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
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"
CKPT="${CKPT:-${SCRATCH_DIR}/training_h100/checkpoints/ckpt_final.pt}"
RUN_TAG="${RUN_TAG:-h100}"
OUT_DIR="${SCRATCH_DIR}/infer_eval_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"

PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512"
SHARED_LR_DIR="/scratch/groups/hilley/bathy_diffusion_2026/lr_tifs"
LR_MORROBAY="${SHARED_LR_DIR}/MorroBay_all_2020_interp_32m_converted.tif"
LR_NORCAL="${SHARED_LR_DIR}/norcal-to-triple-junction-32m.tif"
LR_SB="${SHARED_LR_DIR}/Santa_Barbara_32m-clipped2.tif"

# Per-LR smoke bboxes (each in its LR's own CRS). Use "FULL" to run on
# the entire LR grid.
INFER_BBOX_MORROBAY=${INFER_BBOX_MORROBAY:-"621275 3928924 626275 3933924"}
INFER_BBOX_NORCAL=${INFER_BBOX_NORCAL:-"509319 4198253 514319 4203253"}
INFER_BBOX_SB=${INFER_BBOX_SB:-"764530 3814620 769530 3819620"}

N_REALIZATIONS=${N_REALIZATIONS:-10}      # Phase 4 + Phase 6
MAX_PATCHES=${MAX_PATCHES:-200}            # Phase 6 cap

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "Inference + evaluation"
echo "  Host         : $(hostname)"
echo "  Checkpoint   : $CKPT"
echo "  PATCHES_DIR  : $PATCHES_DIR"
echo "  OUT_DIR      : $OUT_DIR"
echo "  N_REALIZATIONS = $N_REALIZATIONS"
echo "  MAX_PATCHES    = $MAX_PATCHES"
echo "  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Branch       : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "  Started      : $(date)"
echo "======================================================================"

[ ! -f "$CKPT" ] && { echo "ERROR: checkpoint not found at $CKPT" >&2; exit 1; }

# ============================================================
# Phase 4 — Inference per LR
# ============================================================
echo "### Phase 4: inference (per-LR loop)"
LR_NAMES=(MorroBay NorCal SantaBarbara)
LR_PATHS=("$LR_MORROBAY" "$LR_NORCAL" "$LR_SB")
LR_BBOXES=("$INFER_BBOX_MORROBAY" "$INFER_BBOX_NORCAL" "$INFER_BBOX_SB")

for i in "${!LR_NAMES[@]}"; do
    NAME="${LR_NAMES[$i]}"
    LR_PATH="${LR_PATHS[$i]}"
    BBOX="${LR_BBOXES[$i]}"

    echo "  --- ${NAME} (LR=$(basename $LR_PATH)) ---"
    if [ "$BBOX" = "FULL" ] || [ -z "$BBOX" ]; then
        LR_INPUT="$LR_PATH"
        echo "    Using FULL LR raster."
    else
        LR_INPUT="${OUT_DIR}/lr_smoke_${NAME}.tif"
        echo "    Slicing LR to bbox: $BBOX"
        python3 - "$LR_PATH" "$LR_INPUT" $BBOX <<'PY'
import sys, rasterio
from rasterio.windows import from_bounds
src_path, dst_path = sys.argv[1], sys.argv[2]
xmin, ymin, xmax, ymax = map(float, sys.argv[3:7])
with rasterio.open(src_path) as src:
    win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
    data = src.read(1, window=win)
    profile = src.profile.copy()
    profile.update(height=data.shape[0], width=data.shape[1],
                   transform=src.window_transform(win),
                   compress='deflate')
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(data, 1)
print(f"    Wrote {dst_path}  ({data.shape[1]}x{data.shape[0]} pixels)")
PY
    fi

    REAL_DIR="${OUT_DIR}/realizations_${NAME}"
    python3 -m inference.generate_realizations \
        --checkpoint     "$CKPT" \
        --lr_tif         "$LR_INPUT" \
        --out_dir        "$REAL_DIR" \
        --n_realizations "$N_REALIZATIONS" \
        2>&1 | tee "${OUT_DIR}/inference_${NAME}.log"
done
echo

# ============================================================
# Phase 5 — Uncertainty maps per LR
# ============================================================
echo "### Phase 5: uncertainty maps (per-LR loop)"
for NAME in "${LR_NAMES[@]}"; do
    REAL_DIR="${OUT_DIR}/realizations_${NAME}"
    UNCERT_DIR="${OUT_DIR}/uncertainty_${NAME}"
    echo "  --- ${NAME} ---"
    python3 -m inference.uncertainty_maps \
        --realizations_dir "$REAL_DIR" \
        --out_dir          "$UNCERT_DIR" \
        --priority_fraction 0.10 \
        2>&1 | tee "${OUT_DIR}/uncertainty_${NAME}.log"
done
echo

# ============================================================
# Phase 6 — Evaluation on test patches (with calibration plot)
# ============================================================
echo "### Phase 6: evaluation on test patches (max ${MAX_PATCHES} patches)"
EVAL_DIR="${OUT_DIR}/eval"
python3 -m evaluation.evaluate \
    --checkpoint     "$CKPT" \
    --test_patches   "${PATCHES_DIR}/test" \
    --out_dir        "$EVAL_DIR" \
    --n_realizations "$N_REALIZATIONS" \
    --max_patches    "$MAX_PATCHES" \
    2>&1 | tee "${OUT_DIR}/eval.log"
echo

# ============================================================
# Phase 7 — Sample-diversity check (stratified, 5×4 patches)
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
    # Chunk 50 realizations into 5×10 to fit GPU memory under CFG (which doubles batch).
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

os.makedirs('${EVAL_DIR}', exist_ok=True)
with open('${EVAL_DIR}/diversity.json', 'w') as f:
    json.dump({'per_patch': per_patch, 'median_of_medians_std_m': med_med}, f, indent=2)
print(f"Saved: ${EVAL_DIR}/diversity.json")
PY

echo
echo "Done: $(date)"
echo "Outputs at: $OUT_DIR"
