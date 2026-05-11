#!/bin/bash
#SBATCH --job-name=bathy_inference
#SBATCH -p serc
# H100 is the production target. Swap to --gres=gpu:1 if the h100
# selector isn't available on serc (matching train.sbatch's choice).
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/inference_%A_%a.out
#SBATCH --error=logs/inference_%A_%a.err
#
# Optional array — parallelize inference across regions by uncommenting
# the line below and adjusting the range. Each array task processes a
# different bounding box drawn from the BBOXES variable.
# #SBATCH --array=0-3

# ============================================================
# Bathymetry SR — generate stochastic realizations.
#
# Single-job mode (default): runs inference over the full LR grid.
# Array mode: split the LR grid into sub-regions, one per array task.
#   Set --array=0-N above and define an N+1-entry BBOXES list below.
#   Each entry is "left bottom right top" in the LR raster's CRS units.
# ============================================================

set -e
mkdir -p logs

module load py-pytorch/2.4.1_py312

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"

CHECKPOINT="${SCRATCH_DIR}/training/checkpoints/ckpt_final.pt"
FULL_LR="data_for_model/MorroBay_all_2020_interp_32m_converted.tif"

cd "$PROJECT_DIR"

# ---- Region selection ----
# If SLURM_ARRAY_TASK_ID is set, slice the input into N sub-regions. The
# default behavior (no array) processes the full LR raster.
#
# Replace these bboxes with Steve's actual sub-region coordinates.
#   format: "left bottom right top" — in the LR raster's projected CRS.
BBOXES=(
    # tile 0: SW quadrant
    "0      0       1500    1500"
    # tile 1: SE quadrant
    "1500   0       3000    1500"
    # tile 2: NW quadrant
    "0      1500    1500    3000"
    # tile 3: NE quadrant
    "1500   1500    3000    3000"
)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    BBOX="${BBOXES[$SLURM_ARRAY_TASK_ID]}"
    OUT_DIR="${SCRATCH_DIR}/realizations/tile_${SLURM_ARRAY_TASK_ID}"
    LR_INPUT="${SCRATCH_DIR}/lr_subregions/region_${SLURM_ARRAY_TASK_ID}.tif"
    if [ ! -f "$LR_INPUT" ]; then
        echo "Slicing LR for region $SLURM_ARRAY_TASK_ID (bbox: $BBOX)"
        mkdir -p "$(dirname "$LR_INPUT")"
        # gdalwarp wants -te left bottom right top
        gdalwarp -te $BBOX -of GTiff "$FULL_LR" "$LR_INPUT"
    fi
else
    OUT_DIR="${SCRATCH_DIR}/realizations"
    LR_INPUT="$FULL_LR"
fi

echo "Generating realizations from checkpoint: $CHECKPOINT"
echo "LR input: $LR_INPUT"
echo "Output:   $OUT_DIR"

python -m inference.generate_realizations \
    --checkpoint     "$CHECKPOINT" \
    --lr_tif         "$LR_INPUT" \
    --out_dir        "$OUT_DIR" \
    --n_realizations 50

echo "Generating uncertainty maps..."
python -m inference.uncertainty_maps \
    --realizations_dir "$OUT_DIR" \
    --out_dir          "${OUT_DIR}/uncertainty" \
    --priority_fraction 0.10

echo "Inference complete."
