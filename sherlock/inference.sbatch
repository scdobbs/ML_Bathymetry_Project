#!/bin/bash
#SBATCH --job-name=bathy_inference
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# ============================================================
# Bathymetry SR — Inference: Generate Stochastic Realizations
#
# Generates N=50 independent HR realizations over the 3 test survey
# regions, then computes uncertainty maps for each region.
# ============================================================

set -e
mkdir -p logs

# Prevent conda/PYTHONPATH from shadowing modules
conda deactivate 2>/dev/null || true
unset PYTHONPATH

# Load modules and activate venv
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source /home/users/${USER}/bathy_diffusion_env/bin/activate

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"

CHECKPOINT="${SCRATCH_DIR}/training/checkpoints/ckpt_final.pt"

cd "$PROJECT_DIR"

echo "Generating realizations from checkpoint: $CHECKPOINT"

python3 inference/generate_realizations.py \
    --checkpoint     "$CHECKPOINT" \
    --lr_tif         "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --out_dir        "${SCRATCH_DIR}/realizations" \
    --test_surveys \
    --n_realizations 50 \
    --ddim_steps     50 \
    --patch_size     64 \
    --stride         32 \
    --base_ch        64

echo "Generating uncertainty maps for each region..."
for REGION_DIR in "${SCRATCH_DIR}/realizations"/*/; do
    REGION_NAME=$(basename "$REGION_DIR")
    echo "  Processing: $REGION_NAME"
    python3 inference/uncertainty_maps.py \
        --realizations_dir "$REGION_DIR" \
        --out_dir          "${SCRATCH_DIR}/uncertainty_maps/${REGION_NAME}" \
        --priority_fraction 0.10
done

echo "Inference complete."
echo "Realizations: ${SCRATCH_DIR}/realizations/"
echo "Uncertainty maps: ${SCRATCH_DIR}/uncertainty_maps/"
