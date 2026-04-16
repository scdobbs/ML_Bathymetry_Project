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
# Generates N=50 independent HR realizations over the full 32m grid.
# Each realization is saved as a GeoTIFF at 2m resolution.
# Use uncertainty_maps.py after this to compute mean/std/priority.
#
# Note: For very large grids, consider splitting into subregions.
# ============================================================

set -e
mkdir -p logs

module load py-pytorch/2.4.1_py312

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"

# Set checkpoint path — update this after training
CHECKPOINT="${SCRATCH_DIR}/training/checkpoints/ckpt_final.pt"

cd "$PROJECT_DIR"

echo "Generating realizations from checkpoint: $CHECKPOINT"

python inference/generate_realizations.py \
    --checkpoint     "$CHECKPOINT" \
    --lr_tif         "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --out_dir        "${SCRATCH_DIR}/realizations" \
    --n_realizations 50 \
    --ddim_steps     50 \
    --patch_size     64 \
    --stride         32 \
    --base_ch        64

echo "Generating uncertainty maps..."
python inference/uncertainty_maps.py \
    --realizations_dir "${SCRATCH_DIR}/realizations" \
    --out_dir          "${SCRATCH_DIR}/uncertainty_maps" \
    --priority_fraction 0.10

echo "Inference complete."
echo "Uncertainty maps: ${SCRATCH_DIR}/uncertainty_maps"
