#!/bin/bash
#SBATCH --job-name=bathy_preprocess
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# ============================================================
# Bathymetry SR — Data Preprocessing
# Extracts paired LR/HR patches from overlapping survey areas.
# Run once before training.
# ============================================================

set -e
mkdir -p logs

# Load Sherlock PyTorch module
module load py-pytorch/2.4.1_py312

# Project root (adjust to your Sherlock home/scratch path)
PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"

cd "$PROJECT_DIR"

python -m preprocessing.extract_paired_patches \
    --lr_path  "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --hr_dir   "data_for_model/" \
    --out_dir  "${SCRATCH_DIR}/patches" \
    --patch_size 64 \
    --stride 32 \
    --nodata_threshold 0.10 \
    --flat_threshold 0.05

echo "Preprocessing complete. Patches saved to ${SCRATCH_DIR}/patches"
