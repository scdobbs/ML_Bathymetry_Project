#!/bin/bash
#SBATCH --job-name=bathy_train
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================
# Bathymetry SR — Diffusion Model Training
#
# Expected runtime: ~24–48 hours for 500 epochs on a V100/A100.
# The script auto-resumes from the latest checkpoint if restarted.
# Monitor progress: tensorboard --logdir $OUT_DIR/logs/
# ============================================================

set -e
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate bathy_diffusion

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"

cd "$PROJECT_DIR"

echo "Starting training on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Patches: ${SCRATCH_DIR}/patches"
echo "Output:  ${SCRATCH_DIR}/training"

python training/train.py \
    --patches_dir "${SCRATCH_DIR}/patches" \
    --out_dir     "${SCRATCH_DIR}/training" \
    --batch_size  16 \
    --n_epochs    500 \
    --base_ch     64

echo "Training complete."
