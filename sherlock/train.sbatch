#!/bin/bash
#SBATCH --job-name=bathy_train
#SBATCH -p serc
# H100 is the production target. If serc doesn't expose an `h100` GRES
# selector, swap the next line for `--gres=gpu:1` and let SLURM pick.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================
# Bathymetry SR — rectified-flow training
#
# Launches via torchrun so the same script scales from 1 GPU
# (WORLD_SIZE=1, the default with --gres=gpu:1) to N GPUs without
# code changes. SLURM_GPUS_ON_NODE is set automatically by --gres.
# ============================================================

set -e
mkdir -p logs

module load py-pytorch/2.4.1_py312

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"

cd "$PROJECT_DIR"

echo "Starting training on $(hostname) with GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Patches: ${SCRATCH_DIR}/patches"
echo "Output:  ${SCRATCH_DIR}/training"
echo "GPUs on node: ${SLURM_GPUS_ON_NODE:-1}"

torchrun \
    --standalone \
    --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" \
    --module training.train \
    --patches_dir "${SCRATCH_DIR}/patches" \
    --out_dir     "${SCRATCH_DIR}/training" \
    --batch_size  16 \
    --n_epochs    500 \
    --base_ch     64

echo "Training complete."
